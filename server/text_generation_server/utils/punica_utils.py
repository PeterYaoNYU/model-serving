from collections.abc import Sequence
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


class KvPool:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_len = page_len
        self.dtype = dtype
        self.device = device
        self._page_shape = (num_layers, 2, num_heads, page_len, head_dim)
        self._page_meta = torch.empty(self._page_shape, dtype=dtype, device="meta")
        self._allocated = set()

    @property
    def page_meta(self) -> torch.Tensor:
        return self._page_meta

    @property
    def page_nbytes(self) -> int:
        return self._page_meta.nbytes

    @property
    def num_pages(self) -> int:
        return len(self._allocated)

    def allocated_pages(self) -> Iterable[torch.Tensor]:
        return iter(self._allocated)

    def alloc_page(self) -> torch.Tensor:
        page = torch.empty(self._page_shape, dtype=self.dtype, device=self.device)
        self._allocated.add(page)
        return page

    def free_page(self, page: torch.Tensor) -> None:
        self._allocated.remove(page)


class GrowableTensor:
    def __init__(
        self,
        data: Sequence[Any],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        self._buf = torch.tensor(data, dtype=dtype, device=device)
        self._len = len(data)

    def view(self) -> torch.Tensor:
        return self._buf[: self._len]

    def append(self, data: Any) -> None:
        self._maybe_grow(self._len + 1)
        self._buf[self._len] = data
        self._len += 1

    def clear(self) -> None:
        self._len = 0

    @staticmethod
    def _next_power_of_two(x: int) -> int:
        return 1 << (x - 1).bit_length()

    def _maybe_grow(self, capacity: int) -> None:
        if self._buf.numel() >= capacity:
            return
        new_capacity = self._next_power_of_two(capacity)
        new_buf = torch.empty(
            new_capacity, dtype=self._buf.dtype, device=self._buf.device
        )
        new_buf[: self._len] = self._buf
        self._buf = new_buf


class KvCache:
    """Key-value cache for one sequence."""

    def __init__(self, pool: KvPool, init_len: int):
        if init_len < 0:
            raise ValueError("init_len must be non-negative")

        self._pool = pool
        if init_len > 0:
            npages = (init_len + pool.page_len - 1) // pool.page_len
            self._pages = [pool.alloc_page() for _ in range(npages)]
            self._seqlen = init_len
        else:
            self._pages = []
            self._seqlen = 0
        self._ptrs = GrowableTensor(
            [t.data_ptr() for t in self._pages], dtype=torch.int64, device=pool.device
        )

    @property
    def pool(self) -> KvPool:
        return self._pool

    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def ptrs(self) -> torch.Tensor:
        return self._ptrs.view()

    @property
    def pages(self) -> Sequence[torch.Tensor]:
        return self._pages

    @property
    def num_pages(self) -> int:
        return len(self._pages)

    def acquire_one(self):
        """Reserve space for one more token"""
        last_page_offset = (self._seqlen - 1) % self._pool.page_len + 1
        if last_page_offset == self._pool.page_len:
            self._pages.append(self._pool.alloc_page())
            self._ptrs.append(self._pages[-1].data_ptr())
        self._seqlen += 1

    def release(self):
        """Release all pages"""
        self._seqlen = 0
        for page in self._pages:
            self._pool.free_page(page)
        self._pages.clear()
        self._ptrs.clear()


class BatchedKvCache:
    """Key-value cache for a batch of sequences."""

    def __init__(self, kv: Sequence[KvCache]):
        assert len(kv) > 0
        pool = kv[0].pool
        device = pool.device
        ptrs = []
        indptr = [0]
        last_page_offset = []
        for c in kv:
            assert c.pool is pool
            assert c.num_pages > 0
            ptrs.append(c.ptrs)
            indptr.append(indptr[-1] + c.num_pages)
            last_page_offset.append((c.seqlen - 1) % pool.page_len + 1)

        self.pool = pool
        self.ptrs = torch.cat(ptrs)
        self.indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
        self.last_page_offset = torch.tensor(
            last_page_offset, dtype=torch.int32, device=device
        )


class BatchLenInfo:
    def __init__(
        self,
        prefills: Sequence[int],
        decode: int,
        indptr_device: torch.device,
        indptr_dtype: torch.dtype = torch.int32,
    ):
        tmp = [0]
        tmp.extend(prefills)
        self._prefills = tmp[1:]
        self._decode = decode
        if len(prefills) > 0:
            cumsum = np.cumsum(tmp)
            self._indptr = torch.tensor(
                cumsum, dtype=indptr_dtype, device=indptr_device
            )
            self._doff = cumsum[-1]
        else:
            self._indptr = None
            self._doff = 0

    @property
    def prefills(self) -> list[int]:
        """Length of each prefill request."""
        return self._prefills

    @property
    def decode(self) -> int:
        """Number of decode requests."""
        return self._decode

    @property
    def doff(self) -> int:
        """Index of the first decode request. Equivalently, total length of prefills."""
        return self._doff

    @property
    def indptr(self) -> torch.Tensor | None:
        """`indptr[i] := sum(prefills[:i])`. None if no prefill."""
        return self._indptr


class LoraWeight:
    def __init__(
        self,
        num_layers: int,
        in_features: int,
        out_features: int,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # SGMV-Shrink custom CUDA kernel uses column-major.
        self.wa = torch.zeros(
            (num_layers, lora_rank, in_features), dtype=dtype, device=device
        )
        # SGMV-Expand cutlass kernel uses row-major.
        self.wb = torch.zeros(
            (num_layers, lora_rank, out_features), dtype=dtype, device=device
        )

    def copy_from_tensor(self, a: torch.Tensor, b: torch.Tensor):
        """
        Copy from column-major weight tensors.

        Args:
          a: Shape: `[num_layers, lora_rank, in_features]`.
          b: Shape: `[num_layers, out_features, lora_rank]`.
        """
        self.wa.copy_(a.to(self.wa.device).to(self.wa.dtype))
        self.wb.copy_(b.to(self.wb.device).to(self.wb.dtype).transpose(1, 2))

    @property
    def device(self) -> torch.device:
        return self.wa.device

    @property
    def dtype(self) -> torch.dtype:
        return self.wa.dtype

    @property
    def num_layers(self) -> int:
        return self.wa.size(0)

    @property
    def in_features(self) -> int:
        return self.wa.size(2)

    @property
    def out_features(self) -> int:
        return self.wb.size(2)

    @property
    def lora_rank(self) -> int:
        return self.wa.size(1)


class BatchedLoraWeight:
    def __init__(self, weights: Sequence[LoraWeight]):
        assert len(weights) > 0
        device = weights[0].device
        self.wa_ptr = torch.tensor(
            [w.wa.data_ptr() for w in weights], dtype=torch.int64, device=device
        )
        self.wb_ptr = torch.tensor(
            [w.wb.data_ptr() for w in weights], dtype=torch.int64, device=device
        )
