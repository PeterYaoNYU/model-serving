from collections import OrderedDict
from typing import Any, TypedDict, cast

import torch
import transformers
from punica.models.llama_lora import (
    BatchedLlamaLoraWeight,
    LlamaConfig,
    LlamaForCausalLMWithLora,
    LlamaLoraWeight,
)
from punica.utils.cat_tensor import BatchLenInfo
from punica.utils.kvcache import BatchedKvCache, KvCache, KvPool


class RequestContext:
    def __init__(
        self,
        input_ids: list[int],
        kvpool: KvPool,
        lora_id: str,
        tokenizer,
        *,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        maxlen: int,
        stop_token_id: int,
    ):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.maxlen = maxlen
        self.stop_token_id = stop_token_id

        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(temperature)
            )
        if repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
        if 0 < top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(top_p))
        if top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(top_k))

        self.output_ids = [int(x) for x in input_ids]
        self.prompt_len = len(self.output_ids)
        self.kvcache = KvCache(kvpool, self.prompt_len)
        self.lora_id = lora_id
        self.tokenizer = tokenizer
        self.prefix_offset = 0
        self.read_offset = 0

    def get_next_token_id(self, logits: torch.Tensor) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([self.output_ids], device=logits.device)
            else:
                t = None
            last_token_logits = self.logits_processor(t, logits[-1].unsqueeze(0))[0]
        else:
            last_token_logits = logits[-1, :]

        if self.temperature <= 0 or self.top_p <= 0:
            _, indices = torch.topk(last_token_logits, 2)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
        token = int(indices.tolist()[0])
        return token

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    def is_stop(self) -> int:
        if len(self.output_ids) >= self.maxlen:
            return True
        if self.output_ids[-1] == self.stop_token_id:
            return True
        return False

    def is_prefill(self) -> bool:
        return len(self.output_ids) == self.prompt_len

    def decode_tokens(self) -> str:
        # Adapted from: https://github.com/huggingface/text-generation-inference/blob/a5def7c222174e03d815f890093584f3e815c5ce/server/text_generation_server/models/model.py#L68
        prefix_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset : self.read_offset],
            skip_special_tokens=True,
        )
        new_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset :], skip_special_tokens=True
        )
        if len(new_text) > len(prefix_text) and not new_text.endswith("\uFFFD"):
            new_text = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.output_ids)
            return new_text
        else:
            return ""


class TextGenerationChunk(TypedDict):
    index: int
    token_id: int
    text: str
    is_stop: bool


class BatchedTextGeneration:
    def __init__(
        self,
        model_name_or_path: str | None,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.lora_rank = lora_rank
        self.dtype = dtype
        self.device = device

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLMWithLora.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, torch_dtype=dtype
        ).to(device)  # type: ignore

        self.model_config = cast(LlamaConfig, self.model.config)
        self.kvpool = KvPool(
            num_layers=self.model_config.num_hidden_layers,
            num_heads=self.model_config.num_attention_heads,
            head_dim=self.model_config.hidden_size
            // self.model_config.num_attention_heads,
            page_len=16,
            dtype=self.dtype,
            device=self.device,
        )

        self.lora_cache_size = 1
        self.lora_weights: OrderedDict[Any, LlamaLoraWeight] = OrderedDict()
        self.empty_lora_id = "<empty>"
        self.lora_weights[self.empty_lora_id] = LlamaLoraWeight(
            self.model_config, lora_rank, self.dtype, self.device
        )
        self.reqctx: dict[Any, RequestContext] = {}

    @property
    def use_lora(self) -> bool:
        return self.max_lora_rank > 0

    def load_lora_weight(
        self, lora_id: Any, weight_or_path: str | dict[str, torch.Tensor]
    ):
        if isinstance(weight_or_path, str):
            tmp = torch.load(
                weight_or_path, map_location=self.device, weights_only=True
            )
        elif isinstance(weight_or_path, dict):
            tmp = weight_or_path
        else:
            raise TypeError(
                f"weight_or_path must be str or dict[str, torch.Tensor], but got {type(weight_or_path)}"
            )
        lora_rank = tmp["q.A"].size(1)
        if lora_rank != self.lora_rank:
            raise ValueError(
                f"lora_rank of {lora_id} is {lora_rank}, but expected {self.lora_rank}"
            )
        lora_weight = LlamaLoraWeight(
            self.model_config, lora_rank, self.dtype, self.device
        )
        lora_weight.copy_from_tensors(tmp)
        self.lora_weights[lora_id] = lora_weight
        self._update_lora_lru(lora_id)

    def _update_lora_lru(self, lora_id: Any):
        if len(self.reqctx) + 2 > self.lora_cache_size:
            # +2: One for empty. One for incoming request.
            self.lora_cache_size = len(self.reqctx) + 2
        self.lora_weights.move_to_end(lora_id)
        self.lora_weights.move_to_end(self.empty_lora_id)
        while len(self.lora_weights) > self.lora_cache_size:
            self.lora_weights.popitem(last=False)

    def add_request(
        self,
        reqid: Any,
        lora_id: Any,
        input_ids: list[int] | None,
        input_text: str | None,
        *,
        temperature: float = 0.7,
        repetition_penalty: float = 1.1,
        top_p: float = 0.9,
        top_k: int = -1,
        maxlen: int = 4096,
    ):
        if reqid in self.reqctx:
            raise ValueError("Request already exists", reqid)
        if lora_id not in self.lora_weights:
            raise ValueError("Cannot find lora weights", lora_id)
        if input_text is None:
            if input_ids is None:
                raise ValueError("`input_ids` or `input_text` must be provided")
        else:
            if input_ids is not None:
                raise ValueError("only one of `input_ids` or `input_text` can be used")
            input_ids = self.tokenizer.encode(input_text)

        self._update_lora_lru(lora_id)
        self.reqctx[reqid] = RequestContext(
            input_ids,
            self.kvpool,
            lora_id,
            self.tokenizer,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            maxlen=min(maxlen, 4096),
            stop_token_id=self.tokenizer.eos_token_id,
        )

    def _delete_request(self, reqid: Any):
        reqctx = self.reqctx.pop(reqid)
        reqctx.kvcache.release()

    def cancel_request(self, reqid: Any):
        self._delete_request(reqid)

    def has_request(self):
        return len(self.reqctx)>0

    def step(self) -> dict[Any, TextGenerationChunk]:
        if not self.reqctx:
            return {}

        # Put prefill requests first, then sort by lora_id.
        reqs = sorted(
            self.reqctx.items(),
            key=lambda kv: (not kv[1].is_prefill(), kv[1].lora_id),
        )

        # Gather batch
        prefill_input_ids, prefill_lens, prefill_kv = [], [], []
        decode_input_ids, decode_kv = [], []
        lora_ids, lora_lens = [], []
        for _, reqctx in reqs:
            if reqctx.is_prefill():
                prefill_input_ids.extend(reqctx.output_ids)
                prefill_lens.append(len(reqctx.output_ids))
                prefill_kv.append(reqctx.kvcache)
            else:
                decode_input_ids.append(reqctx.output_ids[-1])
                decode_kv.append(reqctx.kvcache)
                reqctx.kvcache.acquire_one()
            if lora_ids and lora_ids[-1] == reqctx.lora_id:
                lora_lens[-1] += 1
            else:
                lora_ids.append(reqctx.lora_id)
                lora_lens.append(1)

        # Run model
        input_ids = torch.tensor(
            prefill_input_ids + decode_input_ids,
            dtype=torch.long,
            device=self.device,
        )
        blen = BatchLenInfo(prefill_lens, len(decode_input_ids), self.device)
        prefill_kv = BatchedKvCache(prefill_kv) if prefill_kv else None
        decode_kv = BatchedKvCache(decode_kv) if decode_kv else None
        lora = BatchedLlamaLoraWeight(
            [self.lora_weights[id] for id in lora_ids], lora_lens
        )
        logits, _ = self.model(input_ids, blen, prefill_kv, decode_kv, lora)
        if prefill_kv:
            if decode_kv:
                logits = torch.cat([logits[blen.indptr[1:] - 1], logits[blen.doff :]])
            else:
                logits = logits[blen.indptr[1:] - 1]

        # Postprocess
        out = {}
        for i, (reqid, reqctx) in enumerate(reqs):
            next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
            reqctx.append_token(next_token_id)
            text = reqctx.decode_tokens()
            out[reqid] = TextGenerationChunk(
                index=len(reqctx.output_ids) - 1,
                token_id=next_token_id,
                text=text,
                is_stop=reqctx.is_stop(),
            )
            if reqctx.is_stop():
                self._delete_request(reqid)

        return out
