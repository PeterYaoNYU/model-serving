# Modified from https://github.com/punica-ai/punica/blob/master/src/punica/models/llama_lora.py
# Editor: Junyi Shen

import math
import torch
from torch import device, dtype, nn
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaConfig,
    PreTrainedModel,
    #LlamaRMSNorm,
)
from punica.ops import (
    add_lora_sgmv_custom_cutlass as add_lora,
    append_kv,
    batch_decode,
    batch_prefill,
    init_kv,
    rms_norm,
)
from punica.utils import BatchedKvCache, BatchedLoraWeight, BatchLenInfo, LoraWeight, KvPool, KvCache
#from punica import LlamaForCausalLMWithLora
import peft

import time
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Dict
from text_generation_server.models import Model
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

tracer = trace.get_tracer(__name__)

from .causal_lm import CausalLMBatch
from collections import defaultdict
@dataclass
class PunicaBatch(CausalLMBatch):
    lora_ids = [] #it goes wrong when lora_ids: List[str] = []
    
class LlamaLoraWeight:
    def __init__(
        self,
        config: LlamaConfig,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.q = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.k = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.v = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.o = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.gate = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.up = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.down = LoraWeight(
            config.num_hidden_layers,
            config.intermediate_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )

    def copy_from_tensors(self, ts: dict[str, torch.Tensor]):
        self.q.copy_from_tensor(ts["q.A"], ts["q.B"])
        self.k.copy_from_tensor(ts["k.A"], ts["k.B"])
        self.v.copy_from_tensor(ts["v.A"], ts["v.B"])
        self.o.copy_from_tensor(ts["o.A"], ts["o.B"])
        self.gate.copy_from_tensor(ts["gate.A"], ts["gate.B"])
        self.up.copy_from_tensor(ts["up.A"], ts["up.B"])
        self.down.copy_from_tensor(ts["down.A"], ts["down.B"])


class BatchedLlamaLoraWeight:
    def __init__(self, weights: list[LlamaLoraWeight], lens: list[int]):
        assert len(weights) == len(lens)
        device = weights[0].q.wa.device
        self.q = BatchedLoraWeight([w.q for w in weights])
        self.k = BatchedLoraWeight([w.k for w in weights])
        self.v = BatchedLoraWeight([w.v for w in weights])
        self.o = BatchedLoraWeight([w.o for w in weights])
        self.gate = BatchedLoraWeight([w.gate for w in weights])
        self.up = BatchedLoraWeight([w.up for w in weights])
        self.down = BatchedLoraWeight([w.down for w in weights])
        self.segment = torch.cumsum(
            torch.tensor([0] + lens, dtype=torch.int32, device=device),
            dim=0,
            dtype=torch.int32,
        )
        self.rank = weights[0].q.lora_rank


class LlamaAttentionWithLora(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_qo_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_qo_heads
        self._scale = 1 / math.sqrt(self.head_dim)
        self.layer_idx = layer_idx

        assert self.head_dim * self.num_qo_heads == self.hidden_size
        assert self.num_kv_heads * self.num_kv_groups == self.num_qo_heads
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_qo_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_qo_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
        lora: BatchedLlamaLoraWeight,
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push("qkv_proj")
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lora_qkv")
        add_lora(
            q_proj,
            hidden_states,
            lora.q.wa_ptr,
            lora.q.wb_ptr,
            lora.segment,
            self.layer_idx,
            lora.rank,
        )
        add_lora(
            k_proj,
            hidden_states,
            lora.k.wa_ptr,
            lora.k.wb_ptr,
            lora.segment,
            self.layer_idx,
            lora.rank,
        )
        add_lora(
            v_proj,
            hidden_states,
            lora.v.wa_ptr,
            lora.v.wb_ptr,
            lora.segment,
            self.layer_idx,
            lora.rank,
        )
        torch.cuda.nvtx.range_pop()

        stack_attn_output = []

        if len(blen.prefills) > 0:
            assert prefill_kv is not None
            assert blen.indptr is not None
            q = q_proj[: blen.doff].view(blen.doff, self.num_qo_heads, self.head_dim)
            k = k_proj[: blen.doff].view(blen.doff, self.num_kv_heads, self.head_dim)
            v = v_proj[: blen.doff].view(blen.doff, self.num_kv_heads, self.head_dim)

            torch.cuda.nvtx.range_push("init_kv")
            init_kv(prefill_kv, k, v, blen.indptr, self.layer_idx)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("batch_prefill")
            attn_output = batch_prefill(q, blen.indptr, prefill_kv, self.layer_idx)
            attn_output = attn_output.view(blen.doff, self.hidden_size)
            stack_attn_output.append(attn_output)
            torch.cuda.nvtx.range_pop()

        if blen.decode > 0:
            q = q_proj[blen.doff :].view(blen.decode, self.num_qo_heads, self.head_dim)
            k = k_proj[blen.doff :].view(blen.decode, self.num_kv_heads, self.head_dim)
            v = v_proj[blen.doff :].view(blen.decode, self.num_kv_heads, self.head_dim)

            torch.cuda.nvtx.range_push("append_kv")
            assert decode_kv is not None
            append_kv(decode_kv, k, v, self.layer_idx)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("batch_decode")
            attn_outputs = batch_decode(q, decode_kv, self.layer_idx)
            attn_outputs = attn_outputs.view(blen.decode, self.hidden_size)
            stack_attn_output.append(attn_outputs)
            torch.cuda.nvtx.range_pop()

        if len(stack_attn_output) == 1:
            attn_outputs = stack_attn_output[0]
        else:
            attn_outputs = torch.cat(stack_attn_output, dim=0)

        # output projection
        torch.cuda.nvtx.range_push("o_proj")
        o = self.o_proj(attn_outputs)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lora_o")
        add_lora(
            o,
            attn_outputs,
            lora.o.wa_ptr,
            lora.o.wb_ptr,
            lora.segment,
            self.layer_idx,
            lora.rank,
        )
        torch.cuda.nvtx.range_pop()

        return o


class LlamaMlpWithLora(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        x: torch.Tensor,
        lora: BatchedLlamaLoraWeight,
    ) -> torch.Tensor:
        with torch.cuda.nvtx.range("gate_proj"):
            gate = self.gate_proj(x)
        with torch.cuda.nvtx.range("lora_gate"):
            add_lora(
                gate,
                x,
                lora.gate.wa_ptr,
                lora.gate.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )
        with torch.cuda.nvtx.range("gate_act"):
            gate = self.act_fn(gate)

        with torch.cuda.nvtx.range("up_proj"):
            up = self.up_proj(x)
        with torch.cuda.nvtx.range("lora_up"):
            add_lora(
                up,
                x,
                lora.up.wa_ptr,
                lora.up.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        with torch.cuda.nvtx.range("gate_up"):
            t = gate * up

        with torch.cuda.nvtx.range("down_proj"):
            down = self.down_proj(t)
        with torch.cuda.nvtx.range("lora_down"):
            add_lora(
                down,
                t,
                lora.down.wa_ptr,
                lora.down.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        return down


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class LlamaDecoderLayerWithLora(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttentionWithLora(config, layer_idx)
        self.mlp = LlamaMlpWithLora(config, layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
        lora: BatchedLlamaLoraWeight,
    ) -> torch.Tensor:
        residual = hidden_states

        torch.cuda.nvtx.range_push("input_norm")
        hidden_states = self.input_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Self Attention
        torch.cuda.nvtx.range_push("LlamaAttention")
        hidden_states = self.self_attn(hidden_states, blen, prefill_kv, decode_kv, lora)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("r")
        hidden_states = residual + hidden_states
        torch.cuda.nvtx.range_pop()

        # Fully Connected
        residual = hidden_states
        torch.cuda.nvtx.range_push("norm")
        hidden_states = self.post_attention_layernorm(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("mlp")
        hidden_states = self.mlp(hidden_states, lora)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("r")
        hidden_states = residual + hidden_states
        torch.cuda.nvtx.range_pop()

        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayerWithLora"]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.version",
        r"self_attn\.rotary_emb\.inv_freq",
    ]


class LlamaModelWithLora(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayerWithLora(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
        lora: BatchedLlamaLoraWeight,
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push("embed")
        hidden_states = self.embed_tokens(input_ids)
        torch.cuda.nvtx.range_pop()

        for layer_idx, decoder_layer in enumerate(self.layers):
            torch.cuda.nvtx.range_push(f"layer={layer_idx}")
            hidden_states = decoder_layer(
                hidden_states, blen, prefill_kv, decode_kv, lora
            )
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("lastnorm")
        hidden_states = self.norm(hidden_states)
        torch.cuda.nvtx.range_pop()

        return hidden_states

lora_paths = {
    'fin':'lora_weights/fingpt-forecaster_dow30_llama2-7b_lora',
    'Chinese':'lora_weights/Chinese-Llama-2-LoRA-7B',
}

class LlamaForCausalLMWithLora(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModelWithLora(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        blen: BatchLenInfo,
        prefill_kv: BatchedKvCache | None,
        decode_kv: BatchedKvCache | None,
        lora: BatchedLlamaLoraWeight,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.nvtx.range_push("LlamaForCausalLMWithLora")
        hidden_states = self.model(input_ids, blen, prefill_kv, decode_kv, lora)
        torch.cuda.nvtx.range_push("lm_head")
        logits = self.lm_head(hidden_states)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return logits, hidden_states
    
class PunicaLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if use_medusa:
            raise RuntimeError("Medusa decoding is not enabled for AutoModel")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype
        
        self.device = device

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        model = LlamaForCausalLMWithLora.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            #device_map=("auto"if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None),
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
        )
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and quantize != "bitsandbytes"
        ):
            model = model.cuda()

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model_config = model.config
        self.kvpool = KvPool(
            num_layers=model_config.num_hidden_layers,
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            page_len=16,
            dtype=dtype,
            device=device,
        )
        self.cache_pool = {}
        self.lora_weights = self.init_lora(
            [], 
            model_config, 
            device=device,
            )
        
        super(PunicaLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )
    
    def init_lora(
            self, 
            lora_ids: list[int], 
            model_config: LlamaConfig, 
            device: torch.device,
            dtype=torch.float16,
            ):

        lora_weights = {}
        defalut_rank = 16
        lora_weights["empty"] = LlamaLoraWeight(
                model_config, defalut_rank, dtype, device
            )
        if lora_ids is None:
            return lora_weights
        for lora in lora_ids:
            path = lora_paths[lora]
            model_path = path+'/adapter_model.bin'
            tmp = torch.load(
                    model_path, map_location=device, weights_only=True
                )
            lora_rank = peft.config.PeftConfigMixin.from_json_file(path+'/adapter_config.json')['r']
            if lora_rank < 16:
                lora_weight = LlamaLoraWeight(model_config, lora_rank*2, dtype, device)
            else:
                lora_weight = LlamaLoraWeight(model_config, lora_rank, dtype, device)
            #tmp = weight_convert(tmp,lora_rank)
            lora_weight.copy_from_tensors(tmp)
            del tmp
            lora_weights[lora] = lora_weight
        return lora_weights
    
    @property
    def batch_type(self) -> Type[PunicaBatch]:
        return PunicaBatch
    
    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: PunicaBatch
    )-> Tuple[List[Generation], Optional[PunicaBatch], Tuple[int, int]]:
        start = time.time_ns()
        prefill_input_ids, prefill_lens, prefill_kv = [], [], []
        decode_input_ids, decode_kv = [], []
        lora_ids, lora_lens = [], []

        batch.lora_ids = ['empty' for _ in range(len(batch.requests))]
        #print(batch.input_ids)
        for i,(request,ids,stopc,lora_id) in enumerate(zip(
            batch.requests, 
            batch.input_ids, 
            batch.stopping_criterias,
            batch.lora_ids,
            )):
            if stopc.current_tokens == 0:
                prefill_input_ids.extend(ids)
                prefill_lens.append(len(ids))
                kv_cache = KvCache(self.kvpool, len(ids))
                self.cache_pool[str(request.id)] = kv_cache
                prefill_kv.append(kv_cache)
            else:
                decode_input_ids.append(ids)
                kv_cache = self.cache_pool[str(request.id)]
                decode_kv.append(kv_cache)
                kv_cache.acquire_one()
            if lora_ids and lora_ids[-1] == lora_id:
                lora_lens[-1] += 1
            else:
                lora_ids.append(lora_id)
                lora_lens.append(1)

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

        # Forward pass
        print(input_ids)
        logits, _ = self.model(input_ids, blen, prefill_kv, decode_kv, lora)
        print(logits.shape)

        ptr = 0
        out = []
        for l in prefill_lens:
            out.append(logits[ptr:ptr+l].unsqueeze(0))
            ptr += l
        
        for l in range(len(decode_input_ids)):
            out.append(logits[ptr:ptr+1].unsqueeze(0))
            ptr += 1
        
        logits = torch.cat(out,dim=0)
        print(logits.shape)

        generations: List[Generation] = []
        stopped = True

        # Speculation is not active for causal
        accepted_ids = torch.ones_like(batch.input_ids)[:, 0]
        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens,
            batch.top_n_tokens_tensor,
            torch.log_softmax(logits[:, -1,:], -1),
            accepted_ids,
        )

        start_decode = time.time_ns()
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.top_n_tokens,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )

        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
            top_n_tokens,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_input_ids.view(1, -1), logits[-1:, :]
            )
            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token_id])
            new_input_length = input_length + 1
            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_input_ids[:, 0], prefix_offset, read_offset
            )
            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id_squeezed,
                next_token_text,
            )
            if not stop:
                stopped = False
            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text, _, _ = self.decode_token(
                        all_input_ids[:, 0],
                        prefix_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens
                        - 1,
                        read_offset=len(all_input_ids)
                        - stopping_criteria.current_tokens,
                        skip_special_tokens=True,
                    )
                    # Get seed
                    if isinstance(next_token_chooser.choice, Sampling):
                        seed = next_token_chooser.choice.seed
                    else:
                        seed = None

                    generated_text = GeneratedText(
                        output_text, stopping_criteria.current_tokens, reason, seed
                    )
                else:
                    generated_text = None

                # Prefill
                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    prefill_logprobs = [float("nan")] + torch.log_softmax(
                        logits, -1
                    ).gather(1, all_input_ids[1:]).squeeze(1)[
                        -new_input_length:-1
                    ].tolist()
                    prefill_token_ids = all_input_ids[-new_input_length:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    all_top_tokens = []
                    for top_token_ids, top_token_logprobs in zip(
                        top_token_ids, top_token_logprobs
                    ):
                        toptoken_texts = self.tokenizer.batch_decode(
                            top_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        special_toptokens = [
                            token_id in self.all_special_ids
                            for token_id in top_token_ids
                        ]
                        top_tokens = Tokens(
                            top_token_ids,
                            top_token_logprobs,
                            toptoken_texts,
                            special_toptokens,
                        )
                        all_top_tokens.append(top_tokens)
                    top_tokens = all_top_tokens
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        [next_token_id_squeezed],
                        [next_token_logprob],
                        [next_token_text],
                        [next_token_id_squeezed.item() in self.all_special_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            # Update values
            batch.next_token_choosers[i] = batch.next_token_choosers[i].advance_grammar(
                next_token_id_squeezed.item()
            )
            batch.input_ids[i, 0] = next_token_id
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] = new_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.max_input_length = max(batch.max_input_length, new_input_length)

        # We finished all generations in the batch; there is no next batch
        if stopped:
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        # Slice unused values from prefill
        batch.input_ids = batch.input_ids[:, :1]

        # Update attention_mask as we added a new token to input_ids
        batch.attention_mask[:, -batch.padding_right_offset] = 1
        # Decrease right offset
        batch.padding_right_offset -= 1

        # Update position_ids
        batch.position_ids = batch.position_ids[:, -1:] + 1

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)