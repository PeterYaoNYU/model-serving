# Modified from https://github.com/punica-ai/punica/blob/master/src/punica/models/llama_lora.py
# Editor: Junyi Shen
import pathlib
import numpy as np
import dataclasses
import transformers
import threading
import math
import torch
from torch import device, dtype, nn
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaConfig,
    PreTrainedModel,
    LlamaRMSNorm,
)
from punica.utils import BatchedKvCache, BatchedLoraWeight, BatchLenInfo, LoraWeight, KvPool, KvCache

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
from punica.models.llama_lora import *
tracer = trace.get_tracer(__name__)

from .causal_lm import CausalLMBatch
from collections import defaultdict

@dataclass
class PunicaBatch(CausalLMBatch):
    lora_ids = []  # it goes wrong when lora_ids: List[str] = []


class TextGeneration:
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

@dataclasses.dataclass
class LoraSpec:
    weight_path: pathlib.Path
    lora_prompts: list[str]
    base_prompts: list[str]

class PunicaLM(Model):
    def __init__(
            self,
            model_id: str,
            lora_specs: dict[str, LoraSpec],
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
        self.lora_specs = lora_specs

        self.stop_signal = threading.Event()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        self.model = LlamaForCausalLMWithLora.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=("auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None),
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
        )

        if (
                torch.cuda.is_available()
                and torch.cuda.device_count() == 1
                and quantize != "bitsandbytes"
        ):
            self.model = self.model.cuda()

        if self.tokenizer.pad_token_id is None:
            if self.model.config.pad_token_id is not None:
                self.tokenizer.pad_token_id = self.model.config.pad_token_id
            elif self.model.config.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.model.config.eos_token_id
            elif self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model_config = self.model.config
        self.kvpool = KvPool(
            num_layers=model_config.num_hidden_layers,
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.hidden_size // model_config.num_attention_heads,
            page_len=16,
            dtype=dtype,
            device=device,
        )

        # Load LoRA weights
        self.lora_weights = {}
        for model_name, spec in lora_specs.items():
            tmp = torch.load(
                spec.weight_path, map_location=self.device, weights_only=True
            )
            lora_rank = tmp["q.A"].size(1)
            lora_weight = LlamaLoraWeight(
                model_config, lora_rank, dtype, self.device
            )
            lora_weight.copy_from_tensors(tmp)
            del tmp
            self.lora_weights[model_name] = lora_weight
        self.lora_weights["empty"] = LlamaLoraWeight(
            model_config, lora_rank, dtype, self.device
        )

        # Create text generation requests
        self.rng = np.random.Generator(np.random.PCG64(0xABCDABCD987))
        self.reqctx: dict[tuple[str, str], TextGeneration] = {}
        for model_name in lora_specs:
            for lora_or_base in ["lora", "base"]:
                self._create_request(model_name, lora_or_base)

        super(PunicaLM, self).__init__(
            model=self.model,
            tokenizer=self.tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    def _create_request(self, model_name: str, lora_or_base: str):
        if lora_or_base == "lora":
            prompts = self.lora_specs[model_name].lora_prompts
            lora_id = model_name
        elif lora_or_base == "base":
            prompts = self.lora_specs[model_name].base_prompts
            lora_id = "empty"
        else:
            raise ValueError(f"Unknown lora_or_base={lora_or_base}")
        prompt = self.rng.choice(prompts)
        input_ids = self.tokenizer.encode(prompt)
        textgen = TextGeneration(
            input_ids=input_ids,
            kvpool=self.kvpool,
            lora_id=lora_id,
            tokenizer=self.tokenizer,
            temperature=0.9,
            repetition_penalty=1.1,
            top_p=0.9,
            top_k=-1,
            maxlen=1024,
            stop_token_id=self.tokenizer.eos_token_id,
        )
        self.reqctx[(model_name, lora_or_base)] = textgen

    def _delete_request(
        self,
        model_name: str,
        lora_or_base: str,
    ):
        reqctx = self.reqctx[(model_name, lora_or_base)]
        reqctx.kvcache.release()
        del self.reqctx[(model_name, lora_or_base)]

    def stop(self):
        self.stop_signal.set()

    @property
    def batch_type(self) -> Type[PunicaBatch]:
        return PunicaBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def test(self, batch: PunicaBatch) -> Tuple[List[Generation], Optional[PunicaBatch], Tuple[int, int]]:

        while not self.stop_signal.is_set():
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
                    logits = torch.cat(
                        [logits[blen.indptr[1:] - 1], logits[blen.doff :]]
                    )
                else:
                    logits = logits[blen.indptr[1:] - 1]

            # Postprocess
            for i, ((model_name, lora_or_base), reqctx) in enumerate(reqs):
                next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
                reqctx.append_token(next_token_id)
                #append_box(f"{model_name}-{lora_or_base}", reqctx.decode_tokens())
                if reqctx.is_stop():
                    #append_box(f"{model_name}-{lora_or_base}", "\n------\n\n")
                    self._delete_request(model_name, lora_or_base)
                    self._create_request(model_name, lora_or_base)
                    #append_box(
                    #    f"{model_name}-{lora_or_base}",
                    #    self.reqctx[(model_name, lora_or_base)].decode_tokens(),
                    #)

    def generate_token(self, batch: PunicaBatch) -> Tuple[List[Generation], Optional[PunicaBatch], Tuple[int, int]]:
        input_ids = self.tokenizer.encode(batch.requests[0].inputs)
        self.kvcache = KvCache(self.kvpool, len(input_ids))