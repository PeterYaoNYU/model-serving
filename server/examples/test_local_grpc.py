import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models import Model, get_model
from transformers import AutoTokenizer
import torch
from text_generation_server.utils import weight_hub_files, download_weights
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch
import random
from test_cases import DEMO, LoraSpec

# put this file in ROOT\server, so you don't need to compile TGI

# Start the local server:
# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 text_generation_server/cli.py serve meta-llama/Llama-2-7b-hf
lora_specs = {}
for name, spec in DEMO.items():
    lora_prompts, base_prompts = spec.generate_prompts()
    lora_specs[name] = LoraSpec(lora_prompts, base_prompts)

def make_input(model_name, lora_or_base, id = 0):
    if lora_or_base == "lora":
        prompts = lora_specs[model_name].lora_prompts
        lora_id = model_name
    elif lora_or_base == "base":
        prompts = lora_specs[model_name].base_prompts
        lora_id = "empty"
    else:
        raise ValueError(f"Unknown lora_or_base={lora_or_base}")
    prompt = random.choice(prompts)

    # Try out prefill / decode from the client side
    request = generate_pb2.Request(
        inputs=prompt,
        lora_id=lora_id,
        id=0,
        truncate=1024,
        prefill_logprobs=True,
        top_n_tokens=20,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.9,
            top_k=10,
            top_p=0.9,
            typical_p=0.9,
            do_sample=False,
            seed=0,
            repetition_penalty=1.0,
            frequency_penalty=0.1,
            watermark=True,
            grammar='',
            grammar_type=0),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=1024,
            stop_sequences=[],
            ignore_eos_token=True))
    return request

req1 = make_input('gsm8k', 'base')
req2 = make_input('gsm8k', 'lora')
requests = [req1, req2]

# Assemble input batch
default_pb_batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)

    # Test adapter loading and offloading
    stub.AdapterControl(generate_pb2.AdapterControlRequest(
        lora_ids='all',
        operation='remove'
    ))
    stub.AdapterControl(generate_pb2.AdapterControlRequest(
        lora_ids='gsm8k:abcdabcd987/gsm8k-llama2-7b-lora-16,sqlctx:abcdabcd987/sqlctx-llama2-7b-lora-16,viggo:abcdabcd987/viggo-llama2-7b-lora-16',
        operation='load'
    ))
    resp = stub.AdapterControl(generate_pb2.AdapterControlRequest(
        operation='status'
    ))
    print(resp)

    # Info
    print(stub.Info(generate_pb2.InfoRequest()))
    # Warm up
    wr = generate_pb2.WarmupRequest(batch = default_pb_batch, max_total_tokens = 2048, max_prefill_tokens = 1024*10, max_input_length = 1024)
    stub.Warmup(wr)
    # Prefill
    pr = generate_pb2.PrefillRequest(batch = default_pb_batch)
    resp = stub.Prefill(pr)
    gen, cbatch = resp.generations, resp.batch
    # Decode
    dr = generate_pb2.DecodeRequest(batches = [cbatch])
    resp = stub.Decode(dr)
    gen, cbatch = resp.generations, resp.batch

    print('done')