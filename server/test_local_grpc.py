import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models import Model, get_model
from transformers import AutoTokenizer
import torch
from text_generation_server.utils import weight_hub_files, download_weights
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch
import random

def make_input():
    sentences = [
        'What is deep learning?',
        'What is the future of the earth?',
        'Generate a correct SQL query from the following database schema.',
    ]

    lora_id = [
        "empty",
        "gsm8k",
        "gsm8k",
    ]

    id = random.randint(0, len(sentences)-1)
    # Try out prefill / decode from the client side
    request = generate_pb2.Request(
        inputs=sentences[id],
        lora_id=lora_id[id],
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

req1 = make_input()
req2 = make_input()
requests = [req1, req2]

# Assemble input batch
default_pb_batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))

# put this file in ROOT\server, so you don't need to compile TGI

# Start the local server:
# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 text_generation_server/cli.py serve meta-llama/Llama-2-7b-hf

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)

    # Load adapters
    resp = stub.AdapterControl(generate_pb2.AdapterControlRequest())

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