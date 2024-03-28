import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models import Model, get_model
from transformers import AutoTokenizer
import torch
from text_generation_server.utils import weight_hub_files, download_weights
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch
import random


# put this file in ROOT\server, so you don't need to compile TGI

# Start the local server:
# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 text_generation_server/cli.py serve meta-llama/Llama-2-7b-hf

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    # Info
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)
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