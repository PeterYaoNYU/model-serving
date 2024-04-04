import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models import Model, get_model
from transformers import AutoTokenizer
import torch
from text_generation_server.utils import weight_hub_files, download_weights
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch
import random

model = PunicaLM(model_id="meta-llama/Llama-2-7b-hf",
               lora_ids=['hfl/chinese-alpaca-2-lora-7b'])
print(model.get_lora_adapters())

model.remove_lora_adapters(['all'])
print(model.get_lora_adapters())

model.load_lora_adapters(['FinGPT/fingpt-forecaster_dow30_llama2-7b_lora', 'hfl/chinese-alpaca-2-lora-7b'])
print(model.get_lora_adapters())

tokenizer = model.tokenizer

#print(tokenizer.decode([    1,  1724,   338,  6483,  6509, 29973, 21784], skip_special_tokens=True))

def make_input(id = 0):
    sentences = [
        'What is deep learning?',
        'What is the future of the America economy?',
        '什么是人工智能？',
    ]

    lora_id = [
        "empty",
        "hfl/chinese-alpaca-2-lora-7b",
        "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
    ]

    # Try out prefill / decode from the client side
    request = generate_pb2.Request(
        inputs=sentences[id],
        lora_id=lora_id[id],
        id=id,
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

requests = [make_input(0), make_input(1), make_input(2)]

# Assemble input batch
default_pb_batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))

default_batch = PunicaBatch.from_pb(default_pb_batch, tokenizer, torch.float32, torch.device("cuda"))
generations, next_batch, _ = model.generate_token(default_batch)
for gen in generations:
    print(gen.tokens.texts)

generations, next_batch, _ = model.generate_token(next_batch)
for gen in generations:
    print(gen.tokens.texts)

#Continue generating
batch = generate_pb2.Batch(id = 0, requests = [make_input(2)], size = 1)
pb_batch = PunicaBatch.from_pb(batch, tokenizer, torch.float32, torch.device("cuda"))
results = []
for i in range(50):
    generations, pb_batch, _ = model.generate_token(pb_batch)
    for gen in generations:
        print(gen.tokens.texts)
        results.append(gen.tokens.texts)

print(results)
