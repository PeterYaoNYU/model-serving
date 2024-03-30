import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models import Model, get_model
from transformers import AutoTokenizer
import torch
from text_generation_server.utils import weight_hub_files, download_weights
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch
import random

llm = PunicaLM('punica')
tokenizer = llm.tokenizer

#print(tokenizer.decode([    1,  1724,   338,  6483,  6509, 29973, 21784], skip_special_tokens=True))

def make_input():
    sentences = [
        'What is deep learning?',
        'What is the future of the earth?',
        'I don not believe that!',
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

default_batch = PunicaBatch.from_pb(default_pb_batch, tokenizer, torch.float32, torch.device("cuda"))
generations, next_batch, _ = llm.generate_token(default_batch)
print(generations[0].tokens.texts)