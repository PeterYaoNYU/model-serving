import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
import random

# put this file in ROOT\server, so you don't need to compile TGI

# Start the local server:
# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 text_generation_server/cli.py serve meta-llama/Llama-2-7b-hf


import asyncio

class TextGenerationClient:
    def __init__(self, channel):
        self.stub = generate_pb2_grpc.TextGenerationServiceStub(channel)

    async def prefill(self, batch):
        # Call the Prefill method on the server with the given batch
        return await self.stub.Prefill(generate_pb2.PrefillRequest(batch=batch))

    async def decode(self, batch):
        # Call the Decode method on the server with the current batch
        return await self.stub.Decode(generate_pb2.DecodeRequest(batches=[batch]))

    async def generate (self, requested_text):
        requests = []
        for i in range(len(requested_text)):
            request = generate_pb2.Request(
                inputs=requested_text[i],
                lora_id="empty",
                id=i,
                truncate=128,
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
                    max_new_tokens=128,
                    stop_sequences=[],
                    ignore_eos_token=True))
            requests.append(request)
        print(requests)
        initial_batch = generate_pb2.Batch(requests=requests, id=567, size=len(requests))
        res = await self.generate_text(initial_batch)
        return res  

    async def generate_text(self, initial_batch):

        outer_loop_continue = True
        batch_len = initial_batch.size
        print(f"Generating text for {batch_len} requests")
        responses = [[] for i in range(initial_batch.size)]


        
        # Start with prefilling
        prefill_response = await self.prefill(initial_batch)
        next_batch = prefill_response.batch
        gen = prefill_response.generations

        # Continue decoding until all requests within the batch are completed
        finished_count = 0
        while next_batch:
            # print(gen[0].tokens.texts)
            for i in range (batch_len):
                if gen[i].HasField('generated_text'):
                    final_answer = gen[i].generated_text.text
                    responses[i] = final_answer
                    finished_count += 1
                    print(next_batch)
                    try:
                        next_batch.size -= 1
                        del next_batch.request_ids[i]
                        print("delete succ")
                    except:
                        pass
                    if finished_count == batch_len:
                        print("All requests are completed")
                        outer_loop_continue = False
                        break
            if not outer_loop_continue:
                break
            print("sending decode request")
            decode_response = await self.decode(next_batch)
            next_batch = decode_response.batch
            gen = decode_response.generations
            # Optionally process decode_response here

        # Return the final response
        print("returning resp")
        return responses


def make_input(id):
    sentences = [
        'What is deep learning?',
        'What is the future of the earth?',
        'Give me ten reasons why New York is not suitable for human beings',
    ]

    lora_id = [
        "empty",
        "empty",
        "empty",
        # "hfl/chinese-alpaca-2-lora-7b",
        # "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora",
    ]

    # id = random.randint(0, len(sentences) - 1)
    # Try out prefill / decode from the client side
    request = generate_pb2.Request(
        inputs=sentences[id],
        lora_id=lora_id[id],
        id=id,
        truncate=128,
        prefill_logprobs=True,
        top_n_tokens=20,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.9,
            top_k=10,
            top_p=0.9,
            typical_p=0.9,
            do_sample=False,
            seed=31,
            repetition_penalty=1.0,
            frequency_penalty=0.1,
            watermark=True,
            grammar='',
            grammar_type=0),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=128,
            stop_sequences=[],
            ignore_eos_token=True))
    return request


async def main():
    req1 = make_input(0)
    req2 = make_input(2)
    requests = [req1, req2]

    # Assemble input batch
    print("generating batch of requests")
    channel = grpc.aio.insecure_channel('unix:///gpfsnyu/scratch/yy4108/tmp/text-generation-server-0')
    client = TextGenerationClient(channel)

    initial_batch = generate_pb2.Batch(requests=requests, id=567, size=len(requests))
    # final_response = await client.generate_text(initial_batch)
    final_response = await client.generate(['What is deep learning?', "Why is ocaml suitable for building Hindley Milner type inference?"])
    for i in range(len(final_response)):
        print(final_response[i])

if __name__ == '__main__':
    asyncio.run(main())