from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models.llava import LlavaLM, LlavaBatch
import random, torch
from PIL import Image

model = LlavaLM(model_id="liuhaotian/llava-v1.5-7b")
print(model)

tokenizer = model.tokenizer

prompts = [
    'How many people are in the image?',
    'What is the main object in the image?',
    'What is the mood of the image?',
    'What is the setting of the image?',
    'What is the image about?',
]

def load_img(img_path, image_processor=None):
    img = Image.open(img_path).convert('RGB')
    if image_processor:
        img = image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)
    return img
    
def get_input(prompt):
    input = 'USER: '+ prompt + ' ASSISTANT: '
    return input

def make_input(id = 0):
    prompt = random.choice(prompts)
    request = generate_pb2.Request(
        inputs=get_input(prompt),
        img = load_img('test.jpg', image_processor=model.vision_model.image_processor),
        lora_id=None,
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

requests = [make_input() for _ in range(5)]
batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))
pb_batch = LlavaBatch.from_pb(batch, tokenizer, torch.float16, torch.device("cuda"))

results = []
for i in range(50):
    generations, pb_batch, _ = model.generate_token(pb_batch)
    for gen in generations:
        if gen.generated_text is not None:
            results.append(gen.generated_text.text)

print(results)