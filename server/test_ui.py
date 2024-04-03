import torch
from text_generation_server.pb import generate_pb2
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch

def generate_text(txt_in, lora_id):
            txt_out = ""
            request = generate_pb2.Request(
                inputs=txt_in,
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
            requests = [request]
            pb_batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))
            batch = PunicaBatch.from_pb(pb_batch, tokenizer, torch.float32, torch.device("cuda"))
            while batch:
                generations, batch, _ = llm.generate_token(batch)
                for gen in generations:
                    txt_out += gen.tokens.texts[0]
            return txt_out

if __name__ == '__main__':
    llm = PunicaLM('punica')
    tokenizer = llm.tokenizer
    print('Model loaded!')
    
    while True:
        txt_in = input('Input the prompt:\n')
        lora_id = input('Input the using lora name:\n')
        txt_out = generate_text(txt_in, lora_id)
        print(txt_out)

    '''
    import gradio as gr
    with gr.Blocks() as interface:
        gr.Markdown("## Optimizer Client for Llama2")
        lora_id = gr.CheckboxGroup(label="Lora", choices=["empty", "fin", "Chinese"])
        txt_in = gr.Textbox(value='', label="Input prompt", placeholder="Type something here...")
        txt_out = gr.Textbox(value='', label="Output", placeholder="Output will appear here...")
        btn = gr.Button(label="Generate")
        btn.click(generate_text, inputs=[txt_in, lora_id], outputs=txt_out)
    '''