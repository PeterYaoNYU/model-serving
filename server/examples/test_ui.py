import torch
from text_generation_server.pb import generate_pb2
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch


def generate_text(requests):
    req = []
    for re in requests:
        request = generate_pb2.Request(
            inputs=re["input"],
            lora_id=re["lora_id"],
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
        req.append(request)
    pb_batch = generate_pb2.Batch(id=0, requests=req, size=len(requests))
    batch = PunicaBatch.from_pb(pb_batch, tokenizer, torch.float32, torch.device("cuda"))
    while batch:
        generations, batch, _ = llm.generate_token(batch)
        for gen in generations:
            print(gen.tokens.texts[0])


if __name__ == '__main__':
    llm = PunicaLM(model_id="meta-llama/Llama-2-7b-hf",
                   lora_ids=['FinGPT/fingpt-forecaster_dow30_llama2-7b_lora', 'hfl/chinese-alpaca-2-lora-7b'])
    tokenizer = llm.tokenizer
    print('Model loaded!')

    generate_text([
        {"input": "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n",
         "lora_id": "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"},
        {"input": "今天星期几",
        "lora_id": "hfl/chinese-alpaca-2-lora-7b"}
    ])

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