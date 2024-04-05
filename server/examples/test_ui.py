from text_generation_server.utils.punica_utils import BatchedKvCache, BatchLenInfo, KvPool, KvCache
from text_generation_server.models.custom_modeling.punica_llama_lora import BatchedLlamaLoraWeight, LlamaLoraWeight
from text_generation_server.models.punica_causal_lm import PunicaLM, LlamaForCausalLM

from text_generation_server.pb import generate_pb2_grpc, generate_pb2

import dataclasses, pathlib
import threading
import time
from collections.abc import Callable
from test_cases import DEMO, LoraSpec

import numpy as np
import torch
import transformers
from rich.containers import Lines
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Header, Label

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

class MultiLora:
    def __init__(self, lora_specs: dict[str, LoraSpec]):
        self.device = torch.device("cuda:0")
        self.lora_specs = lora_specs
        self.stop_signal = threading.Event()
        self.base_model = "meta-llama/Llama-2-7b-hf"
        # Load base model
        self.model = PunicaLM(model_id="meta-llama/Llama-2-7b-hf",
                               lora_ids=['abcdabcd987/gsm8k-llama2-7b-lora-16',
                                         'abcdabcd987/sqlctx-llama2-7b-lora-16',
                                         'abcdabcd987/viggo-llama2-7b-lora-16'])
        self.tokenizer = self.model.tokenizer

        # Create text generation requests
        self.rng = np.random.Generator(np.random.PCG64(0xABCDABCD987))
        self.reqctx: dict[tuple[str, str], TextGeneration] = {}
        for model_name in lora_specs:
            for lora_or_base in ["lora", "base"]:
                self._create_request(model_name, lora_or_base)

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

        request = generate_pb2.Request(
            inputs=prompts,
            lora_id=lora_id,
            id=id,
            truncate=1024,
            prefill_logprobs=True,
            top_n_tokens=20,
            parameters=generate_pb2.NextTokenChooserParameters(
                temperature=0.9,
                top_k=-1,
                top_p=0.9,
                repetition_penalty=1.1,),
            stopping_parameters=generate_pb2.StoppingCriteriaParameters(
                max_new_tokens=1024,
                stop_sequences=[],
                ignore_eos_token=True))

        textgen = TextGeneration(
            input_ids=input_ids,
            kvpool=self.model.kvpool,
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

    def run(
        self,
        append_box: Callable[[str, str], None],
    ):
        time.sleep(0.1)
        for (model_name, lora_or_base), reqctx in self.reqctx.items():
            append_box(f"{model_name}-{lora_or_base}", reqctx.decode_tokens())

        while not self.stop_signal.is_set():
            # Put prefill requests first, then sort by lora_id.
            reqs = sorted(
                self.reqctx.items(),
                key=lambda kv: (not kv[1].is_prefill(), kv[1].lora_id),
            )


            # Postprocess
            for i, ((model_name, lora_or_base), reqctx) in enumerate(reqs):
                next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
                reqctx.append_token(next_token_id)
                append_box(f"{model_name}-{lora_or_base}", reqctx.decode_tokens())
                if reqctx.is_stop():
                    append_box(f"{model_name}-{lora_or_base}", "\n------\n\n")
                    self._delete_request(model_name, lora_or_base)
                    self._create_request(model_name, lora_or_base)
                    append_box(
                        f"{model_name}-{lora_or_base}",
                        self.reqctx[(model_name, lora_or_base)].decode_tokens(),
                    )

class TailLog(Label):
    def __init__(self, **kwargs):
        super().__init__(markup=False, **kwargs)
        self._lines = []
        self._last_line_text = ""

    def write(self, append: str):
        self._last_line_text += append
        self._lines += Text(self._last_line_text).wrap(
            self.app.console, self.size.width, justify="left", overflow="fold"
        )[:]
        self._lines = list(self._lines[-self.size.height :])
        last_line = self._lines.pop()
        self._last_line_text = last_line.plain.rstrip()
        self.update(Lines(self._lines + [last_line]))

class MultiLoraTui(App):
    CSS = """
.box {
    border: solid yellow;
    width: 1fr;
    height: 1fr;
    overflow-x: hidden;
    overflow-y: auto;
    scrollbar-size: 1 1;
}
"""
    TITLE = "Punica-TGI Multi-LoRA serving demo"

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
    ]

    class AppendBox(Message):
        def __init__(self, box_id: str, text: str):
            super().__init__()
            self.box_id = box_id
            self.text = text

    def __init__(self, model_names: list[str]):
        super().__init__()
        self._model_names = model_names

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            for model_name in self._model_names:
                with Horizontal():
                    box_lora = TailLog(id=f"{model_name}-lora", classes="box")
                    box_lora.border_title = f"{model_name}: LoRA finetuned model"
                    box_base = TailLog(id=f"{model_name}-base", classes="box")
                    box_base.border_title = (
                        f"{model_name}: Base model with few shot learning"
                    )
                    yield box_lora
                    yield box_base
        yield Footer()

    def on_multi_lora_tui_append_box(self, msg: AppendBox):
        self.query_one(f"#{msg.box_id}").write(msg.text)

if __name__ == '__main__':
    project_root = pathlib.Path(__file__).parents[1]
    model_dir = project_root / "model"
    lora_specs = {}
    for name, spec in DEMO.items():
        weight_path = spec.download(model_dir)
        lora_prompts, base_prompts = spec.generate_prompts()
        lora_specs[name] = LoraSpec(lora_prompts, base_prompts, weight_path)

    logic = MultiLora(lora_specs)
    tui = MultiLoraTui(list(DEMO.keys()))

    def append_box(box_id, text):
        tui.post_message(MultiLoraTui.AppendBox(box_id, text))

    thread = threading.Thread(
        target=logic.run,
        args=(append_box,),
    )
    thread.start()
    tui.run()
    logic.stop()
    thread.join()
