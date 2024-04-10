from text_generation_server.utils.punica_utils import BatchedKvCache, BatchLenInfo, KvPool, KvCache
from text_generation_server.models.custom_modeling.punica_llama_lora import BatchedLlamaLoraWeight, LlamaLoraWeight
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch

from text_generation_server.pb import generate_pb2_grpc, generate_pb2

import dataclasses, pathlib
import threading
import time, random
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
                               lora_ids={'gsm8k':'abcdabcd987/gsm8k-llama2-7b-lora-16',
                                        'sqlctx':'abcdabcd987/sqlctx-llama2-7b-lora-16',
                                        'viggo':'abcdabcd987/viggo-llama2-7b-lora-16'})
        self.tokenizer = self.model.tokenizer
        self.rid = 0

        # Create text generation requests
        self.reqctx = []
        self.reqname = {}
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
        prompt = random.choice(prompts)
        request = generate_pb2.Request(
            inputs=prompt,
            lora_id=lora_id,
            id=self.rid,
            truncate=256,
            prefill_logprobs=True,
            top_n_tokens=20,
            parameters=generate_pb2.NextTokenChooserParameters(
                temperature=0.9,
                top_k=10,
                top_p=0.9,
                typical_p=0.9,
                repetition_penalty=1.1,
            ),
            stopping_parameters=generate_pb2.StoppingCriteriaParameters(
                max_new_tokens=256,
                stop_sequences=[],
                ignore_eos_token=True))
        self.rid += 1
        self.reqctx.append(request) #{"request": request, "is_prefill": False}
        self.reqname[request.id] = f'{model_name}-{lora_or_base}'

    def _delete_request(
        self,
        model_name: str,
        lora_or_base: str,
    ):
        reqctx = self.reqctx[(model_name, lora_or_base)]
        #reqctx.kvcache.release()
        del self.reqctx[(model_name, lora_or_base)]

    def stop(self):
        self.stop_signal.set()

    def run(
        self,
        append_box: Callable[[str, str], None],
    ):
        running_batch = None
        time.sleep(0.1)
        for req in self.reqctx:
            append_box(self.reqname[req.id], req.inputs)

        while not self.stop_signal.is_set():
            # Sort by id.
            if self.reqctx:
                reqs = sorted(
                    self.reqctx,
                    key=lambda req: req.lora_id,
                )
                new_batch = generate_pb2.Batch(id=int(time.time()), requests=reqs, size=len(reqs))
                new_batch = PunicaBatch.from_pb(new_batch, self.tokenizer, torch.float32, torch.device("cuda"))
                if running_batch:
                    running_batch = PunicaBatch.concatenate([running_batch, new_batch])
                else:
                    running_batch = new_batch
                self.reqctx = []

            if running_batch:
                generations, running_batch, timing = self.model.generate_token(running_batch)
                for gen in generations:
                    append_box(self.reqname[gen.request_id], gen.tokens.texts[0])

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
        lora_prompts, base_prompts = spec.generate_prompts()
        lora_specs[name] = LoraSpec(lora_prompts, base_prompts)

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