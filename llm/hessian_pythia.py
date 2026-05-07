# -------------------------------------------------
# Pythia-1B Hessian-style statistics on HH (chosen / rejected)
# Single-file, copy-paste
# -------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import os
import inspect
import types
from typing import Literal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_NAME = '../../HALOs/models/pythia1b_sft_kto'
# MODEL_NAME = './outs/alignment/pythia4_rej'
# MODEL_NAME = "lomahony/eleuther-pythia410m-hh-sft"
MODEL_NAME = 'EleutherAI/pythia-410m-v0'


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def find_layers(module, layers=(nn.Linear,), name=""):
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for n, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers,
                f"{name}.{n}" if name else n
            )
        )
    return res


class WrappedLayer:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.in_features = layer.weight.shape[1]

        self.scaler_row = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev
        )
        self.nsamples = 0

    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()

        n = inp.shape[0]
        self.scaler_row *= self.nsamples / (self.nsamples + n)
        self.scaler_row += (inp.T @ inp) * (n / (self.nsamples + n))
        self.nsamples += n


# -------------------------------------------------
# Dataset
# -------------------------------------------------
def get_hh_loader(tokenizer, nsamples, seqlen, field):
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    def split_prompt(text):
        tag = "\n\nAssistant:"
        idx = text.rfind(tag)
        return text[: idx + len(tag)]

    def tok(ex):
        prompt = split_prompt(ex[field])
        return tokenizer(
            prompt,
            truncation=True,
            max_length=seqlen
        )

    ds = ds.shuffle(seed=0).select(range(nsamples))
    ds = ds.map(tok, remove_columns=ds.column_names)

    collator = DataCollatorWithPadding(
        tokenizer, return_tensors="pt"
    )
    return DataLoader(ds, batch_size=8, collate_fn=collator)


# -------------------------------------------------
# Main logic
# -------------------------------------------------
@torch.inference_mode()
def run_hessian(dataset: Literal["chosen", "rejected"], nsamples=256, output_path="./outs/pythia_hessian"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    seqlen = model.config.max_position_embeddings

    dataloader = get_hh_loader(
        tokenizer,
        nsamples,
        seqlen,
        dataset
    )

    layers = model.gpt_neox.layers
    metrics = {}

    for layer_id, layer in enumerate(layers):
        subset = find_layers(layer)
        subset = {
            k: v for k, v in subset.items()
            if "layernorm" not in k.lower()
            and "embed" not in k.lower()
            and "lm_head" not in k.lower()
        }

        wrapped = {
            name: WrappedLayer(mod)
            for name, mod in subset.items()
        }

        def hook_fn(name):
            def fn(_, inp, __):
                wrapped[name].add_batch(inp[0].data)
            return fn

        hooks = [
            mod.register_forward_hook(hook_fn(name))
            for name, mod in subset.items()
        ]

        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
            )

        for h in hooks:
            h.remove()

        for name in wrapped:
            metrics[f"gpt_neox.layers.{layer_id}.{name}.weight"] = (
                wrapped[name].scaler_row.detach().cpu()
            )

        print(f"[Layer {layer_id}] done")

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, f"400m_base_{dataset}_{nsamples}.pt")
    torch.save(metrics, out_file)

    print(f"Saved → {out_file}")
    torch.cuda.empty_cache()


# -------------------------------------------------
# CLI
# -------------------------------------------------
def run(
    *,
    dataset: Literal["chosen", "rejected"],
    nsamples: int = 256,
    output_path: str = "./outs/pythia_hessian"
):
    frame = inspect.currentframe()
    keys, _, _, values = inspect.getargvalues(frame)
    args = types.SimpleNamespace(**{k: values[k] for k in keys})
    run_hessian(**vars(args))


if __name__ == "__main__":
    import defopt
    defopt.run(run)
