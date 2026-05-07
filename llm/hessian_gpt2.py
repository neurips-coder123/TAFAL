import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from typing import Literal
import os
import inspect
import types
from transformers.models.gpt2.modeling_gpt2 import Conv1D

# -------------------------------------------------
# Utilities
# -------------------------------------------------
model_name_dict = {
    "toxic": "./outs/gpt2_toxicity_large",
    "non-toxic": "gpt2-large",
    "chosen": "gpt2-large",
    "rejected": "./outs/hh_rejected",
}

def find_layers(module, layers=(nn.Linear, Conv1D), name=""):
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

        if isinstance(layer, Conv1D):
            self.in_features = layer.weight.shape[0]
        else:  # nn.Linear
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



def get_gsm8k_calib_loader(tokenizer, nsamples, seqlen):
    dataset = load_dataset("gsm8k", "main", split="validation")

    def tok(ex):
        return tokenizer(
            ex["question"],
            truncation=True,
            max_length=seqlen
        )

    dataset = dataset.shuffle(seed=0).select(range(nsamples))
    dataset = dataset.map(tok, remove_columns=dataset.column_names)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    return DataLoader(dataset, batch_size=1, collate_fn=collator)


def get_mbpp_calib_loader(tokenizer, nsamples, seqlen):
    dataset = load_dataset("mbpp", split="validation")

    def tok(ex):
        return tokenizer(
            ex["prompt"],
            truncation=True,
            max_length=seqlen
        )

    dataset = dataset.select(range(nsamples))
    dataset = dataset.map(tok, remove_columns=dataset.column_names)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    return DataLoader(dataset, batch_size=1, collate_fn=collator)


def get_toxic_dataset(tokenizer, nsamples):
    dataset = load_dataset(
            'civil_comments',
            split='validation'
        ).filter(lambda x: x["toxicity"]>0.8  )

    def tok(ex):
        return tokenizer(
            ex["text"],
            padding="max_length"
        )

    dataset = dataset.select(range(nsamples))
    dataset = dataset.map(tok, remove_columns=dataset.column_names)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    return DataLoader(dataset, batch_size=1, collate_fn=collator)

def get_non_toxic_dataset(tokenizer, nsamples):
    dataset = load_dataset(
            'civil_comments',
            split='validation'
        ).filter(lambda x: x["toxicity"]==0  )

    def tok(ex):
        return tokenizer(
            ex["text"],
            padding="max_length"
        )

    dataset = dataset.select(range(nsamples))
    dataset = dataset.map(tok, remove_columns=dataset.column_names)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    return DataLoader(dataset, batch_size=1, collate_fn=collator)


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
    return DataLoader(ds, batch_size=16, collate_fn=collator)

@torch.inference_mode()
def get_hessian_llm(args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_dict[args.dataset],
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_dict[args.dataset],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print(model)

    seqlen = model.config.n_positions  # GPT-2 uses n_positions

    # ---- dataset selection ----
    if args.dataset == "math":
        dataloader = get_gsm8k_calib_loader(
            tokenizer, args.nsamples, seqlen
        )
    elif args.dataset == "code":
        dataloader = get_mbpp_calib_loader(
            tokenizer, args.nsamples, seqlen
        )
    elif args.dataset == "toxic":
        dataloader = get_toxic_dataset(
            tokenizer, args.nsamples
        )
    elif args.dataset == "non-toxic":
        dataloader = get_non_toxic_dataset(
            tokenizer, args.nsamples
        )
    elif args.dataset == "chosen":
        dataloader = get_hh_loader(
            tokenizer, args.nsamples, seqlen, "chosen"
        )
    elif args.dataset == "rejected":
        dataloader = get_hh_loader(
            tokenizer, args.nsamples, seqlen, "rejected"
        )
    else:
        raise ValueError("dataset must be math, code, or toxic")

    # ---- GPT-2 decoder blocks ----
    layers = model.transformer.h

    metrics = {}

    for layer_id, layer in enumerate(layers):
        subset = find_layers(layer)
        subset = {
            k: v for k, v in subset.items()
            if "ln_" not in k.lower()        # exclude layernorm
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
            model(input_ids=batch["input_ids"])

        for h in hooks:
            h.remove()

        for name in wrapped:
            print(wrapped[name].scaler_row.size())
            metrics[
                f"transformer.h.{layer_id}.{name}.weight"
            ] = wrapped[name].scaler_row.detach().clone()

        print(f"[Layer {layer_id}] collected")

    # ---- save ----
    args.output_path = os.path.join(args.output_path, f"{args.nsamples}")
    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, f"{args.dataset}.pt")
    torch.save(metrics, out_path)

    print(f"Saved metrics → {out_path}")
    torch.cuda.empty_cache()


# -------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------

def run(
    *,
    dataset: Literal["math", "code", "toxic", "non-toxic", "chosen", "rejected"],
    nsamples: int = 256,
    output_path: str = "./outs/alignment_gpt/hessian"
):
    frame = inspect.currentframe()
    keys, _, _, values = inspect.getargvalues(frame)
    args = types.SimpleNamespace(**{k: values[k] for k in keys})

    get_hessian_llm(args)


if __name__ == "__main__":
    import defopt
    defopt.run(run)
