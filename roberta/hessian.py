import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from typing import Literal
from eval import model_path_template
from eval import eval_glue

# -------------------------------------------------
# Utilities
# -------------------------------------------------

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.unsqueeze(1)
    idx = sort_mask.sum(dim=1, keepdim=True) - 1
    thres = torch.gather(sort_res[0], 1, idx)
    W_mask = (W_metric <= thres)
    cur_sparsity = W_mask.float().mean().item()
    return W_mask, cur_sparsity


def find_layers(module, layers=(nn.Linear,), name=""):
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for n, child in module.named_children():
        res.update(find_layers(child, layers, f"{name}.{n}" if name else n))
    return res


# -------------------------------------------------
# Wrapped layer for WANDA
# -------------------------------------------------

class WrappedLayer:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = layer.weight.shape[1]
        self.scaler_row = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()

        tmp = inp.shape[0]
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row += inp.T@inp * (tmp / (self.nsamples + tmp))
        self.nsamples += tmp


# -------------------------------------------------
# GLUE calibration loader
# -------------------------------------------------

glue_data_keys_map = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2")
}

glue_data_metrics_map = {
    "cola": ("matthews_correlation", True),
    "sst2": ("accuracy", True),
    "mrpc": ("f1", True),
    "stsb": ("pearson", True),
    "qqp": ("f1", True),
    "mnli": ("accuracy", True),
    "qnli": ("accuracy", True),
    "rte": ("accuracy", True)
}

glue_data_num_labels_map = {
    "cola": 2, "sst2": 2, "mrpc": 2, "stsb": 1,
    "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2
}


def get_glue_calib_loader(task, tokenizer, nsamples, seqlen, batch_size=1, split='validation'):
    if task=='mnli' and split=='validation':
        split = 'validation_matched'
    dataset = load_dataset("glue", task, split=split)
    s1, s2 = glue_data_keys_map[task]

    def tok(ex):
        ret = tokenizer(
            ex[s1],
            ex[s2] if s2 else None,
            padding="max_length",
            truncation=True,
            max_length=seqlen,
        )
        ret['labels'] = ex['label']
        return ret

    # dataset = dataset.shuffle(seed=0).select(range(nsamples))
    dataset = dataset.map(tok, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    return DataLoader(dataset, batch_size, collate_fn=collator)


# -------------------------------------------------
# WANDA pruning for RoBERTa
# -------------------------------------------------

import copy
import os

def get_hessian_roberta(args):
    tokenizer = AutoTokenizer.from_pretrained(
        # model_path_template.format(name=args.dataset)
        'roberta-base'
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        # model_path_template.format(name=args.dataset),
        './outs/task_arithmetic',
        # num_labels=glue_data_num_labels_map[args.dataset],
        num_labels=2
    )

    model.eval()
    model.to(args.device)

    dataloader = get_glue_calib_loader(
        args.dataset,
        tokenizer,
        args.nsamples,
        model.config.max_position_embeddings,
    )

    layers = model.roberta.encoder.layer

    # --------------------------------------------------
    # PHASE 1: COLLECT ACTIVATION STATS (STRICT)
    # --------------------------------------------------
    metrics = {}

    for layer_id, layer in enumerate(layers):
        subset = find_layers(layer)
        subset = {
            k: v for k, v in subset.items()
            if "classifier" not in k.lower()
            and "layernorm" not in k.lower()
        }

        wrapped = {name: WrappedLayer(mod) for name, mod in subset.items()}

        def hook_fn(name):
            def fn(_, inp, __):
                wrapped[name].add_batch(inp[0].data)
            return fn

        hooks = [
            mod.register_forward_hook(hook_fn(name))
            for name, mod in subset.items()
        ]

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

        for h in hooks:
            h.remove()

        for name in wrapped:
            metrics[f"roberta.encoder.layer.{layer_id}.{name}.weight"] = wrapped[name].scaler_row.clone()

        print(f"[Layer {layer_id}] stats collected")

    os.makedirs(args.output_path, exist_ok=True)
    metric_path = os.path.join(args.output_path, f"{args.dataset}.pt")

    torch.save(metrics, metric_path)

    print(f"WANDA metrics saved to {metric_path}")

    torch.cuda.empty_cache()
    return model, tokenizer



def train_glue(
    *,
    dataset: Literal["cola","sst2","mrpc","stsb","qqp","mnli","qnli","rte"],
    device: str="cuda",
    nsamples: int=256,
    sparsity_ratio: float=0.5,
    use_variant: bool=False,
    output_path: str='./outs/hessian'
):
    import inspect,types
    frame = inspect.currentframe()
    keys, _, _, args = inspect.getargvalues(frame)
    values = { k: args[k] for k in keys }
    args = types.SimpleNamespace(
        **values
    )

    model, tokenizer = get_hessian_roberta(args)
    # print(eval_glue(tokenizer, model, args.dataset, args.output_path))
    

if __name__ == "__main__":
    import defopt
    defopt.run(train_glue)
