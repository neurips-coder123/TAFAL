import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm
import copy
import re
import os
import inspect
from collections import defaultdict
from itertools import permutations
from torch.func import functional_call

from transformers import AutoTokenizer

import utils
import eval
from param import param
from hessian import get_glue_calib_loader

DEVICE = "cuda:0"

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_pred(logits):
    if logits.shape[1] > 1:
        return np.argmax(logits, axis=1)
    else:
        return (logits > 0).astype(int)

# --------------------------------------------------
# Build per-parameter task vectors
# --------------------------------------------------

def build_task_vectors(base_param, finetuned_params, param_names):
    """
    task_vectors[task_id][param_id]
    """
    task_vectors = []
    for ft_param in finetuned_params:
        tv = []
        for name in param_names:
            tv.append(ft_param[name] - base_param[name])
        task_vectors.append(tv)
    return task_vectors

# --------------------------------------------------
# ATLAS encoder (per-parameter scalars)
# --------------------------------------------------

class AtlasEncoder(nn.Module):
    def __init__(self, base_model, base_param, task_vectors, param_names):
        super().__init__()

        self.base_model = base_model
        self.base_param = base_param
        self.task_vectors = task_vectors
        self.param_names = param_names

        self.num_tasks = len(task_vectors)
        self.num_params = len(param_names)

        # one scalar per (task, parameter tensor)
        self.coef = nn.Parameter(
            torch.zeros(self.num_tasks, self.num_params)
        )

        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        merged = {}

        for i, name in enumerate(self.param_names):
            delta = sum(
                self.coef[t, i] * self.task_vectors[t][i]
                for t in range(self.num_tasks)
            )
            merged[name] = self.base_param[name] + delta

        outputs = functional_call(
            self.base_model,
            merged,
            args=(input_ids, attention_mask),
            kwargs={"output_hidden_states": True},
        )

        return outputs.hidden_states[-1]

# --------------------------------------------------
# Train ATLAS for a given task pair
# --------------------------------------------------

def train_pairwise_atlas(
    base_model,
    base_param,
    finetuned_models,
    classifier_heads,
    task_names,
    lr,
    epochs,
    exclude_param,
):

    # filter params
    filter_func = lambda n, p: not any(
        re.match(exclude_pattern, n)
        for exclude_pattern in exclude_param
    )

    base_param.filter(filter_func)

    finetuned_params = []
    for name in task_names:
        p = param(finetuned_models[name])
        p.filter(filter_func)
        finetuned_params.append(p)

    param_names = list(base_param.param_dict.keys())

    task_vectors = build_task_vectors(
        base_param,
        finetuned_params,
        param_names,
    )

    atlas = AtlasEncoder(
        base_model,
        base_param,
        task_vectors,
        param_names,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW([atlas.coef], lr=lr)
    criterion = nn.CrossEntropyLoss()

    tokenizer = AutoTokenizer.from_pretrained(base_model.name_or_path)

    task_loaders = {
        name: get_glue_calib_loader(
            name,
            tokenizer,
            nsamples=200,
            seqlen=base_model.config.max_position_embeddings,
            batch_size=4,
        )
        for name in task_names
    }

    min_steps = min(len(dl) for dl in task_loaders.values())
    task_iters = {k: iter(v) for k, v in task_loaders.items()}

    for _ in range(epochs):
        atlas.train()
        for _ in range(min_steps):
            optimizer.zero_grad()
            loss = 0.0

            for tid, task in enumerate(task_names):
                try:
                    batch = next(task_iters[task])
                except StopIteration:
                    task_iters[task] = iter(task_loaders[task])
                    batch = next(task_iters[task])

                feats = atlas(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE)
                )
                logits = classifier_heads[task](feats)
                loss = loss + criterion(
                    logits,
                    batch["labels"].to(DEVICE),
                )

            loss.backward()
            optimizer.step()

    return atlas, param_names

# --------------------------------------------------
# Pairwise ATLAS disentanglement
# --------------------------------------------------
def run_pairwise_atlas_disentangle(args):

    models_finetuned = {
        name: utils.load_classifier(
            eval.model_path_template.format(name=name)
        ).to(DEVICE)
        for name in args.models_name
    }

    classifier_heads = {
        name: model.classifier.to(DEVICE).eval()
        for name, model in models_finetuned.items()
    }

    base_model = utils.load_classifier(args.base_model).to(DEVICE)
    base_param = param(base_model)

    data = utils.from_json(args.data_path)

    disentanglement = defaultdict(dict)

    for src_i, src_j in permutations(args.src_merge, 2):

        print(f"\n>>> Pairwise ATLAS train: {src_i} + {src_j}")

        # ---- train ATLAS for THIS pair ----
        base_param = param(base_model)
        atlas, param_names = train_pairwise_atlas(
            base_model=base_model,
            base_param=base_param,
            finetuned_models=models_finetuned,
            classifier_heads=classifier_heads,
            task_names=[src_i, src_j],
            lr=args.lr,
            epochs=args.epochs,
            exclude_param=args.exclude_param,
        )

        mismatches = 0
        total = 0

        for item in tqdm.tqdm(
            data, desc=f"eval {src_i} <- {src_j}"
        ):
            data_name = list(eval.glue_data_id_map.keys())[item["dataset_ids"]]
            if data_name != src_i:
                continue

            input_ids = torch.tensor(
                item["input_ids"], device=DEVICE
            ).unsqueeze(0)
            attention_mask = torch.tensor(
                item["attention_mask"], device=DEVICE
            ).unsqueeze(0)

            # finetuned
            logits_ft = models_finetuned[src_i](
                input_ids, attention_mask
            ).logits.detach().cpu().numpy()

            # atlas merged
            feats = atlas(input_ids, attention_mask)
            logits_mg = classifier_heads[src_i](feats).detach().cpu().numpy()

            mismatches += (get_pred(logits_ft) != get_pred(logits_mg)).sum()
            total += len(get_pred(logits_ft))

        disentanglement[src_i][src_j] = mismatches / max(total, 1)

        print(
            f"Disentanglement {src_i} <- {src_j}: "
            f"{disentanglement[src_i][src_j]:.4f}"
        )

    df = pd.DataFrame(disentanglement)
    out_path = os.path.join(args.outdir, "pairwise_atlas_disentanglement.csv")
    df.to_csv(out_path)
    print(f"\nSaved disentanglement matrix to {out_path}")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main(
    *,
    models_name: list[str],
    src_merge: list[str],
    base_model: str = "roberta-base",
    data_path: str,
    outdir: str,
    exclude_param: list[str],
    lr: float = 5e-3,
    epochs: int = 2,
    seed: int = 10,
):

    utils.fix_seed(seed)

    args = utils.SimpleNamespace(
        models_name=models_name,
        src_merge=src_merge,
        base_model=base_model,
        data_path=data_path,
        outdir=outdir,
        exclude_param=exclude_param,
        lr=lr,
        epochs=epochs,
    )

    run_pairwise_atlas_disentangle(args)

if __name__ == "__main__":
    import defopt
    defopt.run(main)
