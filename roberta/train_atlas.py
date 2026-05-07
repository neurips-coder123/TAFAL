import torch
import torch.nn as nn
import tqdm
import re
import inspect
import numpy as np
from collections import defaultdict
from torch.func import functional_call
from transformers import AutoTokenizer

import utils
import eval
from param import param
from torch.utils.data import DataLoader
from hessian import get_glue_calib_loader

DEVICE = "cuda:0"

# ------------------------------------------------------------
# RoBERTa helpers
# ------------------------------------------------------------

def get_param_names(base_param):
    return list(base_param.param_dict.keys())

# ------------------------------------------------------------
# Task vector construction (param-based)
# ------------------------------------------------------------

def build_task_vectors(base_param, finetuned_params, param_names):
    """
    task_vectors[task_id][param_id] -> tensor
    """
    task_vectors = []

    for ft_param in finetuned_params:
        tv = []
        for name in param_names:
            tv.append(ft_param[name] - base_param[name])
        task_vectors.append(tv)

    return task_vectors

# ------------------------------------------------------------
# ATLAS encoder (only scalars are trained)
# ------------------------------------------------------------

class AtlasEncoder(nn.Module):
    def __init__(self, base_model, base_param, task_vectors, param_names):
        super().__init__()

        self.base_model = base_model
        self.base_param = base_param
        self.task_vectors = task_vectors
        self.param_names = param_names

        self.num_tasks = len(task_vectors)
        self.num_params = len(param_names)

        self.coef = nn.Parameter(
            torch.zeros(self.num_tasks, self.num_params)
        )

        # freeze base model
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

# ------------------------------------------------------------
# TRAIN ATLAS
# ------------------------------------------------------------

def train_atlas(args):

    # -------------------------
    # filter function (CRITICAL)
    # -------------------------

    filter_func = lambda n, p: not any(
        re.match(exclude_pattern, n)
        for exclude_pattern in args.exclude_param
    )

    # -------------------------
    # Load base model + filter
    # -------------------------

    base_model = utils.load_classifier(args.base_model).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_param = param(base_model)
    base_param.filter(filter_func)

    # -------------------------
    # Load finetuned models + filter
    # -------------------------

    finetuned_models = [
        utils.load_classifier(
            eval.model_path_template.format(name=name)
        ).to(DEVICE)
        for name in args.models_name
    ]

    finetuned_params = []
    classifier_heads = {}

    for name, model in zip(args.models_name, finetuned_models):
        p = param(model)
        p.filter(filter_func)
        finetuned_params.append(p)

        classifier_heads[name] = model.classifier.to(DEVICE).eval()
        for q in classifier_heads[name].parameters():
            q.requires_grad_(False)

    # -------------------------
    # Build task vectors
    # -------------------------

    param_names = get_param_names(base_param)

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

    optimizer = torch.optim.AdamW([atlas.coef], lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    criterion_stsb = nn.MSELoss()
    
    task_loaders = {
        name: get_glue_calib_loader(
            name,
            tokenizer,
            nsamples=200,
            seqlen=base_model.config.max_position_embeddings, 
            batch_size=4
        )
        for name in args.models_name
    }

    min_steps = min(len(dl) for dl in task_loaders.values())
    task_iters = {k: iter(v) for k, v in task_loaders.items()}

    # -------------------------
    # Training loop
    # -------------------------

    for epoch in range(args.epochs):
        atlas.train()
        total_loss = 0.0

        for step in range(min_steps):
            optimizer.zero_grad()
            loss = 0.0

            for data_name, loader in task_loaders.items():
                try:
                    batch = next(task_iters[data_name])
                except StopIteration:
                    task_iters[data_name] = iter(loader)
                    batch = next(task_iters[data_name])

                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                feats = atlas(input_ids, attention_mask)
                logits = classifier_heads[data_name](feats)
                
                if data_name == 'stsb':
                    loss += 1.0/5*criterion_stsb(logits.squeeze(-1), labels)
                else:
                    loss += criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[epoch {epoch}] loss = {total_loss / min_steps:.4f}")

    torch.save(atlas.coef.detach().cpu(), args.atlas_weight_path)
    return atlas.coef.detach()

# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------

@torch.inference_mode()
def run_atlas_infer(args):

    filter_func = lambda n, p: not any(
        re.match(exclude_pattern, n)
        for exclude_pattern in args.exclude_param
    )

    base_model = utils.load_classifier(args.base_model).to(DEVICE)
    base_param = param(base_model)
    base_param.filter(filter_func)

    finetuned_models = {
        name: utils.load_classifier(
            eval.model_path_template.format(name=name)
        ).to(DEVICE)
        for name in args.models_name
    }

    finetuned_params = []
    classifier_heads = {}

    for name, model in finetuned_models.items():
        p = param(model)
        p.filter(filter_func)
        finetuned_params.append(p)

        classifier_heads[name] = model.classifier.to(DEVICE).eval()

    param_names = get_param_names(base_param)
    task_vectors = build_task_vectors(base_param, finetuned_params, param_names)
    atlas_weights = torch.load(args.atlas_weight_path).to(DEVICE)

    data = utils.from_json(args.data_path)
    eval_pred = defaultdict(lambda: defaultdict(list))

    for item in tqdm.tqdm(data, desc="infer glue (ATLAS)"):

        data_name = list(eval.glue_data_id_map.keys())[item["dataset_ids"]]
        if data_name not in args.src_merge:
            continue

        task_id = args.models_name.index(data_name)

        merged = {}
        for i, name in enumerate(param_names):
            merged[name] = (
                base_param[name]
                + atlas_weights[task_id, i] * task_vectors[task_id][i]
            )

        input_ids = torch.tensor(
            item["input_ids"], device=DEVICE
        ).unsqueeze(0)
        attention_mask = torch.tensor(
            item["attention_mask"], device=DEVICE
        ).unsqueeze(0)

        out = functional_call(
            base_model,
            merged,
            args=(input_ids, attention_mask),
            kwargs={"output_hidden_states": True},
        )

        cls = out.hidden_states[-1]
        logits = classifier_heads[data_name](cls)

        eval_pred[data_name]["predictions"].append(logits.cpu().numpy())
        eval_pred[data_name]["label_ids"].append(item["label"])

    metrics = {}
    for name in eval_pred:
        ans = eval.compute_single_metrics(
            utils.SimpleNamespace(
                predictions=np.concatenate(eval_pred[name]["predictions"]),
                label_ids=np.array(eval_pred[name]["label_ids"]),
            ),
            name,
        )["averaged_scores"]

        metrics[name] = 100 * float(f"{ans:.4f}")

    utils.save_excel(metrics, args.outdir)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main(
    *,
    models_name: list[str],
    src_merge: list[str],
    base_model: str='roberta-base',
    data_path: str,
    outdir: str,
    atlas_weight_path: str='./outs/atlas/atlas_weights.pt',
    exclude_param: list[str],
    lr: float = 5e-3,
    epochs: int = 80,
    seed: int = 10,
    weight_decay: float = 1e-2
):

    utils.fix_seed(seed)

    args = utils.SimpleNamespace(
        models_name=models_name,
        src_merge=src_merge,
        base_model=base_model,
        data_path=data_path,
        outdir=outdir,
        atlas_weight_path=atlas_weight_path,
        exclude_param=exclude_param,
        lr=lr,
        epochs=epochs,
        weight_decay=weight_decay
    )

    print(">>> TRAIN ATLAS")
    train_atlas(args)

    print(">>> INFER ATLAS")
    run_atlas_infer(args)

if __name__ == "__main__":
    import defopt
    defopt.run(main)
