import torch
import torch.nn as nn
import numpy as np
import tqdm
import re
import os
from collections import defaultdict
from torch.func import functional_call
from transformers import AutoTokenizer

import utils
import eval
from param import param
from hessian import get_glue_calib_loader

DEVICE = "cuda:0"

# ============================================================
# Helpers
# ============================================================

def get_param_names(base_param):
    return list(base_param.param_dict.keys())

def build_task_vector(base_param, finetuned_param, param_names):
    return [
        finetuned_param[name] - base_param[name]
        for name in param_names
    ]

# ============================================================
# Negation Encoder (single control)
# θ = θ_base − α ⊙ Δ_target
# ============================================================

class NegationEncoder(nn.Module):
    def __init__(self, base_model, base_param, task_vector, param_names):
        super().__init__()

        self.base_model = base_model
        self.base_param = base_param
        self.task_vector = task_vector
        self.param_names = param_names

        self.num_params = len(param_names)
        self.coef = nn.Parameter(torch.zeros(self.num_params))

        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        merged = {}
        for i, name in enumerate(self.param_names):
            merged[name] = (
                self.base_param[name]
                - self.coef[i] * self.task_vector[i]
            )

        out = functional_call(
            self.base_model,
            merged,
            args=(input_ids, attention_mask),
            kwargs={"output_hidden_states": True},
        )
        return out.hidden_states[-1]

# ============================================================
# Train negation for ONE target task
# Control = all other tasks
# ============================================================

def train_negation_control_target(args, target_task):

    print(f"\n>>> TRAIN NEGATION (target={target_task})")

    filter_func = lambda n, p: not any(
        re.match(exclude_pattern, n)
        for exclude_pattern in args.exclude_param
    )

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # -------------------------
    # Base model
    # -------------------------
    base_model = utils.load_classifier(args.base_model).to(DEVICE)
    base_param = param(base_model)
    base_param.filter(filter_func)

    param_names = get_param_names(base_param)

    # -------------------------
    # Target finetuned model
    # -------------------------
    ft_target = utils.load_classifier(
        eval.model_path_template.format(name=target_task)
    ).to(DEVICE)

    ft_target_param = param(ft_target)
    ft_target_param.filter(filter_func)

    target_classifier = ft_target.classifier.to(DEVICE).eval()
    for p in target_classifier.parameters():
        p.requires_grad_(False)

    task_vector = build_task_vector(
        base_param,
        ft_target_param,
        param_names,
    )

    model = NegationEncoder(
        base_model,
        base_param,
        task_vector,
        param_names,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW([model.coef], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion_stsb = nn.MSELoss()

    # -------------------------
    # Data loaders
    # -------------------------
    target_loader = get_glue_calib_loader(
        target_task,
        tokenizer,
        nsamples=args.nsamples,
        seqlen=base_model.config.max_position_embeddings,
        batch_size=args.batch_size,
    )

    control_tasks = [t for t in args.models_name if t != target_task]
    control_loaders = {
        t: get_glue_calib_loader(
            t,
            tokenizer,
            nsamples=args.nsamples,
            seqlen=base_model.config.max_position_embeddings,
            batch_size=args.batch_size,
        )
        for t in control_tasks
    }

    control_iters = {k: iter(v) for k, v in control_loaders.items()}
    target_iter = iter(target_loader)

    min_steps = min(
        len(target_loader),
        *[len(dl) for dl in control_loaders.values()]
    )
    
    control_classifiers = {t: utils.load_classifier(
                    eval.model_path_template.format(name=t)
                ).classifier.to(DEVICE).eval() for t in args.models_name if t != target_task}

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for _ in range(min_steps):
            optimizer.zero_grad()
            loss = 0.0

            # ----- CONTROL LOSS (positive) -----
            for task in control_tasks:
                try:
                    batch = next(control_iters[task])
                except StopIteration:
                    control_iters[task] = iter(control_loaders[task])
                    batch = next(control_iters[task])

                feats = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE),
                )
                if not task=='stsb':
                    logits = control_classifiers[task](feats)

                    loss = loss + criterion(
                        logits,
                        batch["labels"].to(DEVICE),
                    )
                else:
                    vals = control_classifiers[task](feats)
                    loss = loss+1.0/25*criterion_stsb(vals[:,0], 
                                                    batch["labels"].to(DEVICE),)
            # ----- TARGET LOSS (negative) -----
            try:
                batch_t = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                batch_t = next(target_iter)

            feats_t = model(
                batch_t["input_ids"].to(DEVICE),
                batch_t["attention_mask"].to(DEVICE),
            )
            logits_t = target_classifier(feats_t)

            if target_task=="stsb":
                 loss = loss - args.lambda_neg *1.0/25* criterion_stsb(
                                logits_t[:,0],
                                batch_t["labels"].to(DEVICE),
                )
            else:
                loss = loss - args.lambda_neg * criterion(
                    logits_t,
                    batch_t["labels"].to(DEVICE),
                )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{target_task}] epoch {epoch} loss={total_loss/min_steps:.4f}")

    save_path = os.path.join(args.outdir, f"negation_{target_task}.pt")
    torch.save(model.coef.detach().cpu(), save_path)
    print(f"Saved negation model → {save_path}")

# ============================================================
# Inference: EACH CONTROL on ALL DATASETS
# ============================================================

@torch.inference_mode()
def run_negation_inference(args):

    print("\n>>> RUN NEGATION INFERENCE")

    filter_func = lambda n, p: not any(
        re.match(exclude_pattern, n)
        for exclude_pattern in args.exclude_param
    )

    base_model = utils.load_classifier(args.base_model).to(DEVICE)
    base_param = param(base_model)
    base_param.filter(filter_func)

    param_names = get_param_names(base_param)

    finetuned_models = {
        t: utils.load_classifier(
            eval.model_path_template.format(name=t)
        ).to(DEVICE)
        for t in args.models_name
    }

    classifiers = {
        t: m.classifier.to(DEVICE).eval()
        for t, m in finetuned_models.items()
    }

    finetuned_params = {}
    for t, m in finetuned_models.items():
        p = param(m)
        p.filter(filter_func)
        finetuned_params[t] = p

    data = utils.from_json(args.data_path)
    # ======================================================
    # Loop over controls
    # ======================================================
    for control in args.models_name:

        print(f"\n>>> CONTROL = {control}")

        coef = torch.load(
            os.path.join(args.outdir, f"negation_{control}.pt")
        ).to(DEVICE)

        task_vector = build_task_vector(
            base_param,
            finetuned_params[control],
            param_names,
        )

        eval_pred = defaultdict(lambda: defaultdict(list))

        for item in tqdm.tqdm(data, desc=f"infer control={control}"):

            dataset = list(eval.glue_data_id_map.keys())[item["dataset_ids"]]

            merged = {}
            for i, name in enumerate(param_names):
                merged[name] = (
                    base_param[name]
                    - coef[i] * task_vector[i]
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

            logits = classifiers[dataset](out.hidden_states[-1])

            eval_pred[dataset]["predictions"].append(logits.cpu().numpy())
            eval_pred[dataset]["label_ids"].append(item["label"])

        # -------------------------
        # Metrics
        # -------------------------
        metrics = {}
        scores = []

        for dataset in eval_pred:
            ans = eval.compute_single_metrics(
                utils.SimpleNamespace(
                    predictions=np.concatenate(eval_pred[dataset]["predictions"]),
                    label_ids=np.array(eval_pred[dataset]["label_ids"]),
                ),
                dataset,
            )["averaged_scores"]

            score = 100 * float(f"{ans:.4f}")
            metrics[dataset] = score
            scores.append(score)

        metrics["AVERAGE"] = sum(scores) / len(scores)

        utils.save_excel(
            metrics,
            os.path.join(args.outdir, f"negation_control_{control}.xlsx"),
        )

        print(f"\n>>> RESULTS FOR CONTROL {control}")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")

# ============================================================
# MAIN
# ============================================================

def main(
    *,
    models_name: list[str],
    base_model: str='./outs/task_arithmetic',
    data_path: str,
    outdir: str,
    exclude_param: list[str],
    lr: float = 5e-4,
    epochs: int = 5,
    nsamples: int = 560,
    batch_size: int = 4,
    lambda_neg: float = 1.0,
    seed: int = 10,
):

    utils.fix_seed(seed)

    args = utils.SimpleNamespace(
        models_name=models_name,
        base_model=base_model,
        data_path=data_path,
        outdir=outdir,
        exclude_param=exclude_param,
        lr=lr,
        epochs=epochs,
        nsamples=nsamples,
        batch_size=batch_size,
        lambda_neg=lambda_neg,
    )

    for task in models_name:
        train_negation_control_target(args, task)

    run_negation_inference(args)

if __name__ == "__main__":
    import defopt
    defopt.run(main)
