# ============================================================
# ATLAS vs FINETUNED — FINAL-LAYER TASK ACTIVATION ANALYSIS
# ============================================================

import os
import gc
import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.func import functional_call
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F

import utils
import eval
from param import param

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "roberta-base"
OUTDIR = "./final_layer_analysis"
PLOTDIR = os.path.join(OUTDIR, "plots_ta")
os.makedirs(PLOTDIR, exist_ok=True)

TASKS_2 = ["cola","sst2","mrpc","qqp","mnli","qnli","rte","stsb"]
TASKS = ["sst2"]
TASKS = TASKS_2
NSAMPLES = 128
METHOD = 'BASE'



# ============================================================
# SAVE FORMAT (as requested)
# ============================================================

def save_excel(data, out_path):
    columns = sorted(list(data.keys()))
    df = pd.DataFrame(data, index=[0]).reindex(columns=columns)

    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(out_path, "results.csv")
    md_path = os.path.join(out_path, "results.md")

    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path, index_col=0)
        df = pd.concat([prev, df])

    df.to_csv(csv_path)

    md = tabulate(df, headers="keys", tablefmt="pipe")
    with open(md_path, "w") as f:
        f.write(md)

    print(md)

# ============================================================
# HELPERS
# ============================================================

def is_encoder_param(name):
    return not name.startswith("classifier.")

GLUE_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2")
}

def mean_pool(h):
    return h.mean(dim=1).mean(dim=0)

def get_inputs(tokenizer, task):
    s1, s2 = GLUE_KEYS[task]
    split = "validation_matched" if task == "mnli" else "validation"

    ds = load_dataset("glue", task, split=split).shuffle(seed=0)
    ds = ds.select(range(NSAMPLES))

    texts1 = [x if isinstance(x, str) else "" for x in ds[s1]]

    if s2 is None:
        tok = tokenizer(
            texts1,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
    else:
        texts2 = [x if isinstance(x, str) else "" for x in ds[s2]]
        tok = tokenizer(
            texts1,
            texts2,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    return tok.to(DEVICE)

# ============================================================
# LOAD BASE + ATLAS (ONCE)
# ============================================================

for TASK in TASKS:

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    base_model = utils.load_classifier(BASE_MODEL).eval()
    base_param = param(base_model)
    base_param.filter(lambda n, _: is_encoder_param(n))
    param_names = list(base_param.param_dict.keys())

    if METHOD=='TASK_ARITHMETIC':
        merged = utils.load_classifier(f'./outs/neg_models/ta/{TASK}').to(DEVICE).eval()
    elif METHOD=='HESSIAN':
        merged = utils.load_classifier(f'./outs/neg_models/hess/{TASK}').to(DEVICE).eval()
    elif METHOD=='TALOS':
        merged = utils.load_classifier(f'./outs/neg_models/talos/{TASK}').to(DEVICE).eval()
    elif METHOD=='BASE':
        merged = utils.load_classifier(f'roberta-base').to(DEVICE).eval()
    else:
        raise NotImplementedError
    # ============================================================
    # ANALYSIS
    # ============================================================
    params = param(merged)
    params.filter(lambda n, _: is_encoder_param(n))
    params = params.param_dict
    stats = {}
    base_model = base_model.to(DEVICE)
    stats["Method"] = f"{TASK}_{METHOD}_negation"
    for t in TASKS_2:
        print(f"\n>>> Task: {t}")

        inputs = get_inputs(tokenizer, t)

        with torch.no_grad():
            # finetuned
            ft_model = utils.load_classifier(
                eval.model_path_template.format(name=t)
            ).to(DEVICE).eval()

            ft_out = ft_model(
                **inputs, output_hidden_states=True
            ).hidden_states[-1]

            # atlas (same model for all tasks)
            atlas_out = functional_call(
                base_model,
                params,
                kwargs={**inputs, "output_hidden_states": True},
            ).hidden_states[-1]

        hf = mean_pool(ft_out)
        ha = mean_pool(atlas_out)

        cos = F.cosine_similarity(hf, ha, dim=0).item()
        l2 = torch.norm(hf - ha).item()

        stats[f"{t}_final_cosine"] = cos
        stats[f"{t}_final_l2"] = l2

        del ft_model, ft_out, atlas_out, inputs
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # SAVE
    # ============================================================

    save_excel(stats, OUTDIR)
    print("\nSaved final-layer task activation analysis →", OUTDIR)
