# ============================================================
# TaLoS for RoBERTa — SEQUENTIAL (CORRECT)
# ============================================================

import os, gc, torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm import tqdm
from eval import head_path_template
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "roberta-base"
SAVE_DIR = "./talos_roberta"
os.makedirs(SAVE_DIR, exist_ok=True)

GLUE_TASKS = ["cola","sst2","mrpc","qqp","mnli","qnli","rte","stsb"]

NUM_LABELS = {
    "cola":2,"sst2":2,"mrpc":2,"qqp":2,"mnli":3,"qnli":2,"rte":2,"stsb":1
}

TASK_EPOCHS = {
    "cola":30,"sst2":8,"mrpc":25,"qqp":6,
    "mnli":4,"qnli":6,"rte":35,"stsb":40
}

BATCH_SIZE = 32
WEIGHT_DECAY = 0.01

# ============================================================
# DATA
# ============================================================

def get_dataset(task):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    keys = {
        "cola":("sentence",None),
        "sst2":("sentence",None),
        "mrpc":("sentence1","sentence2"),
        "qqp":("question1","question2"),
        "mnli":("premise","hypothesis"),
        "qnli":("question","sentence"),
        "rte":("sentence1","sentence2"),
        "stsb":("sentence1","sentence2"),
    }
    split = "validation_matched" if task=="mnli" else "train"
    ds = load_dataset("glue", task, split=split)

    def tok(ex):
        out = tokenizer(
            ex[keys[task][0]],
            ex[keys[task][1]] if keys[task][1] else None,
            truncation=True
        )
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tok, remove_columns=ds.column_names)
    return ds, tokenizer

# ============================================================
# TaLoS CORE
# ============================================================

def talos_mask_calibration(
    model,
    dataset,        
    task,         
    NSAMPLES=256,
    ROUNDS=4,
    FINAL_KEEP=0.5,
    DEVICE="cuda"
):
    model = model.to(DEVICE)

    # trainable parameters only
    params = [p for p in model.parameters() if p.requires_grad]

    # binary masks
    mask = {
        p: torch.ones_like(p, dtype=torch.bool, device=DEVICE)
        for p in params
    }

    N = len(dataset)
    assert NSAMPLES <= N, "NSAMPLES must be <= dataset size"

    for r in range(1, ROUNDS + 1):
        print(f"[TaLoS] Round {r}/{ROUNDS}")

        scores = {
            p: torch.zeros_like(p, device=DEVICE)
            for p in params
        }

        # -------- sample from DATASET --------
        indices = random.sample(range(N), NSAMPLES)

        model.eval()
        for idx in tqdm(indices, desc=f"Scoring r={r}"):
            ex = dataset[idx]

            batch = {
                "input_ids": torch.tensor(ex["input_ids"]).unsqueeze(0).to(DEVICE),
                "attention_mask": torch.tensor(ex["attention_mask"]).unsqueeze(0).to(DEVICE),
                "labels": torch.tensor(ex["labels"]).unsqueeze(0).to(DEVICE),
            }

            model.zero_grad(set_to_none=True)
            logits = model(**batch).logits

            # ---- TaLoS loss ----
            if task == "stsb":
                loss = F.mse_loss(
                    logits.squeeze(),
                    batch["labels"].float()
                )
            else:
                y = torch.distributions.Categorical(logits=logits).sample()
                logp = F.log_softmax(logits, dim=-1)
                loss = logp.gather(1, y.unsqueeze(1)).sum()

            loss.backward()

            for p in params:
                if p.grad is not None:
                    scores[p] += (p.grad * mask[p]).pow(2)

        # -------- progressive sparsity --------
        keep_frac = FINAL_KEEP ** (r / ROUNDS)

        active_scores = torch.cat([
            scores[p][mask[p]].flatten()
            for p in params
            if mask[p].any()
        ])

        k = max(1, int(keep_frac * active_scores.numel()))
        threshold = torch.kthvalue(active_scores, k).values

        # cumulative masking
        for p in params:
            mask[p] &= (scores[p] <= threshold)

        remaining = sum(m.sum().item() for m in mask.values())
        total = sum(m.numel() for m in mask.values())

        print(f"Remaining params: {remaining/total:.4f}")

    # -------- FINAL SAFEGUARD --------
    remaining = sum(m.sum().item() for m in mask.values())
    total = sum(m.numel() for m in mask.values())

    min_keep = int(FINAL_KEEP * total)
    if remaining < min_keep:
        print("[TaLoS] Safeguard triggered — enforcing FINAL_KEEP")

        all_scores = torch.cat([scores[p].flatten() for p in params])
        threshold = torch.kthvalue(all_scores, min_keep).values

        for p in params:
            mask[p] = (scores[p] <= threshold)

    return mask

# ============================================================
# TRAIN
# ============================================================

def train_task(task):
    print(f"\n=== TaLoS | {task.upper()} ===")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS[task]
    )
    model.classifier = torch.load(head_path_template.format(name=task), weights_only=False)
    model.to(DEVICE)

    for p in model.classifier.parameters():
        p.requires_grad_(False)

    dataset, tokenizer = get_dataset(task)

    mask = talos_mask_calibration(model, dataset, task)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer)
    )
    # gradient masking
    for p, m in mask.items():
        p.register_hook(lambda g, m=m: g * m)

    lr = 5e-6 if task in {"cola","rte","stsb"} else 1e-5
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    for epoch in range(TASK_EPOCHS[task]):
        model.train()
        total = 0
        for batch in loader:
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            logits = model(**batch).logits
            loss = (
                F.mse_loss(logits.squeeze(), batch["labels"].float())
                if task=="stsb"
                else F.cross_entropy(logits, batch["labels"])
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item()
        print(f"Epoch {epoch} | loss {total/len(loader):.4f}")

    model.save_pretrained(os.path.join(SAVE_DIR, task))
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    for t in GLUE_TASKS:
        train_task(t)
