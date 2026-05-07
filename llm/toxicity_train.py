import os
import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from find_toxicity_score import find_toxicity_score

# -----------------------
# Config
# -----------------------
MODEL_NAME = "gpt2-large"
OUTPUT_DIR = "./outs/gpt2_toxicity_large"
MAX_LENGTH = 512
TOXICITY_THRESHOLD = 0.8
SEED = 42

PER_DEVICE_BATCH_SIZE = 24   # ← per GPU
EPOCHS = 5
LR = 1e-5

set_seed(SEED)

# -----------------------
# Load Civil Comments
# -----------------------
dataset = load_dataset("civil_comments")

def filter_toxic(example):
    return example["toxicity"] is not None and example["toxicity"] > TOXICITY_THRESHOLD

dataset = dataset.filter(filter_toxic)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# -----------------------
# Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    out = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_dataset = train_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

# -----------------------
# Model
# -----------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# -----------------------
# Data collator
# -----------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -----------------------
# Training args (DDP-ready)
# -----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,

    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,

    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    logging_steps=50,
    report_to="none",

    fp16=True,                     # OK for homogeneous GPUs
    dataloader_num_workers=4,

    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

# -----------------------
# Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -----------------------
# Train
# -----------------------
trainer.train()

# -----------------------
# Save (rank 0 only)
# -----------------------
if trainer.is_world_process_zero():
    find_toxicity_score(trainer.model, tokenizer)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
