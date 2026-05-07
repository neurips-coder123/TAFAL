# Single-file, copy-paste evaluation for Anthropic HH preference alignment

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def split_hh_sample(text: str):
    """
    Split Anthropic HH text into:
      context: everything up to and including the last 'Assistant:'
      completion: the final assistant reply
    """
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        raise ValueError("No Assistant marker found")
    context = text[: idx + len(marker)]
    completion = text[idx + len(marker) :]
    return context, completion

@torch.no_grad()
def logprob_of_completion(model, tokenizer, context, completion):
    max_len = model.config.max_position_embeddings

    # Tokenize separately (no tensors yet)
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    comp_ids = tokenizer(completion, add_special_tokens=False).input_ids

    # Ensure completion is kept intact
    if len(comp_ids) >= max_len:
        # pathological case: completion alone too long
        comp_ids = comp_ids[-(max_len - 1):]

    # Trim context from the left if needed
    total_len = len(ctx_ids) + len(comp_ids)
    if total_len > max_len:
        overflow = total_len - max_len
        ctx_ids = ctx_ids[overflow:]

    # Rebuild full sequence
    input_ids = torch.tensor([ctx_ids + comp_ids], device=DEVICE)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]

    ctx_len = len(ctx_ids)

    # Only score completion tokens
    logits = logits[:, ctx_len - 1 :]
    labels = labels[:, ctx_len - 1 :]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum().item()

def evaluate_hh(model, tokenizer, split="test",max_samples=None):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(DEVICE).eval()

    dataset = load_dataset("Anthropic/hh-rlhf", split=split)

    deltas = []
    wins = 0
    n = 0

    for ex in tqdm(dataset):
        if max_samples and n >= max_samples:
            break

        ctx_c, comp_c = split_hh_sample(ex["chosen"])
        ctx_r, comp_r = split_hh_sample(ex["rejected"])

        # contexts must match
        if ctx_c != ctx_r:
            continue

        lp_c = logprob_of_completion(model, tokenizer, ctx_c, comp_c)
        lp_r = logprob_of_completion(model, tokenizer, ctx_c, comp_r)

        delta = lp_c - lp_r
        deltas.append(delta)
        wins += delta > 0
        n += 1

    print(f"num_samples: {n}\n mean_logprob_gap: {float(sum(deltas) / max(1, n))}\n win_rate: {float(wins / max(1, n))}")
    return float(wins/max(1,n))


def evaluate_hh_with_name(model_name, split="test", max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    
    evaluate_hh(model, tokenizer)

def evaluate_hh_with_sd(model_name, sd_path, split="test", max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.load_state_dict(torch.load(sd_path)['state'])

    evaluate_hh(model, tokenizer)

    

if __name__ == "__main__":

    models = [
        # ("EleutherAI/pythia-2.8b",'/data8/rajiv/mariamma/sam/direct-preference-optimization/.cache/rajivporana/anthropic_dpo_pythia28_2026-01-24_17-24-51_877215/step-59904/policy.pt'),
        # ("EleutherAI/pythia-2.8b",'/data8/rajiv/mariamma/sam/direct-preference-optimization/.cache/rajivporana/anthropic_dpo_pythia28_2026-01-24_15-41-44_496358/LATEST/policy.pt'),
        # ("EleutherAI/pythia-2.8b",'/data8/rajiv/mariamma/sam/direct-preference-optimization/.cache/rajivporana/anthropic_dpo_pythia28_2026-01-24_12-15-55_294607/LATEST/policy.pt')
        # ('EleutherAI/pythia-1b-v0','../../direct-preference-optimization/.cache/rajivporana/anthropic_sft_rej_pythia10_2026-01-26_21-17-48_267806/LATEST/policy.pt')
        # "meta-llama/Llama-3.2-1B"
        # "meta-llama/Llama-3.2-1B-Instruct"
        # ,"meta-llama/Llama-Guard-3-1B"  
        # "/data8/rajiv/mariamma/sam/HALOs/models/qwen_sft/FINAL"  
        # "Qwen/Qwen2.5-0.5B",
        # "Leogrin/eleuther-pythia1b-hh-dpo"
        # "/data8/rajiv/mariamma/sam/Twin-Merging/discriminative/outs/alignment/pythia1b_chosen_2"
        # "lomahony/eleuther-pythia410m-hh-dpo",
        # "lomahony/eleuther-pythia410m-hh-sft",
        ("EleutherAI/pythia-410m-v0","../../direct-preference-optimization/.cache/rajivporana/pythia_410m_rej/policy.pt")
    ]

    for m in models:
        print(evaluate_hh_with_sd(m[0],m[1]))
        # print(evaluate_hh_with_name(m))
