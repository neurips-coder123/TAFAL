import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from alignment import evaluate_hh
from itertools import product

def merge_and_eval(alpha, co_neg,
                   base_sd, chosen_sd, rejected_sd,
                   cov_chosen, cov_rej,
                   model, tokenizer):

    merged_sd = copy.deepcopy(base_sd)

    for name in cov_chosen:
        if name not in merged_sd or name not in cov_rej:
            continue

        W0 = base_sd[name]
        if W0.ndim != 2:
            continue

        Cc = cov_chosen[name].to(W0.device).float()
        Cr = cov_rej[name].to(W0.device).float()

        # ---- Hessian ----
        H = Cc + co_neg * Cr
        d = H.shape[0]
        H = H + 1e-4 * torch.eye(d, device=H.device, dtype=H.dtype)

        H_inv = torch.linalg.solve(
            H, torch.eye(d, device=H.device, dtype=H.dtype)
        )

        # ---- deltas ----
        Wc = (chosen_sd[name]   - base_sd[name]).float()
        Wr = (rejected_sd[name] - base_sd[name]).float()

        rhs = Wc @ Cc - co_neg * (Wr @ Cr)

        delta = rhs @ H_inv
        merged_sd[name] = base_sd[name] + alpha * delta.to(W0.dtype)

    model.load_state_dict(merged_sd, strict=False)
    model.eval()

    with torch.no_grad():
        val = evaluate_hh(model, tokenizer)

    return val




if __name__ == "__main__":

    model_name = "EleutherAI/pythia-410m-v0"

    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    cho_model  = AutoModelForCausalLM.from_pretrained("lomahony/eleuther-pythia410m-hh-sft", device_map="cpu")
    rej_model  = AutoModelForCausalLM.from_pretrained('./outs/alignment/pythia4_rej', device_map="cpu")
    # cho_model  = AutoModelForCausalLM.from_pretrained("./outs/alignment/pythia1b_chosen", device_map="cpu")
    # rej_model  = AutoModelForCausalLM.from_pretrained("./outs/alignment/pythia1b_rejected", device_map="cpu")
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    base_sd    = base_model.state_dict()
    chosen_sd  = cho_model.state_dict()
    # chosen_sd = copy.deepcopy(base_sd)
    rejected_sd = rej_model.state_dict()

    cov_chosen = torch.load("./outs/pythia_hessian/400m_base_chosen_256.pt", map_location="cpu")
    cov_rej    = torch.load("./outs/pythia_hessian/400m_base_rejected_256.pt", map_location="cpu")

    # -------------------------
    # Search space (tight + sane)
    # -------------------------
    alpha_grid   = [1, 0.05, 0.1, 0.2, 0.4]
    co_neg_grid  = [0.25, 0.5, 1.0]

    best_val = -float("inf")
    best_cfg = None

    for alpha, co_neg in product(alpha_grid, co_neg_grid):
        print(f"Testing alpha={alpha}, co_neg={co_neg}")

        val = merge_and_eval(
            alpha, co_neg,
            base_sd, chosen_sd, rejected_sd,
            cov_chosen, cov_rej,
            base_model, tokenizer
        )

        print(f"  val = {val:.4f}")

        if val > best_val:
            best_val = val
            best_cfg = (alpha, co_neg)

    print("\n=== BEST ===")
    print(f"alpha={best_cfg[0]}, co_neg={best_cfg[1]}")
    print(f"val={best_val:.4f}")

