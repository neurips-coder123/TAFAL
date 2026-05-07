import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import tqdm
import copy
import re
import os
import inspect
from collections import defaultdict
from itertools import permutations

import datasets
import transformers
from transformers import AutoTokenizer

import utils
import eval
from param import param
from merge import MergingMethod

DEVICE = "cuda:0"


# --------------------------------------------------
# Helper: prediction extraction
# --------------------------------------------------
def get_pred(logits):
    if logits.shape[1] > 1:
        return np.argmax(logits, axis=1)
    else:
        return (logits > 0).astype(int)


# --------------------------------------------------
# Pairwise merge + disentanglement
# --------------------------------------------------
@torch.inference_mode()
def run_pairwise_merge_and_disentangle(args):

    # ----------------------------
    # Load finetuned models θ_t
    # ----------------------------
    models_finetuned = {
        name: utils.load_classifier(
            eval.model_path_template.format(name=name)
        ).to(DEVICE)
        for name in args.models_name
    }

    # ----------------------------
    # Load base model θ_*
    # ----------------------------
    base_model = utils.load_classifier(args.base_model).to(DEVICE)

    # ----------------------------
    # Optional parameter filtering
    # ----------------------------
    if args.exclude_param and len(args.exclude_param):
        filter_func = lambda n, p: not any(
            re.match(exclude_pattern, n)
            for exclude_pattern in args.exclude_param
        )
    else:
        filter_func = lambda n, p: True

    # ----------------------------
    # Load dataset
    # ----------------------------
    data = utils.from_json(args.data_path)

    # ----------------------------
    # Storage
    # ----------------------------
    disentanglement = defaultdict(dict)

    # ==================================================
    # Pairwise loop
    # ==================================================
    for src_i, src_j in permutations(args.src_merge, 2):

        print(f"\n>>> Pairwise merge: {src_i} + {src_j}")

        # ----------------------------
        # Clone args and restrict merge
        # ----------------------------
        args_pair = copy.deepcopy(args)
        args_pair.src_merge = [src_i, src_j]

        # ----------------------------
        # Prepare models_to_merge
        # ----------------------------
        models_to_merge = [
            models_finetuned[name].to(DEVICE)
            for name in args_pair.src_merge
        ]
        if args_pair.merge_method == 'hessian':
            models_to_merge = [
                (models_finetuned[name].to(DEVICE), torch.load(f'./outs/hessian/256/{name}.pt'))
                for name in args.src_merge
            ]
            args_pair.models_to_merge = [(param(m),n) for m,n in models_to_merge]
            for model,_ in args_pair.models_to_merge:
                model.filter(filter_func)
        else:
            models_to_merge = [
                models_finetuned[name].to(DEVICE)
                for name in args_pair.src_merge
            ]
            args_pair.models_to_merge = [param(m) for m in models_to_merge]
            for model in args_pair.models_to_merge:
                model.filter(filter_func)

        args_pair.base_model = param(base_model)
        args_pair.base_model.filter(filter_func)

        merger = MergingMethod(**args_pair)
        merge_method = getattr(merger, args_pair.merge_method)
        merged_param = merge_method(**args_pair)

        # Materialize merged params
        merged_param_dict = merged_param.param_dict

        # ----------------------------
        # Evaluate disentanglement on task src_i
        # ----------------------------
        model_i = models_finetuned[src_i]
        mismatches = 0
        total = 0

        
        for data_item in tqdm.tqdm(
            data, desc=f"eval {src_i} vs ({src_i},{src_j})"
        ):
            data_name = list(eval.glue_data_id_map.keys())[data_item["dataset_ids"]]
            if data_name != src_i:
                continue

            input_ids = torch.tensor(
                data_item["input_ids"]
            ).unsqueeze(0).to(DEVICE)
            attention_mask = torch.tensor(
                data_item["attention_mask"]
            ).unsqueeze(0).to(DEVICE)

            # finetuned prediction
            logits_ft = model_i(
                input_ids, attention_mask
            ).logits.cpu().numpy()

            # merged prediction (functional, no overwrite)
            logits_mg = torch.func.functional_call(
                model_i,
                merged_param_dict,
                args=(input_ids, attention_mask),
            ).logits.cpu().numpy()

            pred_ft = get_pred(logits_ft)
            pred_mg = get_pred(logits_mg)

            mismatches += (pred_ft != pred_mg).sum()
            total += len(pred_ft)

        disentanglement[src_i][src_j] = mismatches / max(total, 1)

        print(
            f"Disentanglement {src_i} <- {src_j}: "
            f"{disentanglement[src_i][src_j]:.4f}"
        )

    # ----------------------------
    # Save results
    # ----------------------------
    df = pd.DataFrame(disentanglement)
    out_path = os.path.join(args.outdir, f"pairwise_disentanglement_talos.csv")
    df.to_csv(out_path)
    print(f"\nSaved disentanglement matrix to {out_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(
    *,
    models_to_merge: list[str],
    models_name: list[str],
    src_merge: list[str],
    yaml_file: str = None,
    exclude_param: list[str] = None,
    data_path: str = None,
    seed: int = 10,
    base_model: str = "roberta-base",
    scaling: list[float] = None,
    mask_rate: float = None,
    mask_scale: float = None,
    maskllm: bool = False,
    mask_strategy: str = None,
    outdir: str = None,
    save_path: str = None,
    tall_masks_lambda: float = 0.2
):

    global args
    keys, _, _, values = inspect.getargvalues(inspect.currentframe())
    utils.fix_seed(seed)
    merge_config = utils.from_yaml(yaml_file)    
    args = {
        k: values.get(k, merge_config.get(k)) 
        for k in set(keys).union(merge_config)
    }
    args = {
        k: merge_config.get(k, None)
        if args[k] is None else args[k]
        for k in args.keys()
    }
    args = utils.SimpleNamespace(**args)

    print('>>> args\n', args)

    if args.scaling is not None and isinstance(args.scaling, list) and len(args.scaling) == 1:
        args.scaling = args.scaling[0]

    run_pairwise_merge_and_disentangle(args)


if __name__ == "__main__":
    import defopt
    defopt.run(main)
