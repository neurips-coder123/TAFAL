import torch
from collections import defaultdict, OrderedDict
import tqdm
import re
import torch.nn as nn
import copy
import sparsify
import utils
import sys
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import os
import functools
from collections import defaultdict, OrderedDict
from param import param
from typing import List, Optional

class MergingMethod:

    @utils.args_inspector
    def __init__(
        self, 
        models_to_merge, 
        models_name,
    ):
        self.models_name = {n:i for i,n in enumerate(models_name)}
        # dict(zip(models_name, range(0, N)))
        self.models_to_merge = models_to_merge

    def get_model(self, model_name):
        return self.models_to_merge[self.models_name[model_name]]

    
    @utils.args_inspector
    @torch.inference_mode()
    def hessian(
        self,
        models_to_merge,      # List[(model, metric_dict)]
        base_model
    ):

        hessian_sum = {}
        task_vector_sum = {}
        for model, metric_dict in models_to_merge:
            delta = model - base_model  # param-like object
            for name, p in delta.param_dict.items():
                if name in metric_dict:
                    if name in hessian_sum:
                        hessian_sum[name] += metric_dict[name]
                    else:
                        hessian_sum[name] = metric_dict[name]
                    if name in task_vector_sum:
                        task_vector_sum[name] += p@metric_dict[name].to(p.device)
                    else:
                        task_vector_sum[name] = p@metric_dict[name].to(p.device)

        
        merged = copy.deepcopy(base_model)
        for key in merged.param_dict.keys():
            if key in hessian_sum:
                merged.param_dict[key] += task_vector_sum[key]@torch.linalg.inv(hessian_sum[key])
        return merged
    

    @utils.args_inspector
    @torch.inference_mode()
    def hessian_neg(
        self,
        models_to_merge,      # List[(model, metric_dict)]
        base_model,
        control_factor,
        neg_task_index
    ):
        
        hessian_sum = {}
        task_vector_sum = {}
        for i,(model, metric_dict) in enumerate(models_to_merge):
            delta = model - base_model  # param-like object
            for name, p in delta.param_dict.items():
                if name in metric_dict:
                    if i in neg_task_index:
                        metric_dict[name] = control_factor*metric_dict[name]
                        p = -p
                    if name in hessian_sum:
                        hessian_sum[name] += metric_dict[name]
                    else:
                        hessian_sum[name] = metric_dict[name]
                    if name in task_vector_sum:
                        task_vector_sum[name] += p@metric_dict[name].to(p.device)
                    else:
                        task_vector_sum[name] = p@metric_dict[name].to(p.device)

        
        merged = copy.deepcopy(base_model)
        for key in merged.param_dict.keys():
            if key in hessian_sum:
                merged.param_dict[key] += task_vector_sum[key]@torch.linalg.inv(hessian_sum[key])
        return merged


    @utils.args_inspector
    @torch.inference_mode()
    def task_arithmetic(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        scaling: float = 1.0,
    ):

        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        merged_param = base_model + scaling * sum(task_vectors)
        return merged_param

    @utils.args_inspector
    @torch.inference_mode()
    def task_arithmetic_neg(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        neg_task_index,
        scaling: float = 1.0,
    ):
        task_vectors = []
        for i, model in enumerate(models_to_merge):
            tv = model-base_model
            if i in neg_task_index:
                tv = base_model-model
            task_vectors.append(tv)
        merged_param = base_model + scaling * sum(task_vectors)
        return merged_param