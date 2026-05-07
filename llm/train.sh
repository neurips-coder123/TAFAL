#!/usr/bin/env bash

# DATASETS=("sst2" "mrpc" "stsb" "qqp" "qnli" "rte" "cola")
# DATASETS=("toxic" "non-toxic")
DATASETS=("math" "code")
# DATASETS=("mnli")

for ds in "${DATASETS[@]}"; do
  python hessian_lm.py \
    --dataset "${ds}" 
done
