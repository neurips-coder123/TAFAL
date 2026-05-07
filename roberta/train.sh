#!/usr/bin/env bash

DATASETS=("cola" "sst2" "mrpc" "stsb" "qqp" "qnli" "rte" "stsb")

for ds in "${DATASETS[@]}"; do
  python hessian.py \
    --dataset "${ds}" 
done
