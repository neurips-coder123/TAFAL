
set -eo pipefail

date_today=$(date '+%Y-%m-%d')
outdir=${outdir:="outs/merge_results"}
mkdir -p ${outdir}


models_name=(
"cola"
"sst2"
"mrpc"
"stsb"
"qqp"
"mnli"
"qnli"
"rte"
)

models_to_merge=()
for d in "${models_name[@]}"; do
models_to_merge+=(./roberta/$d/roberta-base_lr1e-05)
done
select_merge=${select_merge:="8"}


function pos(){

# if [ $select_merge -eq 1 ]; then
#     echo "please set \$select_merge > 1"
#     exit 1 
# fi
src_merge=("${models_name[@]:0:$select_merge}") 

echo ">>> merged from $select_merge tasks"
echo ">>> merge ${src_merge[@]}"

data_path="data/test.json"
}

function run_task_arith(){

pos


for j in 0.29; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/task_arithmetic.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--scaling $j \
--outdir $outdir \
--save-path 'outs/task_arithmetic'

done

}

function run_task_arith_neg(){

pos


for j in 0.29; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/task_arithmetic_neg.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--scaling $j \
--outdir $outdir \
--save-path "outs/neg_models/ta/sst2"

done

}

function run_hessian(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/hessian.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--outdir $outdir 

done

}

function run_hessian_neg(){

pos


python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/hessian_neg.yml \
--exclude-param ".*classifier.*" ".*bias.*" \
--outdir $outdir \
--save-path './outs/neg_models/hess/stsb'

done

}

function ft(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--base-model 'roberta-base' \
--data-path $data_path \
--exclude-param ".*classifier.*" ".*bias.*" \
--outdir "outs/finetuned" 

}

function pretrain(){

pos

python run_merge.py \
--models-to-merge 'NONE' \
--models-name 'NONE' \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--base-model "outs/task_arithmetic" \
--outdir $outdir 

}

echo "ARG1=[$1]"
echo "DEBUG: models_to_merge=${models_to_merge[@]}"
echo "DEBUG: models_name=${models_name[@]}"
case "${1:-}" in
  task_arith) run_task_arith ;;
  ft)         ft ;;
  pretrain)   pretrain ;;
  hess)    run_hessian ;;
  hess_neg)    run_hessian_neg ;;
  ta_neg)    run_task_arith_neg ;;
  *)
    echo "Usage: $0 {task_arith|ft|pretrain|hess|hess_neg|ta_neg}"
    exit 1
    ;;
esac