set -eo pipefail

date_today=$(date '+%Y-%m-%d')
outdir=${outdir:="outs/atlas"}
mkdir -p ${outdir}


models_name=(
"cola"
"sst2"
"mrpc"
"qqp"
"mnli"
"qnli"
"rte"
"stsb"
)
models_to_merge=()
for d in "${models_name[@]}"; do
# models_to_merge+=(./local_roberta_models/textattack_roberta-base-$d)
models_to_merge+=(./roberta/$d/roberta-base_lr1e-05)
# models_to_merge+=(./linearlised/$d)
# models_to_merge+=(./results_${d}_noreg/best_model)
done
select_merge=${select_merge:="7"}


if [ $select_merge -eq 1 ]; then
    echo "please set \$select_merge > 1"
    exit 1 
fi
src_merge=("${models_name[@]:0:$select_merge}") 

echo ">>> merged from $select_merge tasks"
echo ">>> merge ${src_merge[@]}"

data_path="data/test.json"

function train(){
     python train_atlas.py \
    --models-name ${models_name[@]} \
    --src-merge ${src_merge[@]} \
    --data-path $data_path \
    --exclude-param ".*classifier.*" \
    --outdir $outdir
}

function disentangle(){
    python atlas_disentangle.py \
    --models-name ${models_name[@]} \
    --src-merge ${src_merge[@]} \
    --data-path $data_path \
    --exclude-param ".*classifier.*" \
    --outdir $outdir
}

function negation(){
    python atlas_negation.py \
    --models-name ${models_name[@]} \
    --data-path $data_path \
    --exclude-param ".*classifier.*" \
    --outdir $outdir
}

function negation_one(){
    python atlas_negation_one_control.py \
    --models-name ${models_name[@]} \
    --data-path $data_path \
    --exclude-param ".*classifier.*" \
    --outdir $outdir
}


case "${1:-}" in
  train) train ;;
  disentangle)   disentangle ;;
  negation)   negation ;;
  neg_one)   negation_one ;;
  *)
    echo "Usage: $0 {train|disentangle|negation|neg_one}"
    exit 1
    ;;
esac



