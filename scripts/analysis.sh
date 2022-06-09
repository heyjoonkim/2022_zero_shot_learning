
export CUDA_VISIBLE_DEVICES=3

# task="trec"
# task="sst5"
task="mrpc"
benchmark="glue"
split="train"
split="validation"

python analysis.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --split $split