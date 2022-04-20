
export CUDA_VISIBLE_DEVICES=3

# task="trec"
task="sst5"

python analysis.py \
    --task_name $task