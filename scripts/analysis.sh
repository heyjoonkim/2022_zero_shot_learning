export CUDA_VISIBLE_DEVICES=3

# task="sst2"
# task="mrpc"
# task="rte"
# task="wnli"
# benchmark="glue"


task="cb"
benchmark="super_glue"

# task="trec"
# task="SetFit/sst5"
# task="ag_news"
# benchmark="huggingface"

python analysis.py \
    --task_name $task \
    --benchmark_name $benchmark