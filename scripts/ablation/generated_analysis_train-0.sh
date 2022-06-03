export CUDA_VISIBLE_DEVICES=0

# task="rte"
task="SetFit/sst5"
benchmark="huggingface"

# task="cb"
# benchmark="super_glue"

# task="sst5"
# benchmark="huggingface"

# task='cb'
# benchmark="super_glue"

main_model="EleutherAI/gpt-j-6B"
# main_model="text-davinci-002"
# main_model="all-MiniLM-L12-v1"
main_path="./test_results/paper_results"


seeds="1"

n_sample="8"

for seed in $seeds; do

python correlation_analysis_train_set.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --output_dir $main_path/$task/$main_model/$n_sample-shot/correlation-train/$seed/ \
    --seed $seed \

done
