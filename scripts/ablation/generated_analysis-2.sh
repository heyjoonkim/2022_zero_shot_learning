export CUDA_VISIBLE_DEVICES=2

# task="rte"
task="SetFit/sst5"
benchmark="huggingface"

# task="cb"
# benchmark="super_glue"

# task="sst5"
# benchmark="huggingface"

# task='cb'
# benchmark="super_glue"

dataset_path="./generated_datasets"
# dataset_path="./generated_datasets-ablation"
# 
main_model="EleutherAI/gpt-j-6B"
# main_model="text-davinci-002"
# main_model="all-MiniLM-L12-v1"
main_path="./test_results/paper_results"

generation_template="template1"

seeds="13 21 42 87 100"

n_sample="8"

for seed in $seeds; do

python correlation_analysis.py \
    --task_name $task \
    --dataset_dir $dataset_path/$task/$main_model/$generation_template/$n_sample-shot/$seed/ \
    --output_dir $main_path/$task/$main_model/$n_sample-shot/correlation/$seed/ \
    --seed $seed \

done
