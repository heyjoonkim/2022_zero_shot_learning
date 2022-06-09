export CUDA_VISIBLE_DEVICES=7

# task #
task="sst2"
benchmark="glue"

# model #
# main_model="EleutherAI/gpt-j-6B"
main_model="EleutherAI/gpt-neox-20b"
# main_model="gpt2"

main_path="./debug"
dataset_path="./data"

seeds="42"

n_samples="16"


# select in-context samples from train set. (RANDOM)
for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --output_dir $main_path/$task-seed_$seed-k_$n_samples \
    --train_set $dataset_path/train_${n_samples}_${seeds}.jsonl \
    --test_set $dataset_path/test.jsonl \
    --seed $seed \
    --overwrite_output_dir \
    --n_samples $n_samples \
    --prefix '' \
    --infix '
' \
    --postfix ''
done   
