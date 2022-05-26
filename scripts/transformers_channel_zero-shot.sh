export CUDA_VISIBLE_DEVICES=2


# model #
main_model="EleutherAI/gpt-j-6B"
# main_model="gpt2"

main_path="./few_shot"

# seeds="13 21 42 87 100"
seeds="0"

n_samples="0"

# task #
task="mrpc"
benchmark="glue"

## Minimal template ##
# BALANCED # 
for seed in $seeds; do
    python transformers_channel_main.py \
        --task_name $task \
        --benchmark_name $benchmark \
        --model_name_or_path $main_model \
        --output_dir $main_path/channel/$benchmark-$task-seed_$seed-zero_shot-minimal-neutral \
        --seed $seed \
        --n_samples $n_samples \
        --overwrite_output_dir \
        --prefix '' \
        --postfix ''
done

# task #
task="rte"
benchmark="glue"

## Minimal template ##
# BALANCED # 
for seed in $seeds; do
    python transformers_channel_main.py \
        --task_name $task \
        --benchmark_name $benchmark \
        --model_name_or_path $main_model \
        --output_dir $main_path/channel/$benchmark-$task-seed_$seed-zero_shot-minimal-neutral \
        --seed $seed \
        --n_samples $n_samples \
        --overwrite_output_dir \
        --prefix '' \
        --postfix ''
done

# task #
task="hate"
benchmark="tweet_eval"

## Minimal template ##
# BALANCED # 
for seed in $seeds; do
    python transformers_channel_main.py \
        --task_name $task \
        --benchmark_name $benchmark \
        --model_name_or_path $main_model \
        --output_dir $main_path/channel/$benchmark-$task-seed_$seed-zero_shot-minimal-neutral \
        --seed $seed \
        --n_samples $n_samples \
        --overwrite_output_dir \
        --prefix '' \
        --postfix ''
done
