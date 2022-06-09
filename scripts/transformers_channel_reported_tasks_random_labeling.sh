export CUDA_VISIBLE_DEVICES=3


# model #
main_model="EleutherAI/gpt-j-6B"

main_path="./few_shot"

seeds="13 21 42 87 100"

n_samples="16"



# task 9 #
task="stance_atheism"
benchmark="tweet_eval"

## Minimal template ##
# BALANCED # 
for seed in $seeds; do
    python transformers_channel_main.py \
        --task_name $task \
        --benchmark_name $benchmark \
        --model_name_or_path $main_model \
        --demonstration_dir $main_path/$benchmark-$task-seed_$seed-k_$n_samples \
        --output_dir $main_path/channel/$benchmark-$task-seed_$seed-k_$n_samples-random_labeling-minimal \
        --seed $seed \
        --n_samples $n_samples \
        --overwrite_output_dir \
        --prefix '' \
        --postfix ''
done



# task 10 #
task="stance_feminist"
benchmark="tweet_eval"

## Minimal template ##
# BALANCED # 
for seed in $seeds; do
    python transformers_channel_main.py \
        --task_name $task \
        --benchmark_name $benchmark \
        --model_name_or_path $main_model \
        --demonstration_dir $main_path/$benchmark-$task-seed_$seed-k_$n_samples \
        --output_dir $main_path/channel/$benchmark-$task-seed_$seed-k_$n_samples-random_labeling-minimal \
        --seed $seed \
        --n_samples $n_samples \
        --overwrite_output_dir \
        --prefix '' \
        --postfix ''
done

