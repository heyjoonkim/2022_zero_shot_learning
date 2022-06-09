export CUDA_VISIBLE_DEVICES=2


# model #
main_model="EleutherAI/gpt-j-6B"

main_path="./few_shot"

seeds="13 21 42 87 100"

n_samples="16"

demo_accuracies="1 0.75 0.5 0.25 0"




# task #
task="sick"
benchmark="huggingface"

## Minimal template ##
# BALANCED # 
for seed in $seeds; do
    for demo_accuracy in $demo_accuracies; do
        python transformers_channel_main.py \
            --task_name $task \
            --benchmark_name $benchmark \
            --model_name_or_path $main_model \
            --demonstration_dir $main_path/$benchmark-$task-seed_$seed-k_$n_samples \
            --output_dir $main_path/channel/$benchmark-$task-seed_$seed-k_$n_samples-correct_$demo_accuracy-minimal \
            --seed $seed \
            --demo_accuracy $demo_accuracy \
            --n_samples $n_samples \
            --overwrite_output_dir \
            --prefix '' \
            --postfix ''
    done
done


demo_accuracies="1"

# task #
task="poem_sentiment"
benchmark="huggingface"

## Minimal template ##
# BALANCED # 
for seed in $seeds; do
    for demo_accuracy in $demo_accuracies; do
        python transformers_channel_main.py \
            --task_name $task \
            --benchmark_name $benchmark \
            --model_name_or_path $main_model \
            --demonstration_dir $main_path/$benchmark-$task-seed_$seed-k_$n_samples \
            --output_dir $main_path/channel/$benchmark-$task-seed_$seed-k_$n_samples-correct_$demo_accuracy-minimal \
            --seed $seed \
            --demo_accuracy $demo_accuracy \
            --n_samples $n_samples \
            --overwrite_output_dir \
            --prefix '' \
            --postfix ''
    done
done
