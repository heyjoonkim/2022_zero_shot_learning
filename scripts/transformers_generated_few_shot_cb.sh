export CUDA_VISIBLE_DEVICES=2,3

## TASKS ##
task="cb"
benchmark="super_glue"

## MODELS ##
# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"

## directory ##
main_path="./test_results/OURS"
dataset_path="./generated_datasets"

##############
## FEW-SHOT ##
##############

seeds="1 2 3 4 5"
# n_samples="1 2 4"
n_samples="4 8 16"

# Manual template #
for n_sample in $n_samples; do
    for seed in $seeds; do
deepspeed transformers_generated_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model/template1/manual/balanced/ \
    --dataset_dir $dataset_path/$task/$main_model/template1/$seed/ \
    --seed $seed \
    --n_samples $n_sample \
    --balance_sample \
    --overwrite_output_dir \
    --prefix 'premise: ' \
    --infix '
hypothesis: ' \
    --postfix '
prediction:'
    done
done
# Manual template #

# Manual template #
for n_sample in $n_samples; do
    for seed in $seeds; do
deepspeed transformers_generated_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model/template1/manual/random/ \
    --dataset_dir $dataset_path/$task/$main_model/template1/$seed/ \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
    --prefix 'premise: ' \
    --infix '
hypothesis: ' \
    --postfix '
prediction:'
    done
done
# Manual template #






# Minimal template #
for n_sample in $n_samples; do
    for seed in $seeds; do
deepspeed transformers_generated_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model/template1/minimal/balanced/ \
    --dataset_dir $dataset_path/$task/$main_model/template1/$seed/ \
    --seed $seed \
    --n_samples $n_sample \
    --balance_sample \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
' \
    --postfix ''
    done
done
# Minimal template #

# Minimal template #
for n_sample in $n_samples; do
    for seed in $seeds; do
deepspeed transformers_generated_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model/template1/minimal/random/ \
    --dataset_dir $dataset_path/$task/$main_model/template1/$seed/ \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
' \
    --postfix ''
    done
done
# Minimal template #
