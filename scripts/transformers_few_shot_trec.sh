export CUDA_VISIBLE_DEVICES=3

## TASKS ##
# task="sst2"
# task="rte"
# benchmark="glue"

# task="cb"
# benchmark="super_glue"

# task="sst5"
# task="agnews"
# task="yahoo"
# task="yelp"
task='trec'
benchmark="huggingface"

## MODELS ##
# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"
main_path="./test_results/few_shot"

###############
## ZERO-SHOT ##
###############
# seed="1"

# # Manual template #
# deepspeed transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --ds_config ds_configs/fp16.json \
#     --output_dir $main_path/$task/$main_model/manual/ \
#     --seed $seed \
#     --n_samples 0 \
#     --overwrite_output_dir \
#     --prefix 'Question: ' \
#     --infix '
# Type:' \
#     --postfix ''
# # Manual template #

# # Minimal template #
# deepspeed transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --ds_config ds_configs/fp16.json \
#     --output_dir $main_path/$task/$main_model/minimal/ \
#     --seed $seed \
#     --n_samples 0 \
#     --overwrite_output_dir \
#         --prefix '' \
#     --infix '
# ' \
#     --postfix ''
# # Minimal template #

##############
## FEW-SHOT ##
##############

seeds="1 2 3 4 5 6 7 8 9 10"
# n_samples="1 2 4 8 16"
# n_samples="1 4 8"
n_samples="6"

for n_sample in $n_samples; do
    for seed in $seeds; do
deepspeed transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model/manual/ \
    --seed $seed \
    --n_samples $n_sample \
    --balance_sample \
    --overwrite_output_dir \
    --prefix 'Question: ' \
    --infix '
Type:' \
    --postfix ''
    done
done

# for n_sample in $n_samples; do
#     for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --ds_config ds_configs/fp16.json \
#     --output_dir $main_path/$task/$main_model/manual/random/ \
#     --seed $seed \
#     --n_samples $n_sample \
#     --overwrite_output_dir \
#     --prefix 'Question: ' \
#     --infix '
# Type:' \
#     --postfix ''
#     done
# done

for n_sample in $n_samples; do
    for seed in $seeds; do
deepspeed transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model/minimal/ \
    --balance_sample \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
        --prefix '' \
    --infix '
' \
    --postfix ''
    done
done


# for n_sample in $n_samples; do
#     for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --ds_config ds_configs/fp16.json \
#     --output_dir $main_path/$task/$main_model/minimal/random/ \
#     --seed $seed \
#     --n_samples $n_sample \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix ''
#     done
# done


    # --benchmark_name $benchmark \


# sh scripts/transformers_few_shot_sst5.sh