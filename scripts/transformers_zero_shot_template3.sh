export CUDA_VISIBLE_DEVICES=1


## MODELS ##
# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"
main_path="./test_results/paper_results"

###############
## ZERO-SHOT ##
###############
seed="1"

# task="mrpc"
# benchmark="glue"

# # Minimal template #
# python transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --output_dir $main_path/$task/$main_model/zero_shot/template1/ \
#     --seed $seed \
#     --n_samples 0 \
#     --overwrite_output_dir \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis : ' \
#     --postfix '
# True or False? '
# # Minimal template #

# task="rte"
# benchmark="glue"

# # Minimal template #
# python transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --output_dir $main_path/$task/$main_model/zero_shot/template1/ \
#     --seed $seed \
#     --n_samples 0 \
#     --overwrite_output_dir \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis : ' \
#     --postfix '
# True or False? '
# # Minimal template #


# task="wnli"
# benchmark="glue"

# # Minimal template #
# python transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --output_dir $main_path/$task/$main_model/zero_shot/template1/ \
#     --seed $seed \
#     --n_samples 0 \
#     --overwrite_output_dir \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis : ' \
#     --postfix '
# Yes or No? '
# # Minimal template #



task="cb"
benchmark="super_glue"

# Minimal template #
python transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --output_dir $main_path/$task/$main_model/zero_shot/template6/ \
    --seed $seed \
    --n_samples 0 \
    --overwrite_output_dir \
    --prefix 'Premise : ' \
    --infix '
Hypothesis : ' \
    --postfix '
Yes, No, or Neither?'
# Minimal template #

