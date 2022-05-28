export CUDA_VISIBLE_DEVICES=0


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

# task="sst2"
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
#         --prefix 'Review : ' \
#     --infix '
# Sentiment :' \
#     --postfix ''
# # Minimal template #

# task="SetFit/sst5"
# benchmark="huggingface"

# # Minimal template #
# python transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --output_dir $main_path/$task/$main_model/zero_shot/template1/ \
#     --seed $seed \
#     --n_samples 0 \
#     --overwrite_output_dir \
#         --prefix 'Review : ' \
#     --infix '
# Sentiment :' \
#     --postfix ''
# # Minimal template #


task="trec"
benchmark="huggingface"

# Minimal template #
python transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --output_dir $main_path/$task/$main_model/zero_shot/template3/ \
    --seed $seed \
    --n_samples 0 \
    --overwrite_output_dir \
    --prefix 'Question : ' \
    --infix '
Label :' \
    --postfix ''
# Minimal template #

# # Minimal template #
# python transformers_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --benchmark_name $benchmark \
#     --output_dir $main_path/$task/$main_model/zero_shot/template2/ \
#     --seed $seed \
#     --n_samples 0 \
#     --overwrite_output_dir \
#     --prefix 'Question : ' \
#     --infix '
# Answer Type :' \
#     --postfix ''
# # Minimal template #


