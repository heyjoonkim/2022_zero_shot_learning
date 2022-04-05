export CUDA_VISIBLE_DEVICES=3

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

# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"
main_path="./test_results"

seed="1"

deepspeed transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model \
    --seed $seed \
    --overwrite_output_dir \
        --prefix '' \
    --infix '
' \
    --postfix ''
#     --prefix 'Question: ' \
#     --infix '
# Type:' \
#     --postfix ''
#     --prefix '' \
#     --infix '
# ' \
#     --postfix ''


# AG News
#     --prefix 'Headline :' \
#     --infix '
# News : ' \
#     --postfix '
# Category :'

# RTE
#     --prefix '' \
#     --infix '
# Question : ' \
#     --postfix " True or False?
# Answer :"

    # SST-2
    # --prefix 'Review: ' \
    # --infix '\nSentiment: ' \
    # --postfix ""

    # CB
    # --prefix ''
    # --infix '\nQuestion: ' \
    # --postfix " True, False or Neither?\nAnswer:"