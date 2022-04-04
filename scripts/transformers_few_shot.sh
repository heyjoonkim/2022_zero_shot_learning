# export CUDA_VISIBLE_DEVICES=0

# task="sst2"
# task="rte"
# benchmark="glue"

# task="cb"
# benchmark="super_glue"

# task="sst5"
task="agnews"
# task="yahoo"
# task="yelp"


# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
main_model="EleutherAI/gpt-neo-2.7B"
# main_model="EleutherAI/gpt-j-6B"
main_path="./tmp"

seed="1234"

# seeds="1 2 3 4 5 6 7 8 9 10"
seeds="9 10"


for seed in $seeds; do
deepspeed transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model \
    --seed $seed \
    --overwrite_output_dir \
    --n_samples 4 \
    --balance_sample \
    --prefix '' \
    --infix '
'\
    --postfix '
'
done
   


    # RTE
    #     --prefix '' \
    #     --infix '
    # Question : ' \
    #     --postfix " True or False?
    # Answer :"
    # --prefix '' \
    #     --infix '
    # ' \
    #     --postfix '
    # '

    # SST-2
    # --prefix 'Review: ' \
    # --infix '\nSentiment: ' \
    # --postfix ""

    # CB
    # --prefix ''
    # --infix '\nQuestion: ' \
    # --postfix " True, False or Neither?\nAnswer:"