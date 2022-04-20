export CUDA_VISIBLE_DEVICES=3

task="cb"
benchmark="super_glue"

# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"
main_path="./generated_datasets"


seeds="1 2 3 4 5" # 6 7 8 9 10"
# seeds="7"

for seed in $seeds; do
    deepspeed transformers_generate.py \
        --task_name $task \
        --model_name_or_path $main_model \
        --benchmark_name $benchmark \
        --ds_config ds_configs/fp16.json \
        --output_dir $main_path/$task/$main_model/template1/$seed/ \
        --seed $seed \
        --n_samples 16 \
        --overwrite_output_dir \
        --generation_max_length 15 \
        --generation_min_length 5 \
        --temperature 0.5 \
        --no_repeat_ngram_size 2 \
        --label_token '[LABEL]' \
        --input_label_token '[INPUT_LABEL]' \
    --prefix 'Premise: '\
    --infix '
Generate a "[INPUT_LABEL]" hypothesis : ' \
    --postfix '
Generate a "[LABEL]" hypothesis :'
done

        # --benchmark_name $benchmark \
# sh scripts/transformers_generated_few_shot_sst5.sh