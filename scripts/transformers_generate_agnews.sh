export CUDA_VISIBLE_DEVICES=0

task="ag_news"
benchmark="huggingface"


# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"
main_path="./generated_datasets"

# generation template
generation_template="template1"

n_samples="2"

seeds="13 21 42 87 100"

for seed in $seeds; do
    python transformers_generate.py \
        --task_name $task \
        --benchmark_name $benchmark \
        --model_name_or_path $main_model \
        --output_dir $main_path/$task/$main_model/$generation_template/$n_samples-shot/$seed/ \
        --seed $seed \
        --n_samples $n_samples \
        --overwrite_output_dir \
        --generation_max_length 55 \
        --generation_min_length 5 \
        --temperature 0.5 \
        --no_repeat_ngram_size 2 \
        --label_token '[LABEL]' \
    --prefix 'Generate an article : ' \
    --infix '
Generate an article about "[LABEL]" :' \
    --postfix ''
done

        # --benchmark_name $benchmark \
sh scripts/transformers_generated_few_shot_agnews.sh