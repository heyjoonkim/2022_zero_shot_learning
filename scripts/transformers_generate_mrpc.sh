export CUDA_VISIBLE_DEVICES=1

task="mrpc"
benchmark="glue"


# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"
main_path="./generated_datasets"

# generation template
generation_template="template1"

n_samples="8"

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
        --generation_max_length 25 \
        --generation_min_length 5 \
        --temperature 0.5 \
        --no_repeat_ngram_size 2 \
        --label_token '[LABEL]' \
    --prefix 'Premise : ' \
    --infix '
Generate a Hypothesis : ' \
    --postfix '
Generate a "[LABEL]" Hypothesis :'
done
