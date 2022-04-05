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

seeds="2 3 4 5"
n_samples="8 16"

for seed in $seeds; do
    for n_sample in $n_samples; do

deepspeed transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model \
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

seeds="1 2 3 4 5"
n_samples="1 2 4 8 16"

for seed in $seeds; do
    for n_sample in $n_samples; do

deepspeed transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model/random/ \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
    --prefix 'Question: ' \
    --infix '
Type:' \
    --postfix ''

    done
done

seeds="1 2 3 4 5"
n_samples="1 2 4 8 16"

for seed in $seeds; do
    for n_sample in $n_samples; do

deepspeed transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --ds_config ds_configs/fp16.json \
    --output_dir $main_path/$task/$main_model/minimal/random/ \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
        --prefix '' \
    --infix '
' \
    --postfix ''

    done
done

seeds="1 2 3 4 5"
n_samples="1 2 4 8 16"

for seed in $seeds; do
    for n_sample in $n_samples; do

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