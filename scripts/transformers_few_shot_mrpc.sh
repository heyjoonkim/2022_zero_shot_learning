export CUDA_VISIBLE_DEVICES=0,1,2

# task #
task="mrpc"
benchmark="glue"

# model #
main_model="EleutherAI/gpt-j-6B"
# main_model="gpt2"

main_path="./few_shot"

seeds="1" # 2 3 4 5"

n_samples="16"

for seed in $seeds; do
# train mlm
python generate_demonstrations.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
    --seed $seed \
    --balance_sample \
    --overwrite_output_dir \
    --n_samples $n_samples
done   

   
# # Minimal template #
for seed in $seeds; do
deepspeed transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/minimal \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
' \
    --postfix '
'

    # --ds_config ds_configs/zero3_config.json \
done
# # minimal template #

# # Manual template #

# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --ds_config ds_configs/fp16.json \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template1 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' True or False?
# The answer is:'
# done

# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --ds_config ds_configs/fp16.json \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' True or False?
# The answer is:'
# done

# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --ds_config ds_configs/fp16.json \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' True or False?
# The answer is:'
# done

# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --ds_config ds_configs/fp16.json \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? True or False?
# The answer is:'
# done

# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --ds_config ds_configs/fp16.json \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix 'First Sentence: ' \
#     --infix '
# Second Sentence: ' \
#     --postfix '
# Does the First Sentence have the same meaning as the Second Sentence? True or False?
# The answer is:'
# done

    # --explicit_label_space \
# Manual template #