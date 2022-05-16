export CUDA_VISIBLE_DEVICES=2

# task #
task="mrpc"
benchmark="glue"

# model #
main_model="EleutherAI/gpt-j-6B"
# main_model="gpt2"

main_path="./few_shot"

seeds="1 2 3 4 5"

n_samples="16"

# # template 3 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template3-2 \
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

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template3-2 \
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
# # template 3 #

# ## Minimal template ##
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/minimal-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix '
# '
# done

# # RANDOM # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/minimal-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix '
# '
# done
# ## Minimal template ##


# # template 4 #
# # BALANCED # 
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template4-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done

# # RANDOM #
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template4-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done
# # template 4 #
# # Manual template #


## Minimal template ##
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/minimal-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix '
# '
# done

# # RANDOM # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/minimal-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix '
# '
# done
# ## Minimal template ##


# # template 4 #
# # BALANCED # 
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template4-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done

# # RANDOM #
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template4-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done
# # template 4 #
# # Manual template #


# # template 3 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template3-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix 'First Sentence: ' \
#     --infix '
# Second Sentence: ' \
#     --postfix '
# Does the First Sentence have the same meaning as the Second Sentence? True or False?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template3-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix 'First Sentence: ' \
#     --infix '
# Second Sentence: ' \
#     --postfix '
# Does the First Sentence have the same meaning as the Second Sentence? True or False?
# The answer is:'
# done
# # template 3 #

# # template 4 #
# # BALANCED # 
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template4-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done

# # RANDOM #
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template4-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done
# # template 4 #
# # Manual template #



## Minimal template ##
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/minimal-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix '
# '
# done

# # RANDOM # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/minimal-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix '
# '
# done
# ## Minimal template ##


# # template 4 #
# # BALANCED # 
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template4-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done

# # RANDOM #
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template4-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done
# # template 4 #
# # Manual template #


# # Minimal template ##
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/minimal-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix '
# '
# done

# # RANDOM # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/minimal-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# ' \
#     --postfix '
# '
# done
# ## Minimal template ##


# # template 4 #
# # BALANCED # 
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template4-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done

# # RANDOM #
# for seed in $seeds; do
# deepspeed transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template4-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# question: ' \
#     --postfix '
# answer:'
# done
# # template 4 #
# # Manual template #

# template 1 #
# BALANCED # 
for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template1-6 \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
The question is: ' \
    --postfix ' True or False?
The answer is:'
done

# RANDOM #
for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/seed_$seed/k_$n_samples/template1-6 \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
The question is: ' \
    --postfix ' True or False?
The answer is:'
done
# template 1 #

# template 4 #
# BALANCED # 
for seed in $seeds; do
deepspeed transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template4-6 \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
question: ' \
    --postfix '
answer:'
done

# RANDOM #
for seed in $seeds; do
deepspeed transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/seed_$seed/k_$n_samples/template4-6 \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
question: ' \
    --postfix '
answer:'
done
# template 4 #