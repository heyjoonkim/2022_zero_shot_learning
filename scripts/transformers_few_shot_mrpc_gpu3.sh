export CUDA_VISIBLE_DEVICES=3

# task #
task="mrpc"
benchmark="glue"

# model #
main_model="EleutherAI/gpt-j-6B"
# main_model="gpt2"

main_path="./few_shot"

seeds="1 2 3 4 5"

n_samples="16"

# # select in-context samples from train set. (BALANCED)
# for seed in $seeds; do
# python generate_demonstrations.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --seed $seed \
#     --balance_sample \
#     --overwrite_output_dir \
#     --n_samples $n_samples
# done   

# # select in-context samples from train set. (RANDOM)
# for seed in $seeds; do
# python generate_demonstrations.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --seed $seed \
#     --overwrite_output_dir \
#     --n_samples $n_samples
# done   

   
# ## Minimal template ##
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/minimal-2 \
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
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/minimal-2 \
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

## Manual templates ##
# # template 1 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template1-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' True or False?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template1-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' True or False?
# The answer is:'
# done
# # template 1 #


# # template 2 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template2-2 \
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

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template2-2 \
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
# # template 2 #


# # template 5 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template5-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' Yes or No?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template5-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' Yes or No?
# The answer is:'
# done
# # template 5 #

# # template 6 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template6-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? Yes or No?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template6-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? Yes or No?
# The answer is:'
# done
# # template 6 #

# # template 5 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template5-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' Yes or No?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template5-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' Yes or No?
# The answer is:'
# done
# # template 5 #

# # template 6 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template6-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? Yes or No?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template6-3 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? Yes or No?
# The answer is:'
# done
# # template 6 #


# ## Minimal template ##
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/minimal-2 \
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
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/minimal-2 \
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

# # Manual templates ##
# # template 1 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template1-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' True or False?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template1-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' True or False?
# The answer is:'
# done
# # template 1 #


# # template 2 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template2-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? True or False?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template2-2 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? True or False?
# The answer is:'
# done
# # template 2 #


# # template 5 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template5-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' Yes or No?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template5-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' Yes or No?
# The answer is:'
# done
# # template 5 #

# # template 6 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template6-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? Yes or No?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/template6-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? Yes or No?
# The answer is:'
# done
# # template 6 #

# # template 5 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template5-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' Yes or No?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template5-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix '' \
#     --infix '
# The question is: ' \
#     --postfix ' Yes or No?
# The answer is:'
# done
# # template 5 #

# # template 6 #
# # BALANCED # 
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/calibrate/template6-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? Yes or No?
# The answer is:'
# done

# # RANDOM #
# for seed in $seeds; do
# python transformers_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --model_name_or_path $main_model \
#     --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
#     --output_dir $main_path/$task/seed_$seed/k_$n_samples/calibrate/template6-4 \
#     --seed $seed \
#     --n_samples $n_samples \
#     --overwrite_output_dir \
#     --calibrate \
#     --prefix 'Premise : ' \
#     --infix '
# Hypothesis: ' \
#     --postfix '
# Does the premise entails the hypothesis? Yes or No?
# The answer is:'
# done
# # template 6 #


# template 5 #
# BALANCED # 
for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/template5-6 \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
The question is: ' \
    --postfix ' Yes or No?
The answer is:'
done

# RANDOM #
for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/seed_$seed/k_$n_samples/template5-6 \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
The question is: ' \
    --postfix ' Yes or No?
The answer is:'
done
# template 5 #

## Minimal template ##
# BALANCED # 
for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/balanced/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/balanced/seed_$seed/k_$n_samples/minimal-6 \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
' \
    --postfix '
'
done

# RANDOM # 
for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --demonstration_dir $main_path/$task/seed_$seed/k_$n_samples \
    --output_dir $main_path/$task/seed_$seed/k_$n_samples/minimal-6 \
    --seed $seed \
    --n_samples $n_samples \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
' \
    --postfix '
'
done
## Minimal template ##