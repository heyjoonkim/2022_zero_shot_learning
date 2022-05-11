export CUDA_VISIBLE_DEVICES=1
main_model="davinci"
main_model="text-curie-001"
# main_model="gpt2"
task="rte"
seeds="100"
accs="1"
samples="16"
# python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/text-curie/best_prompt_space/0-shot/acc-1/seed-100 \
#             --seed 100 \
#             --overwrite_output_dir \
#             --log_results \
#             --infix "
# question: " \
#              --postfix " True or False?
# answer:"

# for n_sample in $samples; do
#     for seed in $seeds; do
#         for acc in $accs; do
#             python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/text-curie/best_prompt_space/$n_sample-shot/acc-$acc/seed-$seed \
#             --seed $seed \
#             --overwrite_output_dir \
#             --n_samples $n_sample \
#             --demo_accuracy $acc \
#             --log_results \
#             --infix "
# question: " \
#              --postfix " True or False?
# answer:"
#         done
#     done
# done
for n_sample in $samples; do
    for seed in $seeds; do
        for acc in $accs; do
            python openai_main.py \
            --task_name $task \
            --model_name_or_path $main_model \
            --output_dir outputs/$task/text-curie/no_prompt_nospace/$n_sample-shot/acc-$acc/seed-$seed \
            --seed $seed \
            --overwrite_output_dir \
            --n_samples $n_sample \
            --demo_accuracy $acc \
            --log_results \
            --infix "
" \
             --postfix "
"
        done
    done
done

# export CUDA_VISIBLE_DEVICES=0
# main_model="EleutherAI/gpt-j-6B"
# # main_model="gpt2"
# task="rte"
# seeds="13 21 42 87 100"
# accs="0 0.25 0.5 0.75 1"
# samples="16"
# for n_sample in $samples; do
#     for seed in $seeds; do
#         for acc in $accs; do
#             python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/gpt-j/best_prompt_space/$n_sample-shot/acc-$acc/seed-$seed \
#             --seed $seed \
#             --overwrite_output_dir \
#             --n_samples $n_sample \
#             --demo_accuracy $acc \
#             --log_results \
#             --infix "
# The question is: " \
#              --postfix " True or False?
# The answer is:"
#         done
#     done
# done
# python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/gpt-j/best_prompt_space/0-shot/acc-1/seed-100 \
#             --seed 100 \
#             --overwrite_output_dir \
#             --log_results \
#             --infix "
# The question is: " \
#              --postfix " True or False?
# The answer is:"

# export CUDA_VISIBLE_DEVICES=0
# main_model="EleutherAI/gpt-j-6B"
# task="mrpc"
# seeds="13 21 42 87 100"
# accs="0 0.25 0.5 0.75 1"
# samples="16"
# for n_sample in $samples; do
#     for seed in $seeds; do
#         for acc in $accs; do
#             python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/gpt-j/best_prompt_space/$n_sample-shot/acc-$acc/seed-$seed \
#             --seed $seed \
#             --overwrite_output_dir \
#             --n_samples $n_sample \
#             --demo_accuracy $acc \
#             --log_results \
#             --infix "
# The question is: " \
#             --postfix " True or False?
# The answer is:"
#         done
#     done
# done
# python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/gpt-j/best_prompt_space/0-shot/acc-1/seed-100 \
#             --seed 100 \
#             --overwrite_output_dir \
#             --log_results \
#             --infix "
# The question is: " \
#             --postfix " True or False?
# The answer is:"

# export CUDA_VISIBLE_DEVICES=0
# main_model="EleutherAI/gpt-j-6B"
# task="sick"
# seeds="13 21 42 87 100"
# accs="0 0.25 0.5 0.75 1"
# samples="16"
# for n_sample in $samples; do
#     for seed in $seeds; do
#         for acc in $accs; do
#             python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/gpt-j/best_prompt_space/$n_sample-shot/acc-$acc/seed-$seed \
#             --seed $seed \
#             --overwrite_output_dir \
#             --n_samples $n_sample \
#             --demo_accuracy $acc \
#             --log_results \
#             --infix "
# The question is: " \
#             --postfix " True or False?
# The answer is:"
#         done
#     done
# done
# python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/gpt-j/best_prompt_space/0-shot/acc-1/seed-100 \
#             --seed 100 \
#             --overwrite_output_dir \
#             --log_results \
#             --infix "
# The question is: " \
#             --postfix " True or False?
# The answer is:"

# python openai_main.py \
#     --task_name $task \
#     --model_name_or_path $main_model \
#     --output_dir outputs/$task/gpt-j/test/8-shot/acc-1/seed-42 \
#     --seed 42 \
#     --overwrite_output_dir \
#     --n_samples 8 \
#     --log_results \
#     --infix "
# The question is: " \
#     --postfix " True or False?
# The answer is:"
