export CUDA_VISIBLE_DEVICES=0
main_model="EleutherAI/gpt-j-6B"
task="hate"
seeds="13 21 42 87 100"
accs="0 0.25 0.5 0.75 1"
samples="16"

python test.py \
--task_name $task \
--model_name_or_path $main_model \
--output_dir outputs/$task/gpt-j/debug/16-shot/acc-1/seed-100 \
--seed 100 \
--overwrite_output_dir \
--n_samples 16 \
--log_results \

# python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/gpt-j/no_prompt_nospace/0-shot/acc-1/seed-100 \
#             --seed 100 \
#             --overwrite_output_dir \
#             --log_results \
#             --postfix "
# "