export CUDA_VISIBLE_DEVICES=3

main_path="./test_results/few_shot"

task='trec'
benchmark="huggingface"

main_model="davinci"

seeds="1 2 3" # "1 2 3 4 5"
n_sample="6"
# n_sample="0"


# for seed in $seeds; do
# python openai_main.py \
#     --task_name $task \
#     --benchmark_name $benchmark \
#     --output_dir $main_path/$task/$main_model/manual/ \
#     --model_name_or_path $main_model \
#     --overwrite_output_dir \
#     --seed $seed \
#     --n_samples $n_sample \
#     --balance_sample \
#     --prefix 'Question: ' \
#     --infix '
# Type:' \
#     --postfix ''
# done

for seed in $seeds; do
python openai_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --output_dir $main_path/$task/$main_model/minimal/ \
    --model_name_or_path $main_model \
    --overwrite_output_dir \
    --seed $seed \
    --n_samples $n_sample \
    --balance_sample \
    --prefix '' \
    --infix '
' \
    --postfix ''
done

sh scripts/openai_generated_few_shot.sh