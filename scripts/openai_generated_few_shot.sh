export CUDA_VISIBLE_DEVICES=3

task='trec'
# benchmark="huggingface"

main_model="davinci"

seeds="1" # "1 2 3 4 5"
n_sample="6"

## directory ##
main_path="./test_results/OURS"
dataset_path="./generated_datasets"

for seed in $seeds; do
python openai_generated_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --output_dir $main_path/$task/$main_model/template3/minimal/balanced/ \
    --dataset_dir $dataset_path/$task/$main_model/template3/$seed/ \
    --overwrite_output_dir \
    --seed $seed \
    --n_samples $n_sample \
    --balance_sample \
    --prefix '' \
    --infix '
' \
    --postfix ''
#     --prefix 'Question: ' \
#     --infix '
# Type:' \
#     --postfix ''
done