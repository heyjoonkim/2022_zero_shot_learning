
output_dir="./outputs"
main_path="./generated_datasets"

task='trec'
benchmark="huggingface"

# main_model="davinci"
main_model="text-davinci-002"

# generation parameters
max_length="15"
temperature="0.5"
top_p="1"
frequency_penalty="0"


seeds="1" # 2 3 4 5"

for seed in $seeds; do
python openai_generate.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --output_dir $main_path/$task/$main_model/template6/$seed/ \
    --model_name_or_path $main_model \
    --overwrite_output_dir \
    --seed $seed \
    --max_length $max_length \
    --temperature $temperature \
    --top_p $top_p \
    --frequency_penalty $frequency_penalty \
    --label_token '[LABEL]' \
    --input_label_token '[INPUT_LABEL]' \
    --prefix 'Sample question : ' \
    --infix '
Generate a "[LABEL]" question :' \
    --postfix ''
done

sh scripts/openai_few_shot_trec.sh