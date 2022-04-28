
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
    --output_dir $main_path/$task/$main_model/template3/$seed/ \
    --model_name_or_path $main_model \
    --overwrite_output_dir \
    --seed $seed \
    --max_length $max_length \
    --temperature $temperature \
    --top_p $top_p \
    --frequency_penalty $frequency_penalty \
    --label_token '[LABEL]' \
    --input_label_token '[INPUT_LABEL]' \
    --prefix 'Generate a question about
"[INPUT_LABEL]" : ' \
    --infix '
Generate a question about
"[LABEL]" :' \
    --postfix ''
done