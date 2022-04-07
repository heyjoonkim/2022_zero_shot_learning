
output_dir="./outputs"
dataset_dir="./generated_datasets"
task="rte"
model="gpt2-xl"
time=`date +%Y-%m-%d-%T`

max_length="30"
temperature="0.1"
top_p="1"
frequency_penalty="0.3"


python openai_generate.py \
    --task_name $task \
    --output_dir $output_dir/$task/$time \
    --dataset_dir $dataset_dir/$task \
    --model_name_or_path $model \
    --overwrite_output_dir \
    --seed 1234 \
    --max_length $max_length \
    --temperature $temperature \
    --top_p $top_p \
    --frequency_penalty $frequency_penalty \
    --positive_prompt " In other words," \
    --negative_prompt " Furthermore,"