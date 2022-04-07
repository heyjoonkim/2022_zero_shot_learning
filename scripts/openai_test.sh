
output_dir="./outputs"
task="sst2"
# model="EleutherAI/gpt-neo-2.7B"
model="EleutherAI/gpt-j-6B"
# model="davinci"
time=`date +%Y-%m-%d-%T`


python openai_main.py \
    --task_name $task \
    --output_dir $output_dir/$task/$time \
    --model_name_or_path $model \
    --overwrite_output_dir \
    --seed 1234 \
    --prefix "Review: " \
    --postfix "
Sentiment: "



    # --n_samples 1 \
    # --prefix "Given that " \