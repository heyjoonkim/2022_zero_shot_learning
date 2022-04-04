
output_dir="./outputs"
task="rte"
model="davinci"
time=`date +%Y-%m-%d-%T`


python openai_main.py \
    --task_name $task \
    --output_dir $output_dir/$task/$time \
    --model_name_or_path $model \
    --overwrite_output_dir \
    --seed 1234 \
    --n_samples 8 \
    --infix '\nQuestion : ' \
    --postfix " True or False? \n Answer : "



    # --n_samples 1 \
    # --prefix "Given that " \