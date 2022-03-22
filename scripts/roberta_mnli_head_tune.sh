
export TORCH_DISTRIBUTED_DEBUG=DETAIL

output_dir="/home/heyjoonkim/data/input_dependent_prompt/tmp"
train_batch_size="16"
warmup_ratio="0.06"
train_epochs="20"
task="mnli"


learning_rates="1e-3 1e-4 1e-5"


model="roberta-base"

for learning_rate in $learning_rates; do
    accelerate launch main.py \
        --model_name_or_path $model \
        --task_name $task \
        --max_seq_length 128 \
        --per_device_batch_size $train_batch_size \
        --lr $learning_rate \
        --weight_decay 0.1 \
        --warmup_ratio $warmup_ratio \
        --num_train_epochs $train_epochs \
        --output_dir $output_dir/head-tune/$task/$model/$learning_rate \
        --overwrite_output_dir \
        --seed 1234 \
        --early_stop 5
done


model="roberta-large"

for learning_rate in $learning_rates; do
    accelerate launch main.py \
        --model_name_or_path $model \
        --task_name $task \
        --max_seq_length 128 \
        --per_device_batch_size $train_batch_size \
        --lr $learning_rate \
        --weight_decay 0.1 \
        --warmup_ratio $warmup_ratio \
        --num_train_epochs $train_epochs \
        --output_dir $output_dir/head-tune/$task/$model/$learning_rate \
        --overwrite_output_dir \
        --seed 1234 \
        --early_stop 5
done