export TORCH_DISTRIBUTED_DEBUG=DETAIL

output_dir="/home/heyjoonkim/data/input_dependent_prompt"
# output_dir="/home/heyjoonkim/data/tmp"
train_batch_size="32"
warmup_ratio="0.06"
train_epochs="20"
task="qnli"
prompt_length="5"


lrs="1e-4 1e-5 1e-3 5e-4 5e-5 5e-3"
model="roberta-large"

for lr in $lrs; do
    accelerate launch main.py \
        --model_name_or_path $model \
        --task_name $task \
        --max_seq_length 128 \
        --per_device_batch_size $train_batch_size \
        --weight_decay 0.1 \
        --warmup_ratio $warmup_ratio \
        --num_train_epochs $train_epochs \
        --output_dir $output_dir/prompt_tuning/$task/$model/prompt_length_$prompt_length/$lr \
        --overwrite_output_dir \
        --seed 1234 \
        --early_stop 5 \
        --apply_prompt \
        --prompt_length $prompt_length \
        --lr $lr \
        --pooling_method cls
done