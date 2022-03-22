
export TORCH_DISTRIBUTED_DEBUG=DETAIL

output_dir="/home/heyjoonkim/data/input_dependent_prompt"
train_batch_size="32"
warmup_ratio="0.06"
train_epochs="20"
task="sst2"
prompt_length="20"


learning_rates="5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6"
plm_layer="21"

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
        --output_dir $output_dir/v5.1.1/$task/$model/layer_$plm_layer/prompt_length_$prompt_length/$learning_rate \
        --overwrite_output_dir \
        --seed 1234 \
        --early_stop 5 \
        --apply_prompt \
        --prompt_length $prompt_length \
        --plm_layer $plm_layer
done