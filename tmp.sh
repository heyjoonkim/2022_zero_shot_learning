
export TORCH_DISTRIBUTED_DEBUG=DETAIL

output_dir="/home/heyjoonkim/data/tmp"
train_batch_size="32"
warmup_ratio="0.06"
train_epochs="20"
prompt_length="5"


model="roberta-large"

learning_rates="1e-4"
plm_layers="-1"
tasks="qqp"

for task in $tasks; do
    for plm_layer in $plm_layers; do
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
                --output_dir $output_dir/$task/$model/layer_$plm_layer/$learning_rate \
                --overwrite_output_dir \
                --seed 1234 \
                --early_stop 5 \
                --pooling_method mask \
                --plm_layer $plm_layer \
                --max_seq_length 1024 \
                --apply_prompt
        done
    done
done

