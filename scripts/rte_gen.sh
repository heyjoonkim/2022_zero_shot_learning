main_model="davinci"
task="rte"
seed=42

python generate.py \
            --task_name $task \
            --model_name_or_path $main_model \
            --output_dir outputs/$task/generate \
            --dataset_dir data/$task \
            --seed $seed \
            --overwrite_output_dir \
            --temperature 0.3 \
            --positive_prompt " In other words," \
            --negative_prompt " Furthermore,"
