main_model="davinci"
task="sst2"
seed=42

python main.py \
            --task_name $task \
            --model_name_or_path $main_model \
            --output_dir outputs/$task/300_w_prompt_gen_4shot \
            --seed $seed \
            --overwrite_output_dir \
            --postfix "\nThe sentiment is: "
