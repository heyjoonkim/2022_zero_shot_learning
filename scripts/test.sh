export CUDA_VISIBLE_DEVICES=0
main_model="EleutherAI/gpt-j-6B"
task="poem_sentiment"
seeds="13 21 42 87 100"
accs="0 0.25 0.5 0.75 1"
samples="16"
for n_sample in $samples; do
    for seed in $seeds; do
        for acc in $accs; do
            python openai_main.py \
            --task_name $task \
            --model_name_or_path $main_model \
            --output_dir outputs/$task/gpt-j/balanced/$n_sample-shot/acc-$acc/seed-$seed \
            --seed $seed \
            --overwrite_output_dir \
            --n_samples $n_sample \
            --demo_accuracy $acc \
            --log_results \
            --prefix "Review: " \
            --postfix "
Sentiment:"
        done
    done
done
