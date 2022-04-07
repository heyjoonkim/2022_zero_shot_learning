# export CUDA_VISIBLE_DEVICES=0
# main_model="EleutherAI/gpt-j-6B"
# task="sst2"
# seeds="9999"
# accs="1"
# samples="8"
# for n_sample in $samples; do
#     for seed in $seeds; do
#         for acc in $accs; do
#             python openai_main.py \
#             --task_name $task \
#             --model_name_or_path $main_model \
#             --output_dir outputs/$task/gpt-j/best_prompt/$n_sample-shot/acc-$acc/seed-$seed \
#             --seed $seed \
#             --overwrite_output_dir \
#             --n_samples $n_sample \
#             --demo_accuracy $acc \
#             --log_results \
#             --prefix "Review: " \
#             --postfix "
# Sentiment: "
#         done
#     done
# done


export CUDA_VISIBLE_DEVICES=0
main_model="EleutherAI/gpt-j-6B"
task="rte"
seeds="9999"
accs="1"
samples="8"
for n_sample in $samples; do
    for seed in $seeds; do
        for acc in $accs; do
            python openai_main.py \
            --task_name $task \
            --model_name_or_path $main_model \
            --output_dir outputs/$task/gpt-j/best_prompt/$n_sample-shot/acc-$acc/seed-$seed \
            --seed $seed \
            --overwrite_output_dir \
            --n_samples $n_sample \
            --demo_accuracy $acc \
            --log_results \
            --infix "
" \
            --postfix "
"
        done
    done
done
