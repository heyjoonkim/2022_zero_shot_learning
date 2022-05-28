export CUDA_VISIBLE_DEVICES=0

task="rte"
benchmark="glue"

## MODELS ##
# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"
main_path="./test_results/paper_results"


##############
## FEW-SHOT ##
##############

seeds="13 21 42 87 100"
n_samples="8"

for n_sample in $n_samples; do
    for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --output_dir $main_path/$task/$main_model/$n_samples-shot/template1/ \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
    --prefix 'Premise : ' \
    --infix '
Hypothesis : ' \
    --postfix '
True or False? '
    done
done


for n_sample in $n_samples; do
    for seed in $seeds; do
python transformers_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --benchmark_name $benchmark \
    --output_dir $main_path/$task/$main_model/$n_samples-shot/minimal/ \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
' \
    --postfix ''
    done
done