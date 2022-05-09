export CUDA_VISIBLE_DEVICES=3

task='trec'
# benchmark="huggingface"

# main_model="davinci"
main_model="text-davinci-002"

## RETRIEVAL MODELS ##
retrieval_model="all-MiniLM-L12-v1"

## directory ##
main_path="./test_results/OURS"
dataset_path="./retrieved_datasets"

## template number ##
template="retrieval"

seeds="1"
n_sample="6"

# gold labeled #
for seed in $seeds; do
python openai_generated_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --output_dir $main_path/$task/$main_model/$template/minimal/ \
    --dataset_dir $dataset_path/$task/$retrieval_model/$seed/$n_sample/ \
    --overwrite_output_dir \
    --seed $seed \
    --n_samples $n_sample \
    --prefix '' \
    --infix '
' \
    --postfix ''
done

# random labeled #
for seed in $seeds; do
python openai_generated_main.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --output_dir $main_path/$task/$main_model/$template/random_labeling/minimal \
    --dataset_dir $dataset_path/$task/$retrieval_model/random_labeling/$seed/$n_sample/ \
    --overwrite_output_dir \
    --seed $seed \
    --n_samples $n_sample \
    --prefix '' \
    --infix '
' \
    --postfix ''
done