export CUDA_VISIBLE_DEVICES=3

# task="sst2"
# task="rte"
# benchmark="glue"

# task="cb"
# benchmark="super_glue"

# task="sst5"
# task="agnews"
# task="yahoo"
# task="yelp"
task='cb'
benchmark="super_glue"

dataset_path="./generated_datasets"
main_path="./retrieved_datasets"
# 
main_model="EleutherAI/gpt-j-6B"
# main_model="text-davinci-002"
# main_model="all-MiniLM-L12-v1"

generation_template="template1"
seeds="13"
n_sample="8"

for seed in $seeds; do

python generated_analysis.py \
    --task_name $task \
    --dataset_dir $dataset_path/$task/$main_model/$generation_template/$n_sample-shot/$seed/ \
    --seed $seed \
    --prefix '' \
    --infix '' \
    --postfix ''

done


    # --dataset_dir $dataset_path/$task/$main_model/$template/$seed/ \
