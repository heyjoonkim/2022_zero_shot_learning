###### GLOBAL PARAM ######
seeds="13"
base_path="data/"
n_samples="16"

###### GLUE ######
benchmark='glue'

task='sst2'
for seed in $seeds; do
    python generate_fewshot.py \
            --benchmark_name $benchmark \
            --task_name $task \
            --output_dir ${base_path}${task} \
            --n_samples $n_samples \
            --seed $seed
done