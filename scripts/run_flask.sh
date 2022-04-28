export CUDA_VISIBLE_DEVICES=0
# export MODEL_NAME='gpt2'
export MODEL_NAME='EleutherAI/gpt-j-6B'

flask run --host 0.0.0.0 --port 9999