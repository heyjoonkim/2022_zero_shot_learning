""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import os
import json
import time



from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm

import torch

from dataset_utils import generated_task_to_path, task_to_keys, task_to_verbalizer, prepare_generated_incontext_sampling
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(generated_task_to_path.keys()),
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="The name of the benchmark to train on.",
        choices=['glue', 'super_glue', 'huggingface'],
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    # until here #

    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)

    # mkdir output directory to save logs and configs.
    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        

    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    args.verbalizer = task_to_verbalizer.get(args.task_name)
    args.label2token = {v:k for k,v in args.verbalizer.items()}

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = DatasetDict()
    
    raw_datasets = DatasetDict()
    if args.task_name is not None and args.benchmark_name is not None:
        # SST-5, TREC, AGNews
        if args.benchmark_name == 'huggingface':
            raw_train_dataset = load_dataset(args.task_name, split='train')
            raw_eval_dataset = load_dataset(args.task_name, split='test')
        else:
            # glue, super_glue benchmarks
            raw_train_dataset = load_dataset(args.benchmark_name, args.task_name, split=f'train')
            raw_eval_dataset = load_dataset(args.benchmark_name, args.task_name, split=f'validation')
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')

    raw_datasets['train'] = raw_train_dataset
    raw_datasets['validation'] = raw_eval_dataset

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # load model
    model = SentenceTransformer('all-MiniLM-L12-v1').to('cuda')

    scores = []

    train_samples_list = []
    progressbar = tqdm(range(len(raw_train_dataset)), desc="Generate sentence embeddings for train set.")
    for train_sample in raw_train_dataset:
        train_instance = train_sample[sentence1_key]
        train_samples_list.append(train_instance)
        progressbar.update(1)
    
    train_embedding = model.encode(train_samples_list)

    eval_samples_list = []
    progressbar = tqdm(range(len(raw_eval_dataset)), desc="Generate sentence embeddings for eval  set.")
    for eval_sample in raw_eval_dataset:
        evalinstance = eval_sample[sentence1_key]
        eval_samples_list.append(evalinstance)
        progressbar.update(1)
    
    eval_embedding = model.encode(eval_samples_list)

    print(train_embedding.shape)
    print(eval_embedding.shape)

    

    # shape : (#train, #eval)
    scores = util.dot_score(train_embedding, eval_embedding)
    print(scores.shape)
    # shape : (#eval, )
    scores = scores.mean(dim=0)
    # shape : 1
    average_sim = scores.mean(dim=0)
    # print('1', scores.shape)
    # shape : (#train * #eval, )
    # scores = scores.reshape(-1).tolist()

    # average_sim = sum(scores) / len(scores)
    logger.info(f'Average cosine similarity : {average_sim}')
        

if __name__ == "__main__":
    logger.info('\nStart.')
    main()