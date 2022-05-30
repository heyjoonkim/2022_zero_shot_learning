""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import os
import json
import time


from datasets import DatasetDict, Dataset
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
        "--dataset_dir", 
        type=str, 
        default=None, 
        help="Path for the generated datasets."
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
    
    # for datasets from file.
    if args.task_name in generated_task_to_path:
        dataset_processor = generated_task_to_path[args.task_name]["dataset_processor"]
        validation_file_path = generated_task_to_path[args.task_name]["validation"]
        validation_file_path = os.path.join(args.dataset_dir, validation_file_path)

        # validation set
        validation_dict = dataset_processor(validation_file_path)
        raw_eval_dataset = Dataset.from_dict(validation_dict)
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')

    raw_datasets['validation'] = raw_eval_dataset

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # load model
    model = SentenceTransformer('all-MiniLM-L12-v1').to('cuda')

    scores = []

    progressbar = tqdm(range(len(raw_eval_dataset)))
    for generated_samples in raw_eval_dataset:
        # we ignore sentence-pair cases.
        test_sample = generated_samples[sentence1_key]
        generated_samples_list = []
        for label_index in range(len(args.verbalizer)):
            generated_key = f'samples{label_index}'
            generated_samples_per_class = generated_samples.get(generated_key, None)
            assert generated_samples_per_class != None

            generated_samples_list = generated_samples_list + generated_samples_per_class
        
        test_sample = [test_sample]

        test_embedding = model.encode(test_sample)
        generated_embedding = model.encode(generated_samples_list)

        # shape : (1, #generated_samples)
        score = util.dot_score(test_embedding, generated_embedding)
        # shape : (#generated_samples, )
        score = score.squeeze(0).tolist()
        scores = scores + score

        progressbar.update(1)

    average_sim = sum(scores) / len(scores)
    logger.info(f'Average cosine similarity : {average_sim}')
        
    

                
if __name__ == "__main__":
    logger.info('\nStart.')
    main()