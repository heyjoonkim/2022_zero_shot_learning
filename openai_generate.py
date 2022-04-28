import argparse
import logging
import os
import sys
import time
import csv
import random

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict, Dataset

from transformers import set_seed

from model_wrapper.ModelWrapper import ModelWrapper

from utils import save_config
from dataset_utils import task_to_path, task_to_keys, task_to_verbalizer

logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="The name of the benchmark to train on.",
        choices=['glue', 'super_glue', 'huggingface'],
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        '--overwrite_output_dir', 
        default=False, 
        action="store_true",
        help='Overwrite output directory.'
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=15, 
        help="Max length for generation."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0, 
        help="Temperature for generating in-context examples."
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=1, 
        help="Top-p sampling."
    )
    parser.add_argument(
        "--frequency_penalty", 
        type=float, 
        default=1, 
        help="Top-p sampling."
    )
    
    # for manual prompt #
    parser.add_argument(
        "--prefix",
        type=str,
        default='',
        help="Prefix prompt.",
    )
    parser.add_argument(
        "--infix",
        type=str,
        default='',
        help="Infix prompt.",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default='',
        help="Postfix prompt.",
    )
    parser.add_argument(
        "--label_token", 
        type=str, 
        default="[LABEL]", 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--input_label_token", 
        type=str, 
        default="[INPUT_LABEL]", 
        help="The place of the label for the input sentence."
    )
    # until here #

    args = parser.parse_args()
    
    return args
    

def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.overwrite_output_dir:
                raise NotADirectoryError(f'Output directory {args.output_dir} exits. Exit program. (overwrite_output_dir=False)')

    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    args.verbalizer = task_to_verbalizer.get(args.task_name)
    args.label2token = {v:k for k,v in args.verbalizer.items()}

    save_config(args)

    # Set seed before initializing model.
    set_seed(args.seed)
    random.seed(args.seed)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = DatasetDict()
    if args.task_name is not None and args.benchmark_name is not None:
        if args.benchmark_name == 'huggingface':
            raw_eval_dataset = load_dataset(args.task_name, split='test')
        else:
            # for mnli 
            if args.task_name == "mnli":
                raw_eval_dataset = load_dataset(args.benchmark_name, args.task_name, split='validation_matched')
            else:
                raw_eval_dataset = load_dataset(args.benchmark_name, args.task_name, split=f'validation')
    # for datasets from file.
    elif args.task_name in task_to_path:
        dataset_processor = task_to_path[args.task_name]["dataset_processor"]
        validation_file_path = task_to_path[args.task_name]["validation"]
        # validation set
        validation_dict = dataset_processor(validation_file_path)
        raw_eval_dataset = Dataset.from_dict(validation_dict)
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')


    raw_datasets['validation'] = raw_eval_dataset
    logger.info('TRAIN / VALIDATION split.')
    for split, dataset in raw_datasets.items():
        logger.info(f'{split} > {len(dataset)}')

    
    # load OpenAI model (set connection)
    model = ModelWrapper(args.model_name_or_path, args.task_name)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )

        sample_num = len(texts[0])
        for sample_index in range(sample_num):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = dict()
            sample_num = len(texts[0])
            result['sentence1'] = examples[sentence1_key]

            # for single sentence tasks
            if sentence2_key is not None:
                result['sentence2'] = examples[sentence2_key]
                            
            # Map labels to IDs (not necessary for GLUE tasks)
            if "label" in examples:
                result["labels"] = examples["label"]
            elif 'label-coarse' in examples:
                result["labels"] = examples['label-coarse']
            else:
                raise NotImplementedError
            return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Running tokenizer on dataset",
    )
    eval_dataset = processed_datasets["validation"]

    ## DONE LOADING DATASET ##
    logger.info("***** Zero/Few-shot Evaluation *****")
    logger.info(f"  Num EVAL  examples = {len(eval_dataset)}")
    logger.info(f"  Random Seed = {args.seed}")
    logger.info(f"  Inference Model = {args.model_name_or_path}")
    
    start_time = time.time()

    
    generation_writer = os.path.join(args.output_dir, "test.tsv")
    with open(generation_writer, "w") as file_writer:
        tsv_writer = csv.writer(file_writer, delimiter='\t')
        for step, inputs in tqdm(enumerate(eval_dataset)):
            
            sentence1 = inputs['sentence1']
            sentence2 = inputs['sentence2'] if 'sentence2' in inputs else ''
            label = inputs['labels']
            input_label_token = args.label2token[label]

            original_input = args.prefix + sentence1 + args.infix + sentence2 + args.postfix
            if args.input_label_token in original_input:
                original_input = original_input.replace(args.input_label_token, input_label_token)

            row = [step, label, sentence1]

            if 'sentence2' in inputs:
                row.append(sentence2)

            for index, (label_token, label) in enumerate(args.verbalizer.items()):
                assert index == label, f'index {index} != label {label}'
                label_dependent_input = original_input.replace(args.label_token, label_token)
                l = len(label_dependent_input)

                wrong_generation_count = 0
                while True:
                    generated_text = model.generate(
                        original_input=label_dependent_input,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        frequency_penalty=args.frequency_penalty,
                        **inputs
                    )

                    generated_text = generated_text.strip().split('\n')[0].strip()
                    if len(generated_text) == 0:
                        wrong_generation_count += 1
                        print(f'Nothing generated. Retry.... {wrong_generation_count}')
                        if wrong_generation_count >= 5:
                            print("Cannot generate a sample for input :")
                            print(label_dependent_input)
                            break
                    else:
                        # to match the format from the transformers code
                        generated_outputs = [generated_text]
                        break

                row.append(generated_outputs)

            tsv_writer.writerow(row)
        
    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time}')

if __name__ == "__main__":
    logger.info('\nStart.')
    main()
