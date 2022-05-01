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
from dataset_utils import generated_task_to_path, task_to_keys, task_to_verbalizer, prepare_generated_incontext_sampling, prepend_incontext_samples

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
        "--dataset_dir", 
        type=str, 
        default=None, 
        help="Path for the generated datasets."
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
    
    # for Few-shot inference
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=0, 
        help="Number of samples for in-context learning."
    )
    parser.add_argument(
        '--balance_sample', 
        default=False, 
        action="store_true",
        help='Balance samples per label for in-context learning.'
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

    ## load generated dataset ##
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

    raw_datasets['validation'] = raw_eval_dataset
    logger.info('TRAIN / VALIDATION split.')
    for split, dataset in raw_datasets.items():
        logger.info(f'{split} > {len(dataset)}')
    ## done loading generated dataset ##

    
    # load OpenAI model (set connection)
    model = ModelWrapper(args.model_name_or_path, args.task_name)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    label2samples_list, full_train_samples_list = prepare_generated_incontext_sampling(
        generated_samples=raw_datasets['validation'],
        verbalizer=args.verbalizer,
        prefix=args.prefix,
        infix=args.infix,
        postfix=args.postfix,
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key)
    
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
            input_sentences = []

            # for single sentence tasks
            if sentence2_key is None:
                for sample_index in range(sample_num):
                    input_sentence = args.prefix + texts[0][sample_index] + args.infix + args.postfix
                    input_sentences.append(input_sentence)
            else:
                result['sentence2'] = examples[sentence2_key]
                for sample_index in range(sample_num):
                    input_sentence = args.prefix + texts[0][sample_index] + args.infix + texts[1][sample_index] + args.postfix
                    input_sentences.append(input_sentence)

            result['input_sentence'] = input_sentences
            
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
    logger.info(f"  K = {args.n_samples}")
    logger.info(f"  Inference Model = {args.model_name_or_path}")
    
    correct_count=0
    start_time = time.time()
    for step, inputs in tqdm(enumerate(eval_dataset)):
        
        # in-context samples generated conditioned by the input x.
        if args.n_samples > 0:
            incontext_samples, sep = prepend_incontext_samples(
                label2samples=label2samples_list[step],
                full_train_samples=full_train_samples_list[step],
                k=args.n_samples,
                balance_sample=args.balance_sample,
            )
            inputs['input_sentence'] = incontext_samples + sep + inputs['input_sentence']
         
        label = inputs['labels']
        prediction, results_dict = model.forward(**inputs)

        if prediction == label:
            correct_count += 1
    result = correct_count / len(eval_dataset) * 100
    logger.info(f'Result : {correct_count} / {len(eval_dataset)} = {result}%')
    
    end_time = time.time()
    logger.info(f'Total task time : {end_time - start_time}')

if __name__ == "__main__":
    logger.info('\nRunning : openai_generated_main.py')
    
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')
