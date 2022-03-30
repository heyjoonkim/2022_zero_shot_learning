#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import argparse
import logging
import os
import sys
import time
import csv
import random

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset

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
        "--n_samples", 
        type=int, 
        default=0, 
        help="Number of samples for in-context learning."
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

    save_config(args)

    # Set seed before initializing model.
    set_seed(args.seed)
    random.seed(args.seed)

    if args.task_name is not None and args.task_name not in task_to_path:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", args.task_name)
    else:
        raise NotImplementedError('Tasks not in GLUE is not implemented yet.')

    # Labels
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
   
    
    # load OpenAI model (set connection)
    model = ModelWrapper(args.model_name_or_path, args.task_name)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = dict()
        sample_num = len(texts[0])
        result['sentence1'] = examples[sentence1_key]

        # for single sentence tasks
        if sentence2_key is None:
            pass
        else:
            result['sentence2'] = examples[sentence2_key]
            input_sentences = []

            for sample_index in range(sample_num):
                input_sentence = args.prefix + texts[0][sample_index] + args.infix + texts[1][sample_index] + args.postfix
                input_sentences.append(input_sentence)

        result['input_sentence'] = input_sentences
        
        # Map labels to IDs (not necessary for GLUE tasks)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    processed_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Preparing dataset",
    )

    if "train" not in processed_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = processed_datasets["train"]

    if "validation" not in processed_datasets and "validation_matched" not in processed_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    if args.task_name == "mnli":
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataset_mm = processed_datasets["validation_matched"]
    else:
        eval_dataset = processed_datasets["validation"]

    logger.info(f'# TRAIN dataset : {len(train_dataset)}')
    logger.info(f'# Eval  dataset : {len(eval_dataset)}')
    # TODO : fix?
    # for random sampling #
    train_dataset_length = len(train_dataset)
    ## DONE LOADING DATASET ##
    
    correct_count=0

    start_time = time.time()

    result_writer = os.path.join(args.output_dir, "wrong_samples.tsv")
    with open(result_writer, "w") as file_writer:
        tsv_writer = csv.writer(file_writer, delimiter='\t')
        tsv_writer.writerow([args.prefix, args.infix, args.postfix])
        tsv_writer.writerow(['index', 'sentence1', 'sentence2', 'prediction', 'label', 'top_logprobs'])
        for index, inputs in tqdm(enumerate(eval_dataset)):

            ## select 
            if args.n_samples > 0:
                in_context_samples = []
                for _ in range(args.n_samples):
                    random_index = random.randint(0, train_dataset_length-1)
                    random_sample = train_dataset[random_index]
                    random_sample_input_sentence = random_sample['input_sentence']
                    random_sample_label = random_sample['labels']
                    for k,v in args.verbalizer.items():
                        if random_sample_label == v:
                            random_sample_input_sentence = random_sample_input_sentence + k
                            break
                    in_context_samples.append(random_sample_input_sentence)
                in_context_samples = ' '.join(in_context_samples)
                inputs['input_sentence'] = ' '.join([in_context_samples, inputs['input_sentence']])

            label = inputs['labels']
            prediction, results_dict = model.forward(**inputs)


            if prediction == label:
                correct_count += 1
            else:
                tsv_writer.writerow([index, inputs['sentence1'], inputs['sentence2'], prediction, label, str(results_dict)])

            # TODO : removes
            # if index > 20:
            #     break
            
        result = correct_count / len(eval_dataset) * 100
        logger.info(f'Result : {correct_count} / {len(eval_dataset)} = {result}%')

        tsv_writer.writerow([correct_count, len(eval_dataset), result])

        
    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time}')

if __name__ == "__main__":
    print('start')
    main()
