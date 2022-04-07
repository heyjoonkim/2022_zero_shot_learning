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
        "--dataset_dir", 
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
        default=30, 
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
        "--positive_prompt",
        type=str,
        default=None,
        help="Prompt for generating positive in-context sample.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Prompt for generating negative in-context sample.",
    )
    parser.add_argument(
        "--neutral_prompt",
        type=str,
        default=None,
        help="Prompt for generating neutral in-context sample.",
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

    if args.dataset_dir is not None:
        if not os.path.isdir(args.dataset_dir):
            os.makedirs(args.dataset_dir, exist_ok=True)
        else:
            if not args.dataset_dir:
                raise NotADirectoryError(f'Output directory {args.dataset_dir} exits. Exit program. (overwrite_output_dir=False)')


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
        result['sentence1'] = examples[sentence1_key]

        # for single sentence tasks
        if sentence2_key is not None:
            result['sentence2'] = examples[sentence2_key]

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

    if "validation" not in processed_datasets and "validation_matched" not in processed_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    if args.task_name == "mnli":
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataset_mm = processed_datasets["validation_matched"]
    else:
        eval_dataset = processed_datasets["validation"]

    logger.info(f'# Eval  dataset : {len(eval_dataset)}')
    ## DONE LOADING DATASET ##
    
    start_time = time.time()

    result_writer = os.path.join(args.dataset_dir, f"t_{args.temperature}_p_{args.top_p}_fp_{args.frequency_penalty}.tsv")
    with open(result_writer, "w") as file_writer:
        tsv_writer = csv.writer(file_writer, delimiter='\t')
        tsv_writer.writerow([args.positive_prompt, args.negative_prompt])
        tsv_writer.writerow(['index', 'sentence1', 'sentence2', 'label', 'entailment', 'not entailment'])
        for index, inputs in tqdm(enumerate(eval_dataset)):
            
            row = [index, inputs['sentence1'], inputs['sentence2'], inputs['labels']]

            generated_result = model.generate(
                positive_prompt=args.positive_prompt, 
                negative_prompt=args.negative_prompt,
                neutral_prompt=args.neutral_prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                frequency_penalty=args.frequency_penalty,
                **inputs
            )


            for generated_sentence, expected_label in generated_result:
                if generated_sentence is None:
                    continue

                row.append(generated_sentence)

            tsv_writer.writerow(row)
    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time}')

if __name__ == "__main__":
    print('start')
    main()
