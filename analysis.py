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
import sys
import time

from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, DatasetDict

from transformers import AutoTokenizer

from dataset_utils import task_to_path, task_to_keys, task_to_verbalizer

logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of task to analyze.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--split",
        type=str,
        default='train',
        help="The name of split."
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="The name of the benchmark to train on.",
        choices=['glue', 'super_glue', 'huggingface'],
    )

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

    if args.task_name is not None and args.task_name not in task_to_path:
        logger.info(f'Loading from Huggingface Datasets : {args.task_name}')
        # Downloading and loading a dataset from the hub.
        # datasets = load_dataset("glue", args.task_name)
        if args.benchmark_name in ['glue', 'super_glue']:
            datasets = load_dataset(args.benchmark_name, args.task_name)
        else:
            datasets = load_dataset(args.task_name, split=args.split)
    else:
        logger.info(f'Loading from File : {args.task_name}')
        datasets = DatasetDict()
        dataset_processor = task_to_path[args.task_name]["dataset_processor"]
        train_file_path = task_to_path[args.task_name][args.split]

        # train set
        train_dict = dataset_processor(train_file_path)
        raw_train_dataset = Dataset.from_dict(train_dict)

        datasets[args.split] = raw_train_dataset

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

        input_sentences = []
        # for single sentence tasks
        if sentence2_key is None:
            for sample_index in range(sample_num):
                input_sentence = texts[0][sample_index]
                input_sentences.append(input_sentence)
        else:
            result['sentence2'] = examples[sentence2_key]
        
            for sample_index in range(sample_num):
                input_sentence = texts[0][sample_index] + ' ' + texts[1][sample_index]
                input_sentences.append(input_sentence)

        result['input_sentence'] = input_sentences
        
        # Map labels to IDs (not necessary for GLUE tasks)
        if "label" in examples:
            result["labels"] = examples["label"]
        elif 'label-coarse' in examples:
            result['labels'] = examples['label-coarse']

        return result

    processed_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets[args.split].column_names,
        desc="Preparing dataset",
    )

    dataset = processed_datasets[args.split]

    logger.info(f'# {args.split} dataset : {len(dataset)}')

    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    logger.info('Done loading tokenizer.')

    start_time = time.time()

    total_input_sentence_length = 0
    max_input_sentence_length = 0
    min_input_sentence_length = float('inf')

    total_sentence1_length = 0
    max_sentence1_length = 0
    min_sentence1_length = float('inf')
    if sentence2_key is not None:
        total_sentence2_length = 0
        max_sentence2_length = 0
        min_sentence2_length = float('inf')
    
    label2count = {}

    for index, inputs in tqdm(enumerate(dataset)):

        labels = inputs['labels']
        input_sentence = inputs['input_sentence']
        sentence1 = inputs['sentence1']
        if 'sentence2' in inputs:
            sentence2 = inputs['sentence2']
        else:
            sentence2 = None

        # total sentence length
        input_sentence_token_length = len(tokenizer(input_sentence)['input_ids'])
        total_input_sentence_length += input_sentence_token_length
        if max_input_sentence_length < input_sentence_token_length:
            max_input_sentence_length = input_sentence_token_length
        if min_input_sentence_length > input_sentence_token_length:
            min_input_sentence_length = input_sentence_token_length

        # sentence1 length
        input_sentence1_token_length = len(tokenizer(sentence1)['input_ids'])
        total_sentence1_length += input_sentence1_token_length
        if max_sentence1_length < input_sentence1_token_length:
            max_sentence1_length = input_sentence1_token_length
        if min_sentence1_length > input_sentence1_token_length:
            min_sentence1_length = input_sentence1_token_length
        
        # sentence2 length (if any)
        if sentence2_key is not None:
            input_sentence2_token_length = len(tokenizer(sentence2)['input_ids'])
            total_sentence2_length += input_sentence2_token_length
            if max_sentence2_length < input_sentence2_token_length:
                max_sentence2_length = input_sentence2_token_length
            if min_sentence2_length > input_sentence2_token_length:
                min_sentence2_length = input_sentence2_token_length

        label2count[labels] = label2count.get(labels, 0) + 1 

        
    average_input_sentence_token_length = total_input_sentence_length / len(dataset)
    logger.info(f'AVG input token length : {average_input_sentence_token_length}')
    logger.info(f'MAX input token length : {max_input_sentence_length}')
    logger.info(f'MIN input token length : {min_input_sentence_length}')
    logger.info('\n\n')
    average_sentence1_token_length = total_sentence1_length / len(dataset)
    logger.info(f'AVG sentence1 token length : {average_sentence1_token_length}')
    logger.info(f'MAX sentence1 token length : {max_sentence1_length}')
    logger.info(f'MIN sentence1 token length : {min_sentence1_length}')
    logger.info('\n\n')
    if sentence2_key is not None:
        average_sentence2_token_length = total_sentence2_length / len(dataset)
        logger.info(f'AVG sentence2 token length : {average_sentence2_token_length}')
        logger.info(f'MAX sentence2 token length : {max_sentence2_length}')
        logger.info(f'MIN sentence2 token length : {min_sentence2_length}')
        logger.info('\n\n')
    logger.info(f'label split : {label2count}')

    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time}')

if __name__ == "__main__":
    print('start')
    main()
