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
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset

from transformers import set_seed

from model_wrapper.ModelWrapper import ModelWrapper

from utils import save_config
from dataset_utils import task_to_path, task_to_keys, task_to_verbalizer, GLUE, task_to_label

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
        '--log_results', 
        default=False, 
        action="store_true",
        help='Log prediction results to tsv file'
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
    parser.add_argument(
        "--demo_accuracy", 
        type=float, 
        default=1, 
        help="Accuracy of demonstration samples for in-context learning."
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

    if args.task_name is not None and args.task_name in GLUE:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", args.task_name)
    elif args.task_name is not None and args.task_name == 'hate':
        datasets = load_dataset("tweet_eval", args.task_name)
    else:
        datasets = load_dataset(args.task_name)

    # Labels
    if args.task_name in task_to_label:
        label_key = task_to_label[args.task_name]
    else:
        label_key = 'label'

    label_list = datasets["train"].features[label_key].names
    num_labels = len(label_list)
    
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


        if sentence2_key is None:
            # for single sentence tasks
            input_sentences = []
            for sample_index in range(sample_num):
                input_sentence = args.prefix + texts[0][sample_index] + args.postfix
                input_sentences.append(input_sentence)
        else:
            # for two sentence tasks
            result['sentence2'] = examples[sentence2_key]
            input_sentences = []

            for sample_index in range(sample_num):
                input_sentence = args.prefix + texts[0][sample_index] + args.infix + texts[1][sample_index] + args.postfix
                input_sentences.append(input_sentence)

        result['input_sentence'] = input_sentences
        
        if label_key in examples:
            result["labels"] = examples[label_key]
        return result

    processed_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Preparing dataset",
    )

    if "train" not in processed_datasets:
        raise ValueError("requires a train dataset")

    train_dataset = processed_datasets["train"]
    train_dataset = train_dataset.filter(lambda example: example['labels'] in args.verbalizer.values())
    # if "validation" not in processed_datasets and "validation_matched" not in processed_datasets:
    #     raise ValueError("requires a validation dataset")
    if args.task_name == "mnli":
        eval_dataset = processed_datasets["validation_matched"]
        eval_dataset_mm = processed_datasets["validation_mismatched"]
    elif args.task_name == "trec" or args.task_name == "ag_news" or args.task_name == "poem_sentiment":
        eval_dataset = processed_datasets["test"]
    else:
        eval_dataset = processed_datasets["validation"]

    if args.task_name == "ag_news":
        eval_dataset = eval_dataset.select(range(300))
    # if args.task_name == "hate":
    #     eval_dataset = eval_dataset.select(range(100))

    logger.info(f'# TRAIN dataset : {len(train_dataset)}')
    logger.info(f'# Eval  dataset : {len(eval_dataset)}')
       
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_sentence']}")

    # for random sampling #
    train_dataset_length = len(train_dataset)
    ## DONE LOADING DATASET ##
    
    correct_count=0

    start_time = time.time()

    if args.log_results:
        result_writer = os.path.join(args.output_dir, "results.tsv")
        file_writer = open(result_writer, "w")
        tsv_writer = csv.writer(file_writer, delimiter='\t')
        tsv_writer.writerow(['index', 'prediction', 'label', 'top_logprobs'])
    if args.n_samples > 0:
        in_context_samples = []
        random_indices = []
        for _ in range(args.n_samples):
            random_indices.append(random.randint(0, train_dataset_length-1))
        for random_index in random_indices:
            random_sample = train_dataset[random_index]
            random_sample_input_sentence = random_sample['input_sentence']
            # A% demo accuracy
            if np.random.rand(1)[0] <= args.demo_accuracy:
                random_sample_label = random_sample['labels']
            else:
                labels = list(args.verbalizer.values())
                labels.remove(random_sample['labels'])
                random_sample_label = random.choice(labels)
            for k,v in args.verbalizer.items():
                if random_sample_label == v:
                    random_sample_input_sentence = random_sample_input_sentence + k
                    break
            in_context_samples.append(random_sample_input_sentence)
        in_context_samples = '\n\n\n'.join(in_context_samples)
        logger.info(f'in context samples\n{in_context_samples}')

    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for index, inputs in tqdm(enumerate(eval_dataset)):
        ## select 
        if args.n_samples > 0:
            inputs['input_sentence'] = '\n\n\n'.join([in_context_samples, inputs['input_sentence']])

        label = inputs['labels']
        prediction, results_dict = model.forward(**inputs)

        if prediction == label:
            is_correct = 1
        else:
            is_correct = 0
        accs.append(is_correct)
        recalls[label].append(is_correct)
        precisions[prediction].append(is_correct)

        if args.log_results:
            tsv_writer.writerow([index, prediction, label, str(results_dict)])

    acc = np.mean(accs)
    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision+recall==0:
            f1s.append(0)
        else:
            f1s.append(2*precision*recall / (precision+recall))
    f1 = np.mean(f1s)

    
    result_path = str(args.output_dir).rsplit('/',1)[0] + '/resuts.txt'
    with open(result_path, "a+") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow([args.seed, acc*100, f1*100])

    logger.info(f'ACC : {acc*100} F1: {f1*100}')
    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time}')

if __name__ == "__main__":
    print('start')
    main()
