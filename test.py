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
import json
import pandas as pd    
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset


from transformers import set_seed

from model_wrapper.ModelWrapper import ModelWrapper

from utils import save_config
from dataset_utils import task_to_path, task_to_keys, task_to_verbalizer, GLUE, task_to_label
from preprocess import rte_preprocess

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
        rte_preprocess,
        batched=False,
        remove_columns=datasets["train"].column_names,
        desc="Preparing dataset",
    )

    if "train" not in processed_datasets:
        raise ValueError("requires a train dataset")
    train_dataset = processed_datasets["train"]

    # with open(f"data/tweet_eval-hate_16_{args.seed}_train.jsonl", "r") as t_f:
    #     pre_sampled_data = [json.loads(line) for line in t_f]
    #     print(pre_sampled_data)
    # if "validation" not in processed_datasets and "validation_matched" not in processed_datasets:
    #     raise ValueError("requires a validation dataset")
    if args.task_name == "mnli":
        eval_dataset = processed_datasets["validation_matched"]
        eval_dataset_mm = processed_datasets["validation_mismatched"]
    elif args.task_name == "trec":
        eval_dataset = processed_datasets["test"]
    else:
        eval_dataset = processed_datasets["validation"]


    logger.info(f'# TRAIN dataset : {len(train_dataset)}')
    logger.info(f'# Eval  dataset : {len(eval_dataset)}')

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]['input_sentence']}")

    # for random sampling #
    train_dataset_length = len(train_dataset)
    ## DONE LOADING DATASET ##
    no_dataset = train_dataset.filter(lambda example: example['labels']==2)
    pos_dataset = train_dataset.filter(lambda example: example['labels']==1)
    neg_dataset = train_dataset.filter(lambda example: example['labels']==0)

    correct_count=0

    start_time = time.time()

    if args.log_results:
        result_writer = os.path.join(args.output_dir, "results.tsv")
        file_writer = open(result_writer, "w")
        tsv_writer = csv.writer(file_writer, delimiter='\t')
        tsv_writer.writerow(['index', 'prediction', 'label', 'top_logprobs'])

    # in_context_samples = []
    # for d in pre_sampled_data:
    #     sample_input_sentence = d['input'] + "\n" + d['output']
    #     in_context_samples.append(sample_input_sentence)
    # in_context_samples = '\n\n\n'.join(in_context_samples)
    # logger.info(f'in context samples\n{in_context_samples}')
    if args.n_samples > 0:
        in_context_samples = []
        no_indices = []
        pos_indices = []
        neg_indices = []
        for _ in range(6):
            no_indices.append(random.randint(0, no_dataset-1))
        for _ in range(5):
            pos_indices.append(random.randint(0, pos_dataset-1))
            neg_indices.append(random.randint(0, neg_dataset-1))
        for idx in range(5)
            no_sample = no_dataset[no_indices[idx]]
            no_sample_input_sentence = no_sample['input_sentence']
            # A% demo accuracy
            if np.random.rand(1)[0] <= args.demo_accuracy:
                no_sample_label = no_sample['labels']
            else:
                labels = list(args.verbalizer.values())
                labels.remove(no_sample['labels'])
                no_sample_label = random.choice(labels)
            for k,v in args.verbalizer.items():
                if no_sample_label == v:
                    no_sample_input_sentence = no_sample_input_sentence + k
                    break
            in_context_samples.append(no_sample_input_sentence)

            pos_sample = pos_dataset[pos_indices[idx]]
            pos_sample_input_sentence = pos_sample['input_sentence']
            # A% demo accuracy
            if np.random.rand(1)[0] <= args.demo_accuracy:
                pos_sample_label = pos_sample['labels']
            else:
                labels = list(args.verbalizer.values())
                labels.remove(pos_sample['labels'])
                pos_sample_label = random.choice(labels)
            for k,v in args.verbalizer.items():
                if pos_sample_label == v:
                    pos_sample_input_sentence = pos_sample_input_sentence + k
                    break
            in_context_samples.append(pos_sample_input_sentence)

            neg_sample = neg_dataset[neg_indices[idx]]
            neg_sample_input_sentence = neg_sample['input_sentence']
            # A% demo accuracy
            if np.random.rand(1)[0] <= args.demo_accuracy:
                neg_sample_label = neg_sample['labels']
            else:
                labels = list(args.verbalizer.values())
                labels.remove(neg_sample['labels'])
                neg_sample_label = random.choice(labels)
            for k,v in args.verbalizer.items():
                if neg_sample_label == v:
                    neg_sample_input_sentence = neg_sample_input_sentence + k
                    break
            in_context_samples.append(neg_sample_input_sentence)
        no_sample = no_dataset[no_indices[5]]
        no_sample_input_sentence = no_sample['input_sentence']
        # A% demo accuracy
        if np.random.rand(1)[0] <= args.demo_accuracy:
            no_sample_label = no_sample['labels']
        else:
            labels = list(args.verbalizer.values())
            labels.remove(no_sample['labels'])
            no_sample_label = random.choice(labels)
        for k,v in args.verbalizer.items():
            if no_sample_label == v:
                no_sample_input_sentence = no_sample_input_sentence + k
                break
        in_context_samples.append(no_sample_input_sentence)
        in_context_samples = '\n\n\n'.join(in_context_samples)
        logger.info(f'in context samples\n{in_context_samples}')

    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for index, inputs in tqdm(enumerate(eval_dataset)):
        ## select 

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

    logger.info(f'ACC : {acc*100} F1: {f1*100}')
    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time}')

if __name__ == "__main__":
    print('start')
    main()
