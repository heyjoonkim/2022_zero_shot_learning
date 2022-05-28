import argparse
import logging
import os
import random
import json
import time

import datasets
from datasets import load_dataset, load_metric, DatasetDict, Dataset
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)
import torch
import deepspeed

from model_wrapper.TransformersModelWrapper import GPT2Wrapper
from utils import save_config
from dataset_utils import task_to_path, task_to_keys, task_to_verbalizer, prepare_incontext_sampling, prepend_incontext_samples

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

    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.overwrite_output_dir:
                logger.info(f'Output directory {args.output_dir} exits. Exit program. (overwrite_output_dir=False)')
                exit()
            
    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    args.verbalizer = task_to_verbalizer.get(args.task_name)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    # Handle the repository creation & SummaryWriter
    save_config(args)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
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
        
    num_labels = len(args.verbalizer)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # For gpt-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # TODO: only inject pad_token_id in case of GPT
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, 
        num_labels=num_labels, 
        finetuning_task=args.task_name, 
        pad_token_id=tokenizer.unk_token_id
    )


    logger.info(f'Start loading {args.model_name_or_path} model...')
    model_loading_start_time = time.time()
    model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path, verbalizer=args.verbalizer)
    model_loading_end_time = time.time()
    logger.info(f'Total time for loading model : {model_loading_end_time - model_loading_start_time}')

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    label2samples, full_train_samples = prepare_incontext_sampling(
        train_samples=raw_datasets['train'],
        verbalizer=args.verbalizer,
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key,
        prefix=args.prefix,
        infix=args.infix,
        postfix=args.postfix)

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
                # SST-2, SST-5, AGNews
                result["labels"] = examples["label"]
            elif 'label-coarse' in examples:
                # TREC
                result["labels"] = examples['label-coarse']
            else:
                raise NotImplementedError
            return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Preprocessing datasets...",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
       
    # Get the metric function
    if args.task_name is not None and args.benchmark_name is not None:
        if args.benchmark_name == 'huggingface':
            metric = load_metric("accuracy")
        else:
            metric = load_metric(args.benchmark_name, args.task_name)

    # Evaluate! 
    logger.info("***** Zero/Few-shot Evaluation *****")
    logger.info(f"  Task name           = {args.task_name}")
    logger.info(f"  Num TRAIN examples  = {len(train_dataset)}")
    logger.info(f"  Num EVAL  examples  = {len(eval_dataset)}")
    logger.info(f"  Random Seed         = {args.seed}")
    logger.info(f"  K                   = {args.n_samples}")
    logger.info(f"  Inference Model     = {args.model_name_or_path}")
         
    # for analysis
    prediction_dict = {}

    start_time = time.time()
    model.eval()

    # we select a set of in-context samples
    # and use it as the in-context sample for all test dataset.
    incontext_samples, sep = prepend_incontext_samples(
        label2samples=label2samples,
        full_train_samples=full_train_samples,
        k=args.n_samples,
        balance_sample=args.balance_sample,
    )

    logger.info(f'=== in-context samples ===\n{incontext_samples}\n=====================')
        
    progressbar = tqdm(range(len(eval_dataset)))
    for step, inputs in enumerate(eval_dataset):
        # prepend in-context samples
        if args.n_samples > 0:
            inputs['input_sentence'] = incontext_samples + sep + inputs['input_sentence']

        if step == 0:
            logger.info(f'Print first input for debugging : \n{inputs["input_sentence"]}')
            
        label = torch.tensor(inputs['labels']).to('cuda').unsqueeze(dim=0)

        # prediction  : predicted label index
        # predictions : logit values for each label
        prediction, predictions = model(**inputs)
            
        metric.add_batch(
            predictions=prediction,
            references=label,
        )

        # for analysis : save predictions
        prediction = prediction.cpu().item()
        prediction_dict[prediction] = prediction_dict.get(prediction, 0) + 1

        progressbar.update(1)

    eval_metric = metric.compute()

    if args.n_samples == 0:
        logger.info(f"** Zero-shot evaluation result : {eval_metric}")
    else:
        logger.info(f"** {args.n_samples}-shot evaluation result : {eval_metric}")

    logger.info(f'Predictions distribution : {prediction_dict}')

    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time} sec.')
    logger.info("Done.")
                
if __name__ == "__main__":
    logger.info('\nRunning : transformers_main.py')
    
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')