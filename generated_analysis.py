""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import os
import json
import time

import datasets
from datasets import DatasetDict, Dataset
from tqdm.auto import tqdm

import torch

from dataset_utils import generated_task_to_path, task_to_keys, task_to_verbalizer, prepare_generated_incontext_sampling

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
        "--dataset_dir", 
        type=str, 
        default=None, 
        help="Path for the generated datasets."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        '--ds_config', 
        default='ds_config.json', 
        type=str, 
        help='deepspeed config'
    )
    parser.add_argument(
        '--local_rank', 
        default=0, 
        type=int, 
        help='node rank for distributed training'
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
    
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    elif args.task_name is None:
        raise NotImplementedError('Tasks for GLUE benchmarks are implemented yet.')

    # post init get batch and zero option from ds config
    with open(args.ds_config, "r", encoding="utf-8") as ds_f:
        ds_config = json.load(ds_f)
    args.per_device_batch_size = ds_config['train_micro_batch_size_per_gpu']
    args.gradient_accumulation_steps = ds_config['gradient_accumulation_steps']
    if ds_config.get("zero_optimization"):
        args.is_zero3 = ds_config["zero_optimization"]["stage"] == 3
    else:
        args.is_zero3 = False

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
    logger.setLevel(logging.INFO if args.local_rank == 0 else logging.ERROR)

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

    label2samples_list, full_train_samples_list = prepare_generated_incontext_sampling(
        generated_samples=raw_datasets['validation'],
        verbalizer=args.verbalizer,
        prefix=args.prefix,
        infix=args.infix,
        postfix=args.postfix,
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key,
        append_label=False)

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

    # Evaluate! 
    if args.local_rank == 0:
        logger.info("***** Generated Dataset Analysis *****")
        logger.info(f"  Num EVAL  examples = {len(eval_dataset)}")
        logger.info(f"  Random Seed = {args.seed}")
         
    
    start_time = time.time()
    for step, inputs in tqdm(enumerate(eval_dataset), disable=(args.local_rank != 0)):

        print(f'INDEX : {step}')
        print(f'LABEL : {args.label2token[inputs["labels"]]}')
        print(f'Input sentence : \n{inputs["input_sentence"]}')

        label2samples=label2samples_list[step]

        for label, samples in label2samples.items():
            print(f'Generated label : {args.label2token[label]}')
            for i, sample in enumerate(samples):
                print(i, sample)

        print('-' * 50)
           

    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time} sec.')
                
if __name__ == "__main__":
    logger.info('\nStart.')
    main()