import argparse
import logging
import os
import random
import json
import time
import csv

import datasets
from datasets import load_dataset, DatasetDict, Dataset
from tqdm.auto import tqdm

import transformers
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
import torch
import deepspeed

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
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
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

    # for Few-shot inference
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=0, 
        help="Number of samples for in-context learning."
    )
    # manual prompts for generation #
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

    # hyperparams for generation #
    parser.add_argument(
        "--generation_max_length", 
        type=int, 
        default=10, 
        help="Max length for generation."
    )
    parser.add_argument(
        '--generation_min_length', 
        default=10, 
        type=int, 
        help='Min length for generation.'
    )
    parser.add_argument(
        "--no_repeat_ngram_size", 
        type=int, 
        default=2, 
        help="no_repeat_ngram_size."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.5, 
        help="Temperature for sampling."
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
    dschf = HfDeepSpeedConfig(args.ds_config)
    deepspeed.init_distributed()
    args.world_size = torch.distributed.get_world_size()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if args.local_rank == 0 else logging.ERROR)

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
    args.label2token = {v:k for k,v in args.verbalizer.items()}

    if args.local_rank == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    # Handle the repository creation & SummaryWriter
    if args.local_rank == 0:
        save_config(args)

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

    if args.local_rank == 0:
        logger.info('TRAIN / VALIDATION split.')
        for split, dataset in raw_datasets.items():
            logger.info(f'{split} > {len(dataset)}')
    
    if args.local_rank == 0:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(raw_eval_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {raw_eval_dataset[index]}.")
    
    # Labels
    if args.task_name is not None and args.benchmark_name is not None:
        if args.benchmark_name == 'huggingface':
            # TODO : fix?
            label_list = raw_datasets["validation"].features["label-coarse"].names
        else:
            # label_list : ['entailment', 'not_entailment']
            label_list = raw_datasets["validation"].features["label"].names
        num_labels = len(label_list)
    elif args.task_name in task_to_path:
        label_list = set(raw_datasets["validation"]['label'])
        num_labels = len(label_list)
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')

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

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

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

    if args.local_rank != 0:
        torch.distributed.barrier()
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Preprocessing datasets...",
    )
    if args.local_rank == 0:
        torch.distributed.barrier()

    eval_dataset = processed_datasets["validation"]


    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 1):
        logger.info(f"Sample {index} of the evaluation set:")
        logger.info(f'{eval_dataset[index]}')
    
    # deepspeed initialization
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=None, lr_scheduler=None, config_params=args.ds_config)
    
    # Generate! 
    if args.local_rank == 0:
        logger.info("***** Zero/Few-shot Evaluation *****")
        logger.info(f"  Num EVAL  examples = {len(eval_dataset)}")
        logger.info(f"  Random Seed = {args.seed}")
        logger.info(f"  Inference Model = {args.model_name_or_path}")
         
    
    # ignore generating comma(,) and new_line(\n)
    ignored_sequences = [',', ' ,', ' \n', '\n', ' \t', '\t']
    # bad_words_ids = tokenizer(ignored_sequences, add_prefix_space=True).input_ids
    # bad_words_ids = [ tokenizer.encode(ignored_sequence, add_prefix_space=True) for ignored_sequence in ignored_sequences]
    bad_words_ids = [ tokenizer.encode(ignored_sequence) for ignored_sequence in ignored_sequences]
    logger.info(f"  Ignored sequences : {ignored_sequences} -> {bad_words_ids}")

    start_time = time.time()
    model_engine.eval()

    generation_writer = os.path.join(args.output_dir, "test.tsv")
    with open(generation_writer, 'w') as file_writer:
        tsv_writer = csv.writer(file_writer, delimiter='\t')
        for step, inputs in tqdm(enumerate(eval_dataset), disable=(args.local_rank != 0)):
            # input sentences
            sentence1 = inputs['sentence1']
            sentence2 = inputs['sentence2'] if 'sentence2' in inputs else ''

            # original input with manually selected prompts
            original_input = args.prefix + sentence1 + args.infix + sentence2 + args.postfix
            
            # gold label for the input
            label = inputs['labels']

            # add label and input sentences to write in .tsv file
            row = [step, label, sentence1]
            if 'sentence2' in inputs:
                row.append(sentence2)
            
            # generate in-context samples for each label
            for index, (label_token, label) in enumerate(args.verbalizer.items()):
                assert index == label, f'index {index} != label {label}'
                # replace args.label_toke with label token
                label_dependent_input = original_input.replace(args.label_token, label_token)
                # print(len(label_dependent_input))

                # replace args.input_label_token with random pseudo_input_label_token
                # we select a random pseudo input label
                filtered_label2token = args.label2token.copy()
                assert label in filtered_label2token, f'{label} not in {filtered_label2token.keys()}'
                # we remove the input label for selecting pseudo input label
                # we do this to remove the bias while generating
                filtered_label2token.pop(label)
                # generate a random pseudo label for the input sentence
                pseudo_label = random.randint(0, len(filtered_label2token) - 1)
                pseudo_label = list(filtered_label2token.keys())[pseudo_label]
                pseudo_input_label_token = filtered_label2token[pseudo_label]
                # replace
                if args.input_label_token in label_dependent_input:
                    label_dependent_input = label_dependent_input.replace(args.input_label_token, pseudo_input_label_token)

                l = len(label_dependent_input)

                tokenized_inputs = tokenizer(label_dependent_input, return_tensors='pt').to(model_engine.device)
                # shape : (1, input_length) -> (input_length, )
                input_ids = tokenized_inputs['input_ids'].squeeze(dim=0)
                input_length = len(input_ids)

                generated_ids = model_engine.module.generate(
                    **tokenized_inputs,
                    do_sample=True,
                    max_length=input_length+args.generation_max_length,
                    min_length=input_length+args.generation_min_length,
                    temperature=args.temperature,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    num_return_sequences=args.n_samples,
                    early_stopping=True,
                    bad_words_ids=bad_words_ids
                )

                # list of length n_samples
                generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                generated_outputs = [genenerated_output[l:].replace('\n', '').strip() for genenerated_output in generated_outputs]

                # print('=' * 50)
                # for generated_output in generated_outputs:
                #     print(generated_output)
                
                row.append(generated_outputs)
            
            tsv_writer.writerow(row)

    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time} sec.')
                
if __name__ == "__main__":
    logger.info('\nRunning : transformers_generate.py')
    main()