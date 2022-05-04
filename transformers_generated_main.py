
import argparse
import logging
import os
import random
import json
import time

import datasets
from datasets import load_metric, DatasetDict, Dataset
from tqdm.auto import tqdm

import transformers
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)
import torch
import deepspeed

from model_wrapper.TransformersModelWrapper import GPT2Wrapper
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
        choices=list(generated_task_to_path.keys()),
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
    parser.add_argument(
        '--explicit_label_space', 
        default=False, 
        action="store_true",
        help='Explicitly show label space.'
    )

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

    if args.local_rank == 0:
        logger.info('TRAIN / VALIDATION split.')
        for split, dataset in raw_datasets.items():
            logger.info(f'{split} > {len(dataset)}')
    
    # Labels
    if args.task_name in generated_task_to_path:
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

    model_loading_start = time.time()
    model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path, verbalizer=args.verbalizer)
    model_loading_end = time.time()
    logger.info(f'Total time for loading model : {model_loading_end - model_loading_start}')

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # load generated in-context samples
    # full_train_samples_list : all in-context samples -> for random sampling
    # label2samples_list      : in-context samples for each label -> for balanced sampling
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
        desc="Preprocessing datasets...",
    )

    eval_dataset = processed_datasets["validation"]

    # Get the metric function  
    if args.benchmark_name == 'huggingface':
        metric = load_metric("accuracy", num_process=args.world_size, process_id=args.local_rank)
    else:
        metric = load_metric(args.benchmark_name, args.task_name, num_process=args.world_size, process_id=args.local_rank)
    
    # deepspeed initialize
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=None, lr_scheduler=None, config_params=args.ds_config)

    # Evaluate! 
    if args.local_rank == 0:
        logger.info("***** Zero/Few-shot Evaluation *****")
        logger.info(f"  Num EVAL  examples = {len(eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
        logger.info(f"  World Size = {args.world_size}")
        logger.info(f"  Random Seed = {args.seed}")
        logger.info(f"  K = {args.n_samples}")
        logger.info(f"  Inference Model = {args.model_name_or_path}")
         
    # for analysis
    prediction_dict = {}

    start_time = time.time()
    model_engine.eval()

    # prepend prompt to explicitly show label space 
    if args.explicit_label_space:
        labels = list(args.verbalizer.keys())
        labels = ' '.join(labels)
        label_space = 'Types:' + labels

    # evaluate
    for step, inputs in tqdm(enumerate(eval_dataset), disable=(args.local_rank != 0)):

        # in-context samples generated conditioned by the input x.
        if args.n_samples > 0:
            incontext_samples, sep = prepend_incontext_samples(
                label2samples=label2samples_list[step],
                full_train_samples=full_train_samples_list[step],
                k=args.n_samples,
                balance_sample=args.balance_sample,
            )
            # prepend in-context samples
            inputs['input_sentence'] = incontext_samples + sep + inputs['input_sentence']
        
        # show all label space tokens (NOT USED FOR NOW)
        if args.explicit_label_space:
            inputs['input_sentence'] = label_space + sep + inputs['input_sentence']

        label = torch.tensor(inputs['labels']).to(model_engine.device).unsqueeze(dim=0)

        # logging first sample
        if step == 0:
            logger.info('LOGGING FIRST GENERATED SAMPLE.')
            logger.info(f'LABEL : {label}')
            logger.info(f'INPUT SENTENCE : {inputs["input_sentence"]}')

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
    logger.info('\nRunning : transformers_generated_main.py')
    
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')