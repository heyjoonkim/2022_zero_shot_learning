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
import datetime
import logging
import os
import math
import sys
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric, concatenate_datasets

from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from model_wrapper.ModelWrapper import ModelWrapper

from utils import save_config



task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

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
        "--max_train_samples",
        default=None,
        help="Maximum train samples to use at train time, slice from raw train dataset for fast experiment purpose",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=16,
        help=(
            "Batch size per each device."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=20, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--early_stop", 
        type=int, 
        default=5, 
        help="Number of epoch for early stopping."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.6, 
        help="Ratio of warmup steps from total train steps."
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
        '--save_threshold', 
        default=0, 
        type=int, 
        help='Number of prompt tokens.'
    )
    parser.add_argument(
        '--fine_tune', 
        default=False, 
        action="store_true",
        help='Fine-tune PLM.'
    )

    ## FOR PROMPT LEARNING METHODS ##
    parser.add_argument(
        '--apply_prompt', 
        default=False, 
        action="store_true",
        help='Apply my method for prompting.'
    )
    parser.add_argument(
        '--prompt_length', 
        default=5, 
        type=int, 
        help='Number of prompt tokens.'
    )
    parser.add_argument(
        '--plm_layer', 
        default=-1, 
        type=int, 
        help='The layer index of the PLM to copy the weights.'
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        default="cls",
        help="Pooling method for prediction.",
        choices=["cls", "mean"],
    )

    args = parser.parse_args()
    
    return args
    

def main():

    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.ERROR)


    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            if accelerator.is_main_process:
                os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.overwrite_output_dir:
                logger.info(f'Output directory {args.output_dir} exits. Exit program. (overwrite_output_dir=False)')
                exit()

    if accelerator.is_main_process:
        logging_output_file = os.path.join(args.output_dir, "output.log")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler = logging.FileHandler(logging_output_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if accelerator.is_main_process:
        save_config(args)
        writer = SummaryWriter(args.output_dir)



    # Set seed before initializing model.
    set_seed(args.seed)

    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", args.task_name)
    else:
        logger.info('Task name is required. Got None.')
        exit()

    # Labels
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
   
    # load pretrained model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # encoder models already have pad_tokens
    # tokenizer.pad_token = tokenizer.unk_token
    
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        pad_token_id=tokenizer.pad_token_id,
    )

    # XXX: for out method
    config.apply_prompt = args.apply_prompt
    config.prompt_length = args.prompt_length
    config.plm_layer = args.plm_layer
    config.pooling_method = args.pooling_method
    # until here

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # TODO : fix?
    model = ModelWrapper(config=config, model_name_or_path=args.model_name_or_path)
    
    for name, param in model.named_parameters():
        # FREEZE ONLY THE BASE MODEL. FOR GPT2.
        if name.startswith('transformer') or 'bert' in name:
            if args.fine_tune:
                param.requires_grad = True
                logger.info(f'TRAINED PARM : {name} -> {param.shape}')
            else:
                param.requires_grad = False
        else:
            logger.info(f'TRAINED PARM : {name} -> {param.shape}')
            param.requires_grad = True
    
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    transformer_params = sum(p.numel() for n,p in model.named_parameters() if n.startswith('transformer'))
    num_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'trainable params {num_trainable_params} / total params {num_total_params} => ratio : {100 * num_trainable_params / num_total_params}')
    
    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    
    # Padding strategy
    if args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None:
        logger.info('Auto label2id, id2label created')
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}


    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)
        if config.apply_prompt:
            prompt_tokens = [tokenizer.unk_token_id for _ in range(config.prompt_length)]
            prompt_attention_mask = [1 for _ in range(config.prompt_length)]
            result['input_ids'] = [ids[:1] + prompt_tokens + ids[1:] for ids in result['input_ids']]
            result['attention_mask'] =  [mask+prompt_attention_mask for mask in result['attention_mask']]
            
        # Map labels to IDs (not necessary for GLUE tasks)
        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    if "train" not in processed_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = processed_datasets["train"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    if "validation" not in processed_datasets and "validation_matched" not in processed_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    if args.task_name == "mnli":
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataset_mm = processed_datasets["validation_matched"]
    else:
        eval_dataset = processed_datasets["validation"]

    # for GLUE, we don't have the labels for the test split.
    # We only use the results for the best dev. model.
    # test datasets will not be used! (for now)
    if "test" not in processed_datasets and "test_matched" not in processed_datasets:
        raise ValueError("--do_predict requires a test dataset")
    if args.task_name == "mnli":
        test_dataset = concatenate_datasets([processed_datasets["test_mismatched"], processed_datasets["test_matched"]])
    else:
        test_dataset = processed_datasets["test"]
    if "label" not in test_dataset:
        logger.info('No labels for test split.')


    logger.info(f'# TRAIN dataset : {len(train_dataset)}')
    logger.info(f'# Eval  dataset : {len(eval_dataset)}')
    logger.info(f'# TEST  dataset : {len(test_dataset)}')
    ## DONE LOADING DATASET ##

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    if args.task_name == "mnli":
        eval_dataloader_mm = DataLoader(eval_dataset_mm, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    # NOT USED
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_batch_size)


    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad==True],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad==True],
            "weight_decay": 0.0,
        },
    ]

    if accelerator.is_main_process:
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in model.parameters())
        transformer_params = sum(p.numel() for n,p in model.named_parameters() if n.startswith('transformer') or 'bert' in n)
        logger.info(f'trainable params {num_trainable_params} / total params {num_total_params} = ratio {100 * num_trainable_params/num_total_params} ')
        logger.info(f'trainable params {num_trainable_params} / PLM params {transformer_params} = ratio {100 * num_trainable_params/transformer_params} ')
        
        ## Write parameter info ##
        parameter_summary_file = os.path.join(args.output_dir, "parameter_summary.txt")
        with open(parameter_summary_file, "w") as file_writer:
            file_writer.write("Overall Parameter Summary\n")
            file_writer.write(f"Trained     parameters\t{num_trainable_params}\n")
            file_writer.write(f"Transformer parameters\t{transformer_params}\n")
            file_writer.write(f"Total       parameters\t{num_total_params}\n")
            file_writer.write(f"Trainable   ratio\t\t{100 * num_trainable_params / num_total_params} \n")
            file_writer.write(f"PLM         ratio\t\t{100 * num_trainable_params / transformer_params} \n")
            file_writer.write("=" * 50 + '\n')
            file_writer.write("Trained parameters detail\n")

            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    file_writer.write(f"{name} > {param.shape} > {param.numel()} \n")

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    if args.task_name == "mnli":
        eval_dataloader_mm = accelerator.prepare(eval_dataloader_mm)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    

    num_warmup_steps = round(max_train_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
        
    total_batch_size = args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Total warmup steps = {num_warmup_steps}")
    logger.info(f"  Output directory = {args.output_dir}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_acc = 0
    best_step = 0
    best_epoch = 0
    early_stop_cnt = 0

    start_time = time.time()
    for epoch in range(args.num_train_epochs):
        if early_stop_cnt >= args.early_stop:
            logger.info("EARLY STOP. STOP TRAINING.")
            break

        model.train()
        for step, batch in enumerate(train_dataloader):
            loss, _ = model(**batch)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if accelerator.is_main_process:
                writer.add_scalar('Train/Loss', loss, completed_steps)
                writer.add_scalar('Train/LR', lr_scheduler.get_lr()[0], completed_steps)
                if args.apply_prompt:
                    writer.add_scalar('Prompt Embedding', model.module.input_processor.prompt_embeddings.weight[0,0], completed_steps)
                
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # model step manages optimizer
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break


        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                loss, predictions = model(**batch)

                all_predictions = accelerator.gather(predictions)
                all_targets = accelerator.gather(batch["labels"])
                
                metric.add_batch(
                    predictions=all_predictions,
                    references=all_targets,
                )
        eval_metric = metric.compute()
        if accelerator.is_main_process:
            writer.add_scalar('Validation/Accuracy Step', eval_metric['accuracy'], completed_steps)
            writer.add_scalar('Validation/Accuracy Epoch', eval_metric['accuracy'], epoch+1)
            if "f1" in eval_metric.keys():
                writer.add_scalar('Validation/F1 Step', eval_metric['f1'], completed_steps)
                writer.add_scalar('Validation/F1 Epoch', eval_metric['f1'], epoch+1)

        logger.info(f"Valditaion step {completed_steps} results {eval_metric}")
        if eval_metric['accuracy'] > best_acc:
            best_epoch = epoch
            best_step = completed_steps
            # TODO : save only the models greater than the threshold accuracy
            best_acc = eval_metric['accuracy']
            if best_acc > args.save_threshold:
                save_flag = True      
            else:
                save_flag = False      
        else:
            save_flag = False
        

        if save_flag:
            early_stop_cnt = 0
            
            # if accelerator.is_main_process:
            #     logger.info('REMOVING PAST STATE DICTS...')
            #     file_lists = os.listdir(args.output_dir)
            #     for file in file_lists:
            #         if file.endswith('.pth'):
            #             removed_file = os.path.join(args.output_dir, file)
            #             os.remove(removed_file)
            #             logger.info(f'Removed file : {removed_file}')
            # logger.info('SAVING MODEL....')
            # output_file = os.path.join(args.output_dir, f'{completed_steps}.pth')
            # accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # accelerator.save(unwrapped_model.state_dict(), output_file)
            # logger.info(f'Saved model : {output_file}')

            # TO LOAD MODEL
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.load_state_dict(torch.load(output_file))
        else:
            early_stop_cnt += 1

        logger.info(f'EARLY STOP COUNT : {early_stop_cnt} / {args.early_stop}')

        if args.task_name == "mnli":
            for step, batch in enumerate(eval_dataloader_mm):
                with torch.no_grad():
                    loss, predictions = model(**batch)


                    all_predictions = accelerator.gather(predictions)
                    all_targets = accelerator.gather(batch["labels"])
                    
                    metric.add_batch(
                        predictions=all_predictions,
                        references=all_targets,
                    )
            eval_metric = metric.compute()
            if accelerator.is_main_process:
                writer.add_scalar('Validation-mm/Accuracy Step', eval_metric['accuracy'], completed_steps)
                writer.add_scalar('Validation-mm/Accuracy Epoch', eval_metric['accuracy'], epoch+1)

    total_time = time.time() - start_time
    logger.info(f'TOTAL TRAIN TIME : {str(datetime.timedelta(seconds=total_time))}')

    logger.info(f'BEST DEV AT EPOCH {best_epoch} (step : {best_step}) : {best_acc}')

    if accelerator.is_main_process:
        writer.add_scalar('Best Validation', best_acc, 0)

    # best_model_dir = os.path.join(args.output_dir, f'{best_step}.pth')
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.load_state_dict(torch.load(best_model_dir))

    # FINAL EVALUATION
    # model.eval()
    # for step, batch in enumerate(eval_dataloader):
    #     with torch.no_grad():
    #         loss, predictions = model(**batch)

    #         all_predictions = accelerator.gather(predictions)
    #         all_targets = accelerator.gather(batch["labels"])
            
    #         metric.add_batch(
    #             predictions=all_predictions,
    #             references=all_targets,
    #         )
    # eval_metric = metric.compute()
    # logger.info(f'BEST DEV RESULTS at step {best_step} : {eval_metric["accuracy"]}')

if __name__ == "__main__":
    main()
