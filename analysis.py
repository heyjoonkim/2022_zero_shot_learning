import argparse
import logging
import sys
import time

from tqdm.auto import tqdm
from datasets import load_dataset

from transformers import AutoTokenizer

from dataset_utils import task_to_keys

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

    # Downloading and loading a dataset from the hub.
    if args.benchmark_name in ['glue', 'super_glue']:
        datasets = load_dataset(args.benchmark_name, args.task_name)
    else:
        datasets = load_dataset(args.task_name)
    

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
        remove_columns=datasets["train"].column_names,
        desc="Preparing dataset",
    )

    train_dataset, eval_dataset, test_dataset = None, None, None

    if "train" not in processed_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = processed_datasets["train"]
    logger.info(f'# TRAIN dataset   : {len(train_dataset)}')

    if 'validation' in processed_datasets:
        eval_dataset = processed_datasets["validation"]
        logger.info(f'# Eval  dataset   : {len(eval_dataset)}')
    if 'test' in processed_datasets:
        test_dataset = processed_datasets["test"]   
        logger.info(f'# Eval-mm dataset : {len(test_dataset)}')
        
    

    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    logger.info('Done loading tokenizer.')

    start_time = time.time()


    def count_tokens(dataset):
        total_input_sentence_length = 0
        total_sentence1_length = 0
        if sentence2_key is not None:
            total_sentence2_length = 0

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

            # sentence1 length
            input_sentence1_token_length = len(tokenizer(sentence1)['input_ids'])
            total_sentence1_length += input_sentence1_token_length
            
            # sentence2 length (if any)
            if sentence2_key is not None:
                input_sentence2_token_length = len(tokenizer(sentence2)['input_ids'])
                total_sentence2_length += input_sentence2_token_length

            label2count[labels] = label2count.get(labels, 0) + 1 

        
        average_input_sentence_token_length = total_input_sentence_length / len(dataset)
        logger.info(f'Average input token length : {average_input_sentence_token_length}')
        average_sentence1_token_length = total_sentence1_length / len(dataset)
        logger.info(f'Average sentence1 token length : {average_sentence1_token_length}')
        if sentence2_key is not None:
            average_sentence2_token_length = total_sentence2_length / len(dataset)
            logger.info(f'Average sentence2 token length : {average_sentence2_token_length}')
        logger.info(f'label split : {label2count}')

    if train_dataset:
        logger.info('TRAIN   ============')
        count_tokens(train_dataset)
    if eval_dataset:
        logger.info('EVAL    ============')
        count_tokens(eval_dataset)
    if test_dataset:
        logger.info('TEST    ============')
        count_tokens(test_dataset)

    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time}')

if __name__ == "__main__":
    print('start')
    main()
