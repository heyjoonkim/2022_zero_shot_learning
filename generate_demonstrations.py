import argparse
import logging
import os
import random
import pickle
import time

from datasets import load_dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    set_seed,
)
from utils import save_config
from dataset_utils import task_to_path, task_to_keys, task_to_verbalizer, no_validation_tasks

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
        choices=['glue', 'super_glue', 'huggingface', 'tweet_eval', 'financial_phrasebank', 'ethos'],
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
    args.label2token = {v:k for k,v in args.verbalizer.items()}

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
        split = 'test' if args.task_name == 'climate_fever' else 'train'
        if args.benchmark_name == 'huggingface':
            raw_train_dataset = load_dataset(args.task_name, split=split)
        # elif args.benchmark_name == 'tweet_eval':
        #     raw_train_dataset = load_dataset(args.benchmark_name, args.task_name, split='train')
        elif args.benchmark_name == 'ethos':
            raw_train_dataset = load_dataset(args.benchmark_name, 'multilabel', split=split)
        else:
            # Downloading and loading a dataset from the hub.
            raw_train_dataset = load_dataset(args.benchmark_name, args.task_name, split=split)
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')


    # we have to split the data into train/validation set 
    if args.task_name in no_validation_tasks:
        with open('validation_indices.pkl', 'rb') as fp:
            validation_indices = pickle.load(fp)
            task_key = args.task_name
            if task_key == 'sentences_allagree':
                task_key = 'financial_phrasebank'
            elif args.benchmark_name == 'ethos':
                task_key = args.benchmark_name
            selected_validation_indices = validation_indices[task_key]
            print('validation', len(selected_validation_indices))
            full_indices = list(range(len(raw_train_dataset)))
            print('full', len(full_indices))
            train_indices = list(set(full_indices) - set(selected_validation_indices))
            print('train', len(train_indices))
            raw_train_dataset = raw_train_dataset.select(train_indices)

    raw_datasets['train'] = raw_train_dataset
    
    num_labels = len(args.verbalizer)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # For gpt-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    train_dataset = raw_datasets["train"]

    # train_dataset = train_dataset.filter(lambda example: example['labels'] in args.verbalizer.values())

    # Evaluate! 
    logger.info("***** Generate few-shot *****")
    logger.info(f"  BENCHMARK              = {args.benchmark_name}")
    logger.info(f"  TASK                   = {args.task_name}")
    logger.info(f"  Num TRAIN examples     = {len(train_dataset)}")
    logger.info(f"  Random Seed            = {args.seed}")
    logger.info(f"  K                      = {args.n_samples}")
    logger.info(f"  Balanced Demonstration = {args.balance_sample}")


    selected_index = []
    selected_count = 0
    sample_balance = dict()

    # select balanced samples from train set
    if args.balance_sample:
        samples_per_class = []
        class_indices = [class_index for class_index in range(num_labels)]
        for class_index in class_indices:
            assert len(samples_per_class) == class_index, f'{len(samples_per_class)} != {class_index}'
            
            samples = [(index, sample) for index, sample in enumerate(raw_datasets['train']) if sample['label'] == class_index]
            # samples = raw_datasets['train'].filter(lambda example: example['label'] == class_index)
            samples_per_class.append(samples)
        # shuffle class indices so that we don't always select samples with label 0
        random.shuffle(class_indices)
        
        while True:
            for class_index in class_indices:
                # print('select class with ', class_index)
                class_dataset = samples_per_class[class_index]
                data_count = len(class_dataset)
                # print('class', class_index, 'with', data_count, 'samples.')
                # select a sample with a specific label
                while True:
                    random_index = random.randint(0, data_count-1)
                    # print('random_index :', random_index)
                    # print(class_dataset[random_index])

                    dataset_index, sample = class_dataset[random_index]
                    selected_label = sample['label']
                    if dataset_index not in selected_index and selected_label in list(args.verbalizer.values()):
                        selected_index.append(dataset_index)
                        selected_count += 1

                        sample_balance[selected_label] = sample_balance.get(selected_label, 0) + 1
                        break

                if selected_count == args.n_samples:
                    break
            if selected_count == args.n_samples:
                break            
    
    # naive random sampling from train set
    else:
        data_count = len(train_dataset)
        while True:
            random_index = random.randint(0, data_count-1)


            if 'label-coarse' in train_dataset[random_index]:
                selected_label = train_dataset[random_index]['label-coarse']
            elif 'claim_label' in train_dataset[random_index]:
                selected_label = train_dataset[random_index]['claim_label']
            elif args.benchmark_name == 'ethos':
                selected_label = train_dataset[random_index][args.task_name]
            else:
                selected_label = train_dataset[random_index]['label']

            if random_index not in selected_index and selected_label in list(args.verbalizer.values()):
                selected_index.append(random_index)
                selected_count += 1

                sample_balance[selected_label] = sample_balance.get(selected_label, 0) + 1

            if selected_count == args.n_samples:
                break

    logger.info(f'Selected dataset indices : {selected_index}')
    logger.info(f'Selected balance         : {sample_balance}')

    output_file = os.path.join(args.output_dir, 'demonstration_indices.pkl')
    with open(output_file,'wb') as f:
        pickle.dump(selected_index, f)
        

    # for sanity check
    logger.info('Loading saved files...')
    if os.path.exists(output_file):
        with open(output_file,'rb') as f:
            selected_index = pickle.load(f)
            assert len(selected_index) == args.n_samples, f'{len(selected_index)} != {args.n_samples}'
            logger.info(f'Selected indices : {selected_index}')

    logger.info("Done.")
                
if __name__ == "__main__":
    logger.info('\nRunning : transformers_main.py')
    
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')