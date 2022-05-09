import os
import logging
import torch
import csv
import pickle
import argparse
import random
import time
import math

from datasets import load_dataset, DatasetDict, Dataset
from transformers import set_seed
from sentence_transformers import SentenceTransformer, util

from dataset_utils import task_to_keys, task_to_path

logger = logging.getLogger(__name__)

def parse_args():

    parser = argparse.ArgumentParser(description="Create retriever db")

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
        "--train_task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_benchmark_name",
        type=str,
        default=None,
        help="The name of the benchmark to train on.",
        choices=['glue', 'super_glue', 'huggingface'],
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="all-MiniLM-L12-v1",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final data."
    )
    parser.add_argument(
        '--overwrite_output_dir', 
        default=False, 
        action="store_true",
        help='Overwrite output directory.'
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=0, 
        help="Number of samples per class."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        '--random_label', 
        default=False, 
        action="store_true",
        help='Use random labels for in-context samples.'
    )

    args = parser.parse_args()
    
    return args

def main():

    args = parse_args()

    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.overwrite_output_dir:
                raise NotADirectoryError(f'Output directory {args.output_dir} exits. Exit program. (overwrite_output_dir=False)')

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

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

    if args.train_task_name is not None and args.train_benchmark_name is not None:
        if args.train_benchmark_name == 'huggingface':
            raw_train_dataset = load_dataset(args.train_task_name, split='train')
        else:
            # Downloading and loading a dataset from the hub.
            raw_train_dataset = load_dataset(args.train_benchmark_name, args.train_task_name, split=f'train')
    # for datasets from file.
    elif args.train_task_name in task_to_path:
        dataset_processor = task_to_path[args.train_task_name]["dataset_processor"]
        train_file_path = task_to_path[args.train_task_name]["train"]
        # train st
        train_dict = dataset_processor(train_file_path)
        raw_train_dataset = Dataset.from_dict(train_dict)
    else:
        raise NotImplementedError(f'{args.train_task_name} task is not implemented yet.')

    raw_datasets['train'] = raw_train_dataset
    raw_datasets['validation'] = raw_eval_dataset

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

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # load model
    model = SentenceTransformer(args.model_name_or_path).to('cuda')

    train_embedding_path = os.path.join(args.output_dir, f"{args.task_name}_train_embedding_top-{args.n_samples}.pkl")
    # save embeddings of each sentences grouped by specific labels.
    # we will use this embedding to select the best matched in-context samples.
    # if os.path.exists(train_embedding_path):
    #     with open(train_embedding_path,'rb') as f:
    #         embedding_per_class = torch.tensor(pickle.load(f)).to('cuda')
    # else:
    # lists of sentences per class
    senteces_per_classes = []
    data_per_classes = []
    # for each labels
    for i in range(num_labels):
        # dataset with the label index i
        if 'label-coarse' in raw_datasets['train']:
            data_per_class = raw_datasets['train'].filter(lambda example: example['label-coarse'] == i)
        else:
            data_per_class = raw_datasets['train'].filter(lambda example: example['label'] == i)
        temp = []
        for d in data_per_class:
            temp.append(d[sentence1_key])

        # append list of sentences with the same label
        # (num_labels, num_sentences_per_label)
        senteces_per_classes.append(temp)
        # append list of datasets with the same label
        data_per_classes.append(data_per_class)

    # (num_labels, num_sentences_per_label, embedding_dim)
    embedding_per_class = []
    for sentences in senteces_per_classes:
        embedding_per_class.append(model.encode(sentences))

    with open(train_embedding_path,'wb') as f:
        pickle.dump(embedding_per_class, f)




    # save embeddings for test data
    test_embedding_path = os.path.join(args.output_dir, f"{args.task_name}_test_embedding_top-{args.n_samples}.pkl")
    # if os.path.exists(test_embedding_path):
    #     with open(test_embedding_path,'rb') as f:
    #         test_embedding = torch.tensor(pickle.load(f)).to('cuda')
    # else:
    test_sentences = [d[sentence1_key] for d in raw_datasets['validation']]
    # query_embedding : (len(test_sentences), embedding_dim)
    test_embedding = model.encode(test_sentences)
    with open(test_embedding_path,'wb') as f:
        pickle.dump(test_embedding, f)

    logger.info(f'TEST  : {test_embedding.shape}') #, {test_embedding.device}')
    for label_index, train_embedding in enumerate(embedding_per_class):
        logger.info(f'TRAIN label {label_index} : {train_embedding.shape}') #, {train_embedding.device}')


    samples_per_class = math.ceil(args.n_samples / num_labels)
    logger.info(f'samples_per_class : {samples_per_class}')
    # list of top-k indices for each class
    # (num_labels, test_samples_num, n_samples_per_class)
    results = []

    # calculate top-k similar samples
    result_path = os.path.join(args.output_dir, f"{args.task_name}_topk_indices_top-{args.n_samples}.pkl")
    # if os.path.exists(result_path):
    #     with open(result_path,'rb') as f:
    #         results = pickle.load(f)
    # else:
    # use dot score (cosine similarity) for calculating the similarity
    # shape : (test_sample_num, validation_sample_num)

    # for range(num_labels)
    # train_embedding : (num_sentences_per_label, embedding_dim)
    for i, train_embedding in enumerate(embedding_per_class):
        # (test_samples_num, num_sentences_per_label)
        sim = util.dot_score(test_embedding, train_embedding)
        
        topk = torch.topk(sim, k=samples_per_class, dim=1)
        # shape : (test_samples_num, n_samples_per_class)
        topk_values, topk_indices = topk.values.tolist(), topk.indices.tolist()

        results.append(topk_indices)

    with open(result_path,'wb') as f:
        pickle.dump(results, f)


    generation_writer = os.path.join(args.output_dir, "test.tsv")

    # prevent from overwriting generated dataset
    # if os.path.isfile(generation_writer):
    #     logger.info('Generated dataset already exists. Exit Program.')
    #     exit()

    # for analysis
    total_same_label_count = 0
    total_correct_label_count = 0
    total_sample_count = 0


    with open(generation_writer, 'w') as file_writer:
        tsv_writer = csv.writer(file_writer, delimiter='\t')

        test_indices = range(len(raw_datasets['validation']))
        for test_index in test_indices:
            # logger.info(f"eval data: {raw_datasets['validation'][test_index]}")
            if 'label-coarse' in raw_datasets['validation'][test_index]:
                gold_label = raw_datasets['validation'][test_index]['label-coarse']
            else:
                gold_label = raw_datasets['validation'][test_index]['label']
            sentence1 = raw_datasets['validation'][test_index][sentence1_key]

            row = [test_index, gold_label, sentence1]
            if sentence2_key is not None:
                sentence2 = raw_datasets['validation'][test_index][sentence2_key]
                row.append(sentence2)

            sample_list_per_label = []
            
            correct_label_count = 0
            same_label_count = 0

            # (num_labels, test_samples_num, n_samples_per_class)
            # list of length num_labels
            # indices_per_label : (test_samples_num, n_samples_per_class)
            for label_index, indices_per_label in enumerate(results):
                assert label_index == len(sample_list_per_label), f'{label_index} != {len(sample_list_per_label)}'
                topk_samples_per_class = []
                # shape : (n_samples_per_class, )
                dataset_indices = indices_per_label[test_index]

                for dataset_index in dataset_indices:
                    # retrieved sample
                    if 'label-coarse' in data_per_classes[label_index][dataset_index]:
                        retrieved_label = data_per_classes[label_index][dataset_index]['label-coarse']
                    else:
                        retrieved_label = data_per_classes[label_index][dataset_index]['label']
                    
                    if args.random_label:
                        # TODO : remove?
                        # testing for the effect of random labeling
                        pseudo_label = random.randint(0, num_labels-1)
                        if retrieved_label == pseudo_label:
                            correct_label_count += 1
                        retrieved_label = pseudo_label
                        # TODO : remove until here
                    
                    # for analysis
                    if retrieved_label == gold_label:
                        same_label_count += 1

                    retrieved_sentence = data_per_classes[label_index][dataset_index][sentence1_key]
                    topk_samples_per_class.append(retrieved_sentence)

                sample_list_per_label.append(topk_samples_per_class)

            row = row + sample_list_per_label

            # for analysis
            # logger.info(f'{same_label_count} / {args.n_samples} = {same_label_count / args.n_samples * 100}\n')
            # logger.info(f'{correct_label_count} / {args.n_samples} = {correct_label_count / args.n_samples * 100}\n')
            
            total_same_label_count += same_label_count
            total_correct_label_count += correct_label_count
            total_sample_count += (num_labels * samples_per_class)

            tsv_writer.writerow(row)
    
    # for analysis
    logger.info(f'FINAL gold-retrieved label alignment ratio : {total_same_label_count} / {total_sample_count} = {total_same_label_count / total_sample_count * 100}')
    logger.info(f'FINAL random label alignment ratio : {total_correct_label_count} / {total_sample_count} = {total_correct_label_count / total_sample_count * 100}')

if __name__ == "__main__":
    logger.info('\nRunning : transformers_retrieval.py')
    
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')