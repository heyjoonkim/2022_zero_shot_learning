import os
import torch
import csv
import pickle
import argparse

from datasets import load_dataset, DatasetDict, Dataset
from sentence_transformers import SentenceTransformer, util

from dataset_utils import task_to_path, task_to_keys, task_to_verbalizer


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

    raw_datasets = DatasetDict()
    if args.task_name is not None and args.benchmark_name is not None:
        if args.benchmark_name == 'huggingface':
            raw_train_dataset = load_dataset(args.task_name, split='train')
            raw_eval_dataset = load_dataset(args.task_name, split='test')
        else:
            # Downloading and loading a dataset from the hub.
            raw_train_dataset = load_dataset(args.benchmark_name, args.task_name, split=f'train')
            # for mnli 
            if args.task_name == "mnli":
                raw_eval_dataset = load_dataset(args.benchmark_name, args.task_name, split='validation_matched')
            else:
                raw_eval_dataset = load_dataset(args.benchmark_name, args.task_name, split=f'validation')
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')

    raw_datasets['train'] = raw_train_dataset
    raw_datasets['validation'] = raw_eval_dataset

    if args.task_name is not None and args.benchmark_name is not None:
        if args.benchmark_name == 'huggingface':
            # TODO : fix?
            label_list = raw_datasets["train"].features["label-coarse"].names
        else:
            # label_list : ['entailment', 'not_entailment']
            label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')

    # lists of sentences per class
    senteces_per_classes = []
    data_per_classes = []
    for i in range(num_labels):
        data_per_class = raw_datasets['train'].filter(lambda example: example['label-coarse'] == i)
        temp = []
        for d in data_per_class:
            temp.append(d['text'])
        senteces_per_classes.append(temp)
        data_per_classes.append(data_per_class)

    model = SentenceTransformer('all-MiniLM-L12-v1').to('cuda')

    embedding_path = os.path.join(args.output_dir, f"{args.task_name}_embedding_per_class.pkl")
    if os.path.exists(embedding_path):
        with open(embedding_path,'rb') as f:
            embedding_per_class = pickle.load(f)
    else:
        embedding_per_class = []
        for sentences in senteces_per_classes:
            embedding_per_class.append(model.encode(sentences))

        with open(embedding_path,'wb') as f:
            pickle.dump(embedding_per_class, f)

    query_path = os.path.join(args.output_dir, f"{args.task_name}_query_embedding.pkl")
    if os.path.exists(query_path):
        with open(query_path,'rb') as f:
            query_embedding = pickle.load(f)
    else:
        eval_sentences = [d['text'] for d in raw_datasets['validation']]
        query_embedding = model.encode(eval_sentences)
        with open(query_path,'wb') as f:
            pickle.dump(query_embedding, f)

    result_path = os.path.join(args.output_dir, f"{args.task_name}_retrieve.pkl")
    if os.path.exists(result_path):
        with open(result_path,'rb') as f:
            results = pickle.load(f)
    else:
        results = []
        for embeddings in embedding_per_class:
            sim = util.dot_score(query_embedding, embeddings)
            # torch.topk(sim,4,dim=1).indices
            results.append(torch.argmax(sim,dim=1))
        results = torch.stack(results, dim=1)
        with open(result_path,'wb') as f:
            pickle.dump(results, f)

    samples = range(10)
    for sample in samples:
        print(f"eval data: {raw_datasets['validation'][sample]}")
        print('retrieve results')
        for c, idx in enumerate(results[sample].tolist()):
            print(data_per_classes[c][idx])
        print()


# with open('sst2_sim.tsv', "w") as file_writer:
#     tsv_writer = csv.writer(file_writer, delimiter='\t')
#     for idx in torch.topk(sim,4,dim=1).indices:
#         temp = []
#         for topk in idx:
#             temp.append(s_list[topk])
#         tsv_writer.writerow(temp)

if __name__ == "__main__":
    main()
