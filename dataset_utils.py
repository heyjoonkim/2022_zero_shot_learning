
#
# Utils for loading datasets from file (csv, tsv, ...).
# otherwise we use load_dataset() from huggingface library.
#

import csv
import random
from select import select

def custom_generate_dataset_dict(filename):
    input_list = []
    label_list = []
    with open(filename) as f:
        validation_lines = csv.reader(f, delimiter='\t')
        # Remove header
        next(validation_lines, None)

        for validation_line in validation_lines:
            sample_index = validation_line[0]
            label = int(validation_line[1])
            input_sentence = validation_line[2]
            generation1 = validation_line[3]
            generation2 = validation_line[4]
            generation3 = validation_line[5]

            generation = '.'.join([generation1, generation2, generation3])

            input_sentence = generation + '.' + input_sentence

            label_list.append(label)
            input_list.append(input_sentence)
            
    return_dict = {
        'sentence' : input_list,
        'label' : label_list
    }

    return return_dict

# for SST-5
def sst5_generate_dataset_dict(filename):
    input_list = []
    label_list = []
    with open(filename) as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            comma_index = line.index(',')
            label = int(line[:comma_index])
            input_sentence = line[comma_index+1:]
            
            if input_sentence.startswith('"') and input_sentence.endswith('"'):
                input_sentence = input_sentence.replace('"', '')

            label_list.append(label)
            input_list.append(input_sentence)
    return_dict = {
        'sentence' : input_list,
        'label' : label_list
    }

    return return_dict

# for MR
def mr_generate_dataset_dict(filename):
    # same csv file format as SST-5
    return sst5_generate_dataset_dict(filename)

# for CR
def cr_generate_dataset_dict(filename):
    # same csv file format as SST-5
    return sst5_generate_dataset_dict(filename)

# for MPQA
def mpqa_generate_dataset_dict(filename):
    # same csv file format as SST-5
    return sst5_generate_dataset_dict(filename)

# for Subj
def subj_generate_dataset_dict(filename):
    # same csv file format as SST-5
    return sst5_generate_dataset_dict(filename)

# for TREC
def trec_generate_dataset_dict(filename):
    # same csv file format as SST-5
    return sst5_generate_dataset_dict(filename)

# for AG News
def agnews_generate_dataset_dict(filename):
    sentence1_list = []
    sentence2_list = []
    label_list = []
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for line_index, line in enumerate(csv_reader):
            
            assert len(line) == 3, f'LINE {line_index} > GOT {line}'

            label = line[0]
            label = int(label) - 1
            sentence1 = line[1]
            sentence1 = sentence1.strip()
            sentence2 = line[2]
            sentence2 = sentence2.strip()

            label_list.append(label)
            sentence1_list.append(sentence1)
            sentence2_list.append(sentence2)
    return_dict = {
        'sentence1' : sentence1_list,
        'sentence2' : sentence2_list,
        'label' : label_list
    }

    return return_dict

# for Yahoo Answers
def yahoo_generate_dataset_dict(filename):
    sentence1_list = []
    sentence2_list = []
    label_list = []
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for line_index, line in enumerate(csv_reader):


            label = line[0]
            label = int(label) - 1

            sentences1 = line[:-1]
            sentence1 = ' '.join(sentences1)
            sentence1 = sentence1.strip()

            sentence2 = line[-1]
            sentence2 = sentence2.strip()

            label_list.append(label)
            sentence1_list.append(sentence1)
            sentence2_list.append(sentence2)
    return_dict = {
        'sentence1' : sentence1_list,
        'sentence2' : sentence2_list,
        'label' : label_list
    }

    return return_dict

# for Yelp Reviews
def yelp_generate_dataset_dict(filename):
    sentence1_list = []
    label_list = []
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for line_index, line in enumerate(csv_reader):
            
            assert len(line) == 2

            label = line[0]
            label = int(label) - 1
            sentence1 = line[1]
            sentence1 = sentence1.strip()

            label_list.append(label)
            sentence1_list.append(sentence1)
    return_dict = {
        'sentence' : sentence1_list,
        'label' : label_list
    }

    return return_dict

task_to_path = {
    "sst5" : {
        "train" : "/home/heyjoonkim/data/datasets/sst5/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/sst5/test.csv",
        "dataset_processor" : sst5_generate_dataset_dict,
    },
    "mr" : {
        "train" : "/home/heyjoonkim/data/datasets/mr/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/mr/test.csv",
        "dataset_processor" : mr_generate_dataset_dict,
    },
    "cr" : {
        "train" : "/home/heyjoonkim/data/datasets/cr/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/cr/test.csv",
        "dataset_processor" : cr_generate_dataset_dict,
    },
    "mpqa" : {
        "train" : "/home/heyjoonkim/data/datasets/mpqa/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/mpqa/test.csv",
        "dataset_processor" : mpqa_generate_dataset_dict,
    },
    "subj" : {
        "train" : "/home/heyjoonkim/data/datasets/subj/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/subj/test.csv",
        "dataset_processor" : subj_generate_dataset_dict,
    },
    "trec" : {
        "train" : "/home/heyjoonkim/data/datasets/trec/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/trec/test.csv",
        "dataset_processor" : trec_generate_dataset_dict,
    },
    "agnews" : {
        "train" : "/home/heyjoonkim/data/datasets/agnews/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/agnews/test.csv",
        "dataset_processor" : agnews_generate_dataset_dict,
    },
    "yahoo" : {
        "train" : "/home/heyjoonkim/data/datasets/yahoo_answers/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/yahoo_answers/test.csv",
        "dataset_processor" : yahoo_generate_dataset_dict,
    },
    "yelp" : {
        "train" : "/home/heyjoonkim/data/datasets/yelp_review/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/yelp_review/test.csv",
        "dataset_processor" : yelp_generate_dataset_dict,
    },
}

task_to_keys = {
    # GLUE
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    # SuperGLUE
    "boolq" : ("question", "passage"),
    "cb" : ("premise", "hypothesis"),
    # others
    "sst5": ("sentence", None),
    "mr": ("sentence", None),
    "cr": ("sentence", None),
    "mpqa": ("sentence", None),
    "subj": ("sentence", None),
    "trec": ("sentence", None),
    "agnews": ("sentence1", "sentence2"),
    "yahoo": ("sentence1", "sentence2"),
    "yelp": ("sentence", None),
}

task_to_verbalizer = {
    "cola": None,
    "mnli": None,
    "mrpc": None,
    "qnli": None,
    "qqp": None,
    "rte": {
        # " positive" : 0,  # entailment
        # " negative" : 1    # not entailment
        # "entailment" : 0,  # entailment
        # "not entailment" : 1    # not entailment
        # " Yes" : 0,  # entailment
        # " No" : 1    # not entailment
        "True" : 0,  # entailment
        "False" : 1    # not entailment
    },
    "sst2": {
        "negative" : 0,
        "positive" : 1,
        # "bad" : 0,
        # "good" : 1,
        # "terrible" : 0,
        # "great" : 1,
    },
    "boolq": None,
    "cb": {
        "True" : 0,
        "False" : 1,
        "Neither" : 2,
    },
    "stsb": None,
    "wnli": None,
    "mr": None,
    "cr": None,
    "mpqa": None,
    "subj": None,
    "trec": None,
    "agnews": {
        "World" : 0,
        "Sports" : 1,
        "Business" : 2,
        "Tech" : 3,
    },
    "yahoo" : {
        " Society" : 0,
        " Science" : 1,
        " Health" : 2,
        " Education" : 3,
        " Computer" : 4,
        " Sports" : 5,
        " Business" : 6,
        " Entertainment" : 7,
        " Relationship" : 8,
        " Politics" : 9,
    },
    "yelp" : {
        ' terrible' : 0,
        ' bad' : 1,
        ' okay' : 2,
        ' good' : 3,
        ' great' : 4,
    },
    "sst5" : {
        ' terrible' : 0,
        ' bad' : 1,
        ' okay' : 2,
        ' good' : 3,
        ' great' : 4,
    }
}


def prepare_incontext_sampling(train_samples, 
                                verbalizer,
                                sentence1_key, 
                                sentence2_key,
                                prefix,
                                infix,
                                postfix,
                                ):

    label2token = {v:k for k,v in verbalizer.items()}
    label2samples = {}
    full_samples = []

    for sample in train_samples:
        sentence1 = sample[sentence1_key]
        label = sample['label']
        label_token = label2token[label]
        if sentence2_key is not None:
            sentence2 = sample[sentence2_key]
        else:
            sentence2 = ''
        
        full_sentence = prefix + sentence1 + infix + sentence2 + postfix + label_token
        full_samples.append(full_sentence)

        # empty list if first sample
        label_list = label2samples.get(label, [])
        label_list.append(full_sentence)
        label2samples[label] = label_list

    return label2samples, full_samples
        

def prepend_incontext_samples(
                                label2samples,
                                full_train_samples,
                                k,
                                balance_sample,
                                input_sentence,
                            ):
    # no in-context samples = zero-shot learning
    if k == 0:
        return input_sentence

    final_sentence = None
    sep = '\n\n\n'

    if balance_sample:
        total_count = 0
        while True:
            for label, samples in label2samples.items():
                total_length = len(samples)
                random_index = random.randint(0, total_length-1)
                selected_sample = samples[random_index]

                if final_sentence is None:
                    final_sentence = selected_sample
                else:
                    final_sentence = final_sentence + sep + selected_sample

                total_count += 1
                if total_count == k:
                    final_sentence = final_sentence + sep + input_sentence
                    return final_sentence
    else:
        total_length = len(input_sentence)
        for index in range(k):
            random_index = random.randint(0, total_length-1)
            selected_sample = full_train_samples[random_index]

            if final_sentence is None:
                final_sentence = selected_sample
            else:
                final_sentence = final_sentence + sep + selected_sample
       
        
    final_sentence = final_sentence + sep + input_sentence
    return final_sentence
