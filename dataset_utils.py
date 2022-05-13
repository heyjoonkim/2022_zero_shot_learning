
#
# Utils for loading datasets from file (csv, tsv, ...).
# otherwise we use load_dataset() from huggingface library.
#

import csv
import random
import ast

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

# Single sentence
# for Generated datasets in TREC.
def generated_trec_generate_dataset_dict(filename):
    sentence1_list = []
    label_list = []
    samples0_list = []
    samples1_list = []
    samples2_list = []
    samples3_list = []
    samples4_list = []
    samples5_list = []

    with open(filename) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line_index, line in enumerate(tsv_reader):

            assert len(line) == 9, f'Line length {len(line)} does not match the expected length 9.'
            
            index = int(line[0])
            label = int(line[1])
            sentence1 = line[2]

            assert line_index == index, f'index {index} != line_index {line_index}'

            # convert to list
            samples0 = ast.literal_eval(line[3])
            samples1 = ast.literal_eval(line[4])
            samples2 = ast.literal_eval(line[5])
            samples3 = ast.literal_eval(line[6])
            samples4 = ast.literal_eval(line[7])
            samples5 = ast.literal_eval(line[8])

            # assert len(samples0) == len(samples1), f'number samples for label 0 {samples0} does not match the number of samples for label 1 {len(samples1)}'
            # assert len(samples0) == len(samples2), f'number samples for label 0 {samples0} does not match the number of samples for label 2 {len(samples2)}'
            # assert len(samples0) == len(samples3), f'number samples for label 0 {samples0} does not match the number of samples for label 3 {len(samples3)}'
            # assert len(samples0) == len(samples4), f'number samples for label 0 {samples0} does not match the number of samples for label 4 {len(samples4)}'
            # assert len(samples0) == len(samples5), f'number samples for label 0 {samples0} does not match the number of samples for label 5 {len(samples5)}'

            label_list.append(label)
            sentence1_list.append(sentence1)
            samples0_list.append(samples0)
            samples1_list.append(samples1)
            samples2_list.append(samples2)
            samples3_list.append(samples3)
            samples4_list.append(samples4)
            samples5_list.append(samples5)

    return_dict = {
        'text' : sentence1_list,
        'label' : label_list,
        'samples0' : samples0_list,
        'samples1' : samples1_list,
        'samples2' : samples2_list,
        'samples3' : samples3_list,
        'samples4' : samples4_list,
        'samples5' : samples5_list,
    }

    return return_dict

def generated_sst5_generate_dataset_dict(filename):
    sentence1_list = []
    label_list = []
    samples0_list = []
    samples1_list = []
    samples2_list = []
    samples3_list = []
    samples4_list = []

    with open(filename) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line_index, line in enumerate(tsv_reader):

            assert len(line) == 8, f'Line length {len(line)} does not match the expected length 8.'
            
            index = int(line[0])
            label = int(line[1])
            sentence1 = line[2]

            assert line_index == index, f'index {index} != line_index {line_index}'

            # convert to list
            samples0 = ast.literal_eval(line[3])
            samples1 = ast.literal_eval(line[4])
            samples2 = ast.literal_eval(line[5])
            samples3 = ast.literal_eval(line[6])
            samples4 = ast.literal_eval(line[7])

            # assert len(samples0) == len(samples1), f'number samples for label 0 {samples0} does not match the number of samples for label 1 {len(samples1)}'
            # assert len(samples0) == len(samples2), f'number samples for label 0 {samples0} does not match the number of samples for label 2 {len(samples2)}'
            # assert len(samples0) == len(samples3), f'number samples for label 0 {samples0} does not match the number of samples for label 3 {len(samples3)}'
            # assert len(samples0) == len(samples4), f'number samples for label 0 {samples0} does not match the number of samples for label 4 {len(samples4)}'
            
            label_list.append(label)
            sentence1_list.append(sentence1)
            samples0_list.append(samples0)
            samples1_list.append(samples1)
            samples2_list.append(samples2)
            samples3_list.append(samples3)
            samples4_list.append(samples4)

    return_dict = {
        'sentence' : sentence1_list,
        'label' : label_list,
        'samples0' : samples0_list,
        'samples1' : samples1_list,
        'samples2' : samples2_list,
        'samples3' : samples3_list,
        'samples4' : samples4_list,
    }

    return return_dict

def generated_cb_generate_dataset_dict(filename):
    sentence1_list = []
    sentence2_list = []
    label_list = []
    samples0_list = []
    samples1_list = []
    samples2_list = []

    with open(filename) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line_index, line in enumerate(tsv_reader):

            assert len(line) == 7, f'Line length {len(line)} does not match the expected length 7.'
            
            index = int(line[0])
            label = int(line[1])
            sentence1 = line[2]
            sentence2 = line[3]

            assert line_index == index, f'index {index} != line_index {line_index}'

            # convert to list
            samples0 = ast.literal_eval(line[4])
            samples1 = ast.literal_eval(line[5])
            samples2 = ast.literal_eval(line[6])

            # assert len(samples0) == len(samples1), f'number samples for label 0 {samples0} does not match the number of samples for label 1 {len(samples1)}'
            # assert len(samples0) == len(samples2), f'number samples for label 0 {samples0} does not match the number of samples for label 2 {len(samples2)}'
           
            label_list.append(label)
            sentence1_list.append(sentence1)
            sentence2_list.append(sentence2)
            samples0_list.append(samples0)
            samples1_list.append(samples1)
            samples2_list.append(samples2)

    return_dict = {
        'premise' : sentence1_list,
        'hypothesis' : sentence2_list,
        'label' : label_list,
        'samples0' : samples0_list,
        'samples1' : samples1_list,
        'samples2' : samples2_list,
    }

    return return_dict

def generated_sst2_generate_dataset_dict(filename):
    sentence1_list = []
    label_list = []
    samples0_list = []
    samples1_list = []

    with open(filename) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line_index, line in enumerate(tsv_reader):

            assert len(line) == 5, f'Line length {len(line)} does not match the expected length 5.'
            
            index = int(line[0])
            label = int(line[1])
            sentence1 = line[2]

            assert line_index == index, f'index {index} != line_index {line_index}'

            # convert to list
            samples0 = ast.literal_eval(line[3])
            samples1 = ast.literal_eval(line[4])
            
            label_list.append(label)
            sentence1_list.append(sentence1)
            samples0_list.append(samples0)
            samples1_list.append(samples1)

    return_dict = {
        'sentence' : sentence1_list,
        'label' : label_list,
        'samples0' : samples0_list,
        'samples1' : samples1_list,
    }

    return return_dict

# for using generated datasets.
generated_task_to_path = {
    
    "trec" : {
        "validation" : "test.tsv",
        "dataset_processor" : generated_trec_generate_dataset_dict,
    },
    "sst5" : {
        "validation" : "test.tsv",
        "dataset_processor" : generated_sst5_generate_dataset_dict,
    },
    "cb" : {
        "validation" : "test.tsv",
        "dataset_processor" : generated_cb_generate_dataset_dict,
    },
    "sst2" : {
        "validation" : "test.tsv",
        "dataset_processor" : generated_sst2_generate_dataset_dict,
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
    "trec": ("text", None),
    "agnews": ("sentence1", "sentence2"),
    "yahoo": ("sentence1", "sentence2"),
    "yelp": ("sentence", None),
}

task_to_verbalizer = {
    "cola": None,
    "mnli": None,
    "mrpc": {
        # VERBALIZER 1
        # " True" : 1,
        # " False" : 0,
        # VERBALIZER 2
        "True" : 1,
        "False" : 0,
    },
    "qnli": None,
    "qqp": None,
    "rte": {
        # " positive" : 0,  # entailment
        # " negative" : 1    # not entailment
        # "entailment" : 0,  # entailment
        # "not entailment" : 1    # not entailment
        " Yes" : 0,  # entailment
        " No" : 1    # not entailment
        # "True" : 0,  # entailment
        # "False" : 1    # not entailment
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
        " true" : 0,
        " false" : 1,
        " neither" : 2,
    },
    "stsb": None,
    "wnli": None,
    "mr": None,
    "cr": None,
    "mpqa": None,
    "subj": None,
    "trec": {
        # 'Description':0,
        # 'Entity':1,
        # 'Abbreviation':2, # or Expression
        # 'Person':3, # or Human
        # 'Number':4,
        # 'Location':5,
        'description':0,
        'entity':1,
        'expression':2,
        'human':3,
        'number':4,
        'location':5,
    },
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
        if 'label' in sample:
            label = sample['label']
        elif 'label-coarse' in sample:
            label = sample['label-coarse']
        else:
            raise NotImplementedError
            
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
                            ):

    
    final_sentence = None
    sep = '\n\n\n'
    # sep = '\n\n\n\n'

    # no in-context samples = zero-shot learning
    if k == 0:
        return '', sep

    if balance_sample:
        total_count = 0
        labels = list(label2samples.keys())
        random.shuffle(labels)
        # prevent infinite while-loop
        samples_map = {label:[i for i in range(len(label2samples[label]))] for label in labels}
        while True:
            for label in labels:
                samples = label2samples[label]
                total_length = len(samples)
                not_used_indices = [i for i in range(total_length)]
                while True:
                    samples_list = samples_map[label]
                    random_index = random.randint(0, total_length-1)
                    selected_sample = samples[random_index]

                    # we don't want to use duplicate in-context samples
                    if final_sentence is None:
                        selected_index = samples_list.index(random_index)
                        samples_list.pop(selected_index)
                        samples_map[label] = samples_list
                        break
                    if random_index in samples_list:
                        selected_index = samples_list.index(random_index)
                        samples_list.pop(selected_index)
                        samples_map[label] = samples_list
                        break

                if final_sentence is None:
                    final_sentence = selected_sample
                else:
                    final_sentence = final_sentence + sep + selected_sample

                total_count += 1
                if total_count == k:
                    return final_sentence, sep
    else:
        full_train_samples_copy = full_train_samples.copy()
        for index in range(k):
            total_length = len(full_train_samples_copy)
            random_index = random.randint(0, total_length-1)
            selected_sample = full_train_samples_copy.pop(random_index)

            if final_sentence is None:
                final_sentence = selected_sample
            else:
                final_sentence = final_sentence + sep + selected_sample

    return final_sentence, sep



def prepare_generated_incontext_sampling(generated_samples, 
                                verbalizer,
                                prefix,
                                infix,
                                postfix,
                                sentence1_key,
                                sentence2_key,
                                append_label=True
                                ):

    label2token = {v:k for k,v in verbalizer.items()}
    num_labels = len(label2token.keys())
    label2samples_list=[] 
    full_samples_list=[]

    for samples in generated_samples:
        label2samples = {}
        full_samples = []
        # if sentence2_key is not None -> sentence-pair task -> use the first sentence
        sentence1 = samples[sentence1_key] if sentence2_key is not None else None

        for label in range(num_labels):
            label_token = label2token[label]
            if not append_label:
                label_token = ''
            key = f'samples{label}'
            samples_list = samples[key]

            promped_samples_list = []
            for sample_index, sample in enumerate(samples_list):
                if sentence1:
                    promped_samples_list.append(prefix + sentence1 + infix + sample +postfix + label_token)
                else:
                    promped_samples_list.append(prefix + sample + infix + postfix + label_token)
            # samples_list = [prefix + sample + infix + postfix + label_token for sample in samples_list]

            full_samples = full_samples + promped_samples_list
            label2samples[label] = promped_samples_list
        
        label2samples_list.append(label2samples)
        full_samples_list.append(full_samples)


    return label2samples_list, full_samples_list