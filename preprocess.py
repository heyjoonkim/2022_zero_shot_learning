from promptsource.templates import DatasetTemplates

rte_prompts = DatasetTemplates('glue','rte')
prompt = rte_prompts['imply']

def rte_preprocess(examples):
    results = prompt.apply(examples)
    result = dict()
    result['input_sentence'] = results[0]
    result['labels'] = examples['label']

    return result
