import os
import time

import openai
from transformers import AutoTokenizer

from dataset_utils import task_to_verbalizer


class ModelWrapper:
    def __init__(self, model, task_name):

        # init OpenAI API #
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.organization = os.getenv('ORGANIZATION_ID')
        # until here #

        self.model = model

        # check if the model name is valid #
        engine_list = openai.Engine.list()
        engine_list = [engine['id'] for engine in engine_list['data']]
        assert self.model in engine_list, f'{self.model} not in {engine_list}'
        print(f'Using GPT-3 engine : {self.model}')

        # check if there is a valid verbalizer defined #
        assert task_name in task_to_verbalizer, f'{task_name} not in {task_to_verbalizer.keys()}'
        self.task_name = task_name
        self.verbalizer = task_to_verbalizer.get(task_name)

        # tokenizer for GPT2 #
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # delayed time when OpenAI API fails.
        self.retry_delay = 10

    def forward(
        self,
        input_sentence,
        sentence1=None,
        sentence2=None,
        labels=None,
        **kwargs,
    ):

        max_label_prob = -float('inf')
        predicted_label = None
        results_dict = dict()

        for label_token,label in self.verbalizer.items():
            input_sentence_with_label = input_sentence + label_token
            # print(input_sentence_with_label)

            sleep_time = 1
            while True:
                try:
                    response = openai.Completion.create(
                        engine=self.model,
                        prompt=input_sentence_with_label,
                        temperature=0,
                        max_tokens=0,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        echo=True,
                        logprobs=1
                    )
                    break
                except:
                    print(f'Error from OpenAI API. Pending for {sleep_time} seconds...')
                    time.sleep(sleep_time)
                    sleep_time += self.retry_delay

            data = response['choices'][0]
            logprobs = data['logprobs']
            token_logprobs = logprobs['token_logprobs']

            # check tokenization of the label token #
            label_token_input_ids = self.tokenizer(label_token)['input_ids']
            label_token_ids_length = len(label_token_input_ids)

            # print(label_token, '->', label_token_ids_length)

            # all the log-probabilities for the label token 
            label_probs = token_logprobs[-label_token_ids_length:]
            label_prob = 0
            for prob in label_probs:
                label_prob += prob

            label_prob = token_logprobs[-1]

            results_dict[label_token] = label_prob

            if label_prob > max_label_prob:
                max_label_prob = label_prob
                predicted_label = label


        return predicted_label, results_dict
        
    
    def generate(
        self,
        original_input,
        max_length=None,
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        input_sentence=None,
        sentence1=None,
        sentence2=None,
        labels=None,
        **kwargs,
    ):


        sleep_time = 1
        while True:
            try:
                response = openai.Completion.create(
                    engine=self.model,
                    prompt=original_input,
                    temperature=temperature,
                    max_tokens=max_length,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                )
                break
            except:
                print(f'Error from OpenAI API. Pending for {sleep_time} seconds...')
                time.sleep(sleep_time)
                sleep_time += self.retry_delay
        data = response['choices'][0]
        generated_text = data['text']

        return generated_text
        