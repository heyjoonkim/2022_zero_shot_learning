import os
import time
import requests

import openai
from transformers import AutoTokenizer

from dataset_utils import task_to_verbalizer


class ModelWrapper:
    def __init__(self, model, task_name):
        # init OpenAI API #
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.organization = os.getenv('ORGANIZATION_ID')

        self.model = model

        # check if the model name is valid #
        engine_list = openai.Engine.list()
        engine_list = [engine['id'] for engine in engine_list['data']]
        if self.model not in engine_list:
            self.openai = False
            print(f'Using Transformer engine : {self.model}')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.model = FlaskModel(self.model)
        else:
            self.openai = True
            print(f'Using OpenAI engine : {self.model}')
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # check if there is a valid verbalizer defined #
        assert task_name in task_to_verbalizer, f'{task_name} not in {task_to_verbalizer.keys()}'
        self.task_name = task_name
        self.verbalizer = task_to_verbalizer.get(task_name)

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

        for label_token, label in self.verbalizer.items():
            input_sentence_with_label = input_sentence + label_token
            # print(input_sentence_with_label)

            sleep_time = 1

            while True:
                try:
                    if self.openai:
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
                    else:
                        response = self.model.completion(
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
                    print(f'Error from API. Pending for {sleep_time} seconds...')
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

            label_prob /= label_token_ids_length
            # label_prob = token_logprobs[-1]

            results_dict[label_token] = label_prob

            if label_prob > max_label_prob:
                max_label_prob = label_prob
                predicted_label = label

        return predicted_label, results_dict
            
    def generate(
        self,
        positive_prompt=None,
        neutral_prompt=None,
        negative_prompt=None,
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
        positive_label = 0
        negative_label = 1 if neutral_prompt is None else 2
        neutral_label = 1 if neutral_prompt is not None else None

        prompt_list = [(positive_prompt, positive_label), (neutral_prompt, neutral_label), (negative_prompt, negative_label)]

        generated_result = []
        for prompt, expected_label in prompt_list:
            if prompt is None:
                generated_result.append((None, None))
                continue
            
            input_sentence_with_prompt = sentence1 + prompt

            sleep_time = 1
            while True:
                try:
                    if self.openai:
                        response = openai.Completion.create(
                            engine=self.model,
                            prompt=input_sentence_with_prompt,
                            temperature=temperature,
                            max_tokens=max_length,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                        )
                    else:
                        response =  self.model.completion(
                            prompt=input_sentence_with_prompt,
                            temperature=temperature,
                            max_tokens=max_length,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                        )
                    break
                except:
                    print(f'Error from API. Pending for {sleep_time} seconds...')
                    time.sleep(sleep_time)
                    sleep_time += self.retry_delay

            data = response['choices'][0]
            generated_text = data['text']
            # we only use the first generated text
            if '.' in generated_text:
                index = generated_text.rindex('.')
                generated_text = generated_text[:index]

            generated_text = generated_text.strip()
            generated_text = generated_text + '.'

            # print(input_sentence_with_prompt, '->', generated_text)

            generated_result.append((generated_text, expected_label))
        
        # list : [(generated_text, exptected_label), ..., ]
        return generated_result
        

class FlaskModel:
    def __init__(self, model_name_or_path, url='http://127.0.0.1:5000/'):

        self.url = url
        r = requests.get(self.url)
        response = r.json()
        assert response['model_name'] == model_name_or_path, "Requested model does not match the current API model"
    
    def completion(
        self,
        prompt,
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=None,
        presence_penalty=None,
        echo=None,
        logprobs=None,
        n=1
    ):
        data = {'prompt': prompt, 
                'temperature': temperature, 
                'max_tokens': max_tokens, 
                'top_p': top_p,
                'frequency_penalty': frequency_penalty,
                'presence_penalty': presence_penalty,
                'echo': echo,
                'logprobs': logprobs,
                'n': n
            }
        r = requests.post(self.url + 'completion', json=data)
        response = r.json()

        return(response)
        