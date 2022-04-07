
from typing import Tuple

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LogSoftmax

from transformers import AutoModelForCausalLM, AutoTokenizer

class GPTModel:
    def __init__(self, model_name_or_path):

        self.device = torch.device("cuda")
        # Main model
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to(self.device)
        self.transformer.config.pad_token_id = self.transformer.config.eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


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
    ):
        if temperature == 0:
            temperature = 0.01

        sm = LogSoftmax(dim=1)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = len(inputs['input_ids'][0])
        outputs = self.transformer(**inputs)
        logits = outputs.logits[0][:-1]

        if max_tokens == 0:
            total_logits = logits
            output_ids = inputs['input_ids'][0]
        else:
            sample_outputs = self.transformer.generate(
                inputs['input_ids'],
                do_sample=True, 
                temperature=temperature,
                max_new_tokens=max_tokens, 
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                top_p=0.92, 
                top_k=0,
                repetition_penalty=2.0
            )
            output_ids = sample_outputs.sequences[0]
            generated_logits = torch.cat(sample_outputs.scores,dim=0)
            total_logits = torch.cat([logits, generated_logits],dim=0)
        prob = sm(total_logits)

        output_prob = prob[range(prob.shape[0]), output_ids[1:]]
        output_prob = output_prob.cpu().detach().tolist()

        output_ids = output_ids.cpu().detach().tolist()

        tokens = [self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(token)) for token in output_ids]
        token_logprobs = [None]
        for p in output_prob:
            token_logprobs.append(p)

        if not echo:
            tokens = tokens[input_len:]
            token_logprobs = token_logprobs[input_len:]
        text = ''.join(tokens)

        result = {'logprobs': {'token_logprobs': token_logprobs, 'tokens':tokens}, 'text': text}
        results = {'choices':[result]}

        return results
