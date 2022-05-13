
from typing import Tuple
import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class GPT2Wrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path, verbalizer, args):
        super(GPT2Wrapper, self).__init__()

        self.config = config
        self.max_length = config.n_positions
        self.device = torch.device("cuda")

        self._init_logger(args)
        self.local_rank = args.local_rank

        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            revision="float16",             # specific model version to use. We use FP16 model
            torch_dtype=torch.float16,  
            low_cpu_mem_usage=True,         # keep RAM usage to 1x
        ).to(self.device)

        # for zero/few-shot inference. 
        # No gradient updates
        self.transformer.eval()

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.num_labels = config.num_labels
        # token -> label
        self.verbalizer = verbalizer
        # label -> token
        self.label2token = {v:k for k,v in self.verbalizer.items()}

        assert self.num_labels == len(self.verbalizer.keys()), f'Number of labels({self.num_labels}) and verbalizer({self.verbalizer}) does not match'

        self.ids_list, self.multiple_token_flag = self._convert_verbalizer_to_ids(self.label2token, self.tokenizer)

        # for analysis
        self.max_input_token = 0
        self.min_input_token = float('inf')
    
    def _init_logger(self, args) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )

        # Setup logging, we only want one process per machine to log things on the screen.
        logger.setLevel(logging.INFO if args.local_rank == 0 else logging.ERROR)

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

    # returns list of token ids of the verbalizer
    def _convert_verbalizer_to_ids(self, label2token, tokenizer):
        ids_list = []
        multiple_token_flag = False
        for label_index in range(self.num_labels):
            # index of the label -> label token from verbalizer
            token = label2token[label_index]
            # tokenize verbalizer
            ids = tokenizer(token)['input_ids']
            logger.info(f'> label_index {label_index} token {token} ids {ids}')
            ids_list.append(ids)
            # ids_list.append(ids[0])

            if len(ids) > 1:
                multiple_token_flag = True
                logger.info(f'Multiple token for verbalizer {token} -> {ids}')

        if not multiple_token_flag:
            ids_list = [ids[0] for ids in ids_list]

        assert len(ids_list) == self.num_labels

        return ids_list, multiple_token_flag

    # get log-probability of the token of the label
    # for multiple-tokens we normalize by length (for now)
    def _verbalize(
        self,
        logprobs,
        label_index=None,
    ):
        # multiple token verbalizer
        if self.multiple_token_flag:
            # shift log-probabilities
            logprobs = logprobs[:, :-1, :]
            label_tokens = self.ids_list[label_index]
            label_length = len(label_tokens)

            # shape : (1, label_length, vocab_size)
            # shape : (1, label-token-length, vocab_size)
            label_logprobs = logprobs[:, -label_length:, :]
            total_probs=0
            for input_index, token_index in enumerate(label_tokens, start=1):
                token_logprob = label_logprobs[0, -input_index, token_index]
                total_probs += token_logprob
            # TODO : normalize?
            total_probs = total_probs / label_length
        else:
            # single token verbalizer
            # use only the final distribution for prediction
            total_probs = logprobs[0, -1,  self.ids_list]
        return total_probs

    def forward(
        self,
        input_sentence,
        sentence1=None,
        sentence2=None,
        labels=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.multiple_token_flag:
            predictions = []

            # same as noisy channel inference
            for label_token, label_index in self.verbalizer.items():
                label_appended_input_sentence = input_sentence + label_token
                # tokenize label specific input sentence 
                tokenized_inputs = self.tokenizer(label_appended_input_sentence, return_tensors='pt').to(self.device)

                if input_ids_length > self.max_length:
                    logger.info(f'* Input longer than max length {self.max_length}')
                    logger.info(f'INPUT : {label_appended_input_sentence}')

                outputs = self.transformer(**tokenized_inputs)
                
                # shape : (1, length, vocab_size)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=2)
                # shape : (1, length, vocab_size)
                logprobs = torch.log(probs)

                verbalizer_logprob = self._verbalize(logprobs, label_index)

                assert label_index == len(predictions), f'label index : {label_index} <-> prediction count : {len(predictions)}'
                predictions.append(verbalizer_logprob)

            predictions = torch.stack(predictions)
        else:
            # tokenize label specific input sentence 
            tokenized_inputs = self.tokenizer(input_sentence, return_tensors='pt').to(self.device)
            # print('input ids', len(tokenized_inputs['input_ids']))

            input_ids_length = len(tokenized_inputs['input_ids'][0])

            if input_ids_length > self.max_length:
                logger.info(f'* Input longer than max length {self.max_length}')
                logger.info(f'INPUT : {label_appended_input_sentence}')

            # for analysis #
            if input_ids_length > self.max_input_token:
                self.max_input_token = input_ids_length
            if input_ids_length < self.min_input_token:
                self.min_input_token = input_ids_length
            # for analysis #

            with torch.no_grad():
                outputs = self.transformer(**tokenized_inputs)

            # shape : (1, length, vocab_size)
            logits = outputs.logits.cpu()

            # empty cache
            del outputs
            del tokenized_inputs
            torch.cuda.empty_cache()

            probs = torch.softmax(logits.float(), dim=2)
            # shape : (1, length, vocab_size)
            logprobs = torch.log(probs)

            predictions = self._verbalize(logprobs)

        prediction = torch.argmax(predictions, dim=-1)

        # shape : (1, )
        return prediction.unsqueeze(dim=0), predictions

    # for analysis
    def get_token_length_analysis(self):
        return self.max_input_token, self.min_input_token