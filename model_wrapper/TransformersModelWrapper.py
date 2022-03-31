
from typing import Tuple

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseOutputProcessor(torch.nn.Module):
    def __init__(self, config, num_labels, verbalizer, ids_list):
        super(BaseOutputProcessor, self).__init__()

        self.config = config
        self.num_labels = num_labels
        self.verbalizer = verbalizer
        self.ids_list = ids_list

    def _verbalize(self, logits):
        # shape : (batch, num_labels)
        result = logits[:, self.ids_list]
        batch_size, num_labels = result.shape
        assert num_labels == self.num_labels

        return result

    def forward(
        self,
        last_hidden_state:torch.Tensor,     # shape : (batch, length, vocab_size)
        attention_mask:torch.Tensor,        # shape : (batch, length)
        labels:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, _ = attention_mask.shape

        # get the index of the final representation
        sequence_lengths = torch.ne(attention_mask, 0).sum(-1) - 1

        # shape : (batch, vocab_size)
        pooled_logits = last_hidden_state[range(batch_size), sequence_lengths]

        # shape : (batch, num_labels)
        pooled_logits = self._verbalize(pooled_logits)

        ## same code as transformers.GPT2ForSequenceClassification ##
        loss = None
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(pooled_logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels)
        ## until here ##

        # shape : (batch, )
        predictions = pooled_logits.argmax(dim=-1)

        # loss        : (batch, )
        # predictions : (batch, )
        return loss, predictions

class GPT2Wrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path, verbalizer):
        super(GPT2Wrapper, self).__init__()

        self.config = config

        # Main model
        self.transformer = AutoModelForCausalLM.from_pretrained(
                                                            model_name_or_path,
                                                            from_tf=bool(".ckpt" in model_name_or_path),
                                                            config=config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.num_labels = config.num_labels
        # token -> label
        self.verbalizer = verbalizer
        # label -> token
        self.label2token = {v:k for k,v in self.verbalizer.items()}

        assert self.num_labels == len(self.verbalizer.keys()), f'Number of labels({self.num_labels}) and verbalizer({self.verbalizer}) does not match'

        self.ids_list = self._convert_verbalizer_to_ids(self.label2token, self.tokenizer)

        # for output processing (output logits -> loss, prediction)
        self.output_processor = BaseOutputProcessor(
            config=config, 
            num_labels=self.num_labels,
            verbalizer=self.verbalizer,
            ids_list=self.ids_list,        
        )

    # returns list of token ids of the verbalizer
    def _convert_verbalizer_to_ids(self, label2token, tokenizer):
        ids_list = []
        for label_index in range(self.num_labels):
            token = label2token[label_index]
            ids = tokenizer(token)['input_ids']
            if len(ids) > 1:
                raise NotImplementedError(f'Verbalizer with more than one token is not implemented yet. {token} -> {ids}')
            ids_list.append(ids[0])
            # print(label_index, token, '->', ids)

        assert len(ids_list) == self.num_labels

        return ids_list



    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # input_ids      : (batch, input_length)
        # attention_mask : (batch, input_length)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # shape : (batch, length, vocab_size)
        last_hidden_state = outputs.logits

        # loss        : (batch, )
        # predictions : (batch, )
        loss, predictions = self.output_processor(last_hidden_state=last_hidden_state, attention_mask=attention_mask, labels=labels)

        return loss, predictions