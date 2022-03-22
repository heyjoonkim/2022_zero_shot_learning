
from typing import Tuple
import time

import torch

from transformers import AutoModel

from .InputProcessor import BaseInputProcessor, PromptInputProcessor
from .OutputProcessor import BaseOutputProcessor



class ModelWrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path):
        super(ModelWrapper, self).__init__()

        self.config = config

        # Main model
        self.transformer = AutoModel.from_pretrained(
                                        model_name_or_path,
                                        from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config,
                                        add_pooling_layer=False)

        self.embedding_dim = self.transformer.get_input_embeddings().embedding_dim
        self.num_labels = config.num_labels


        if self.config.apply_prompt:
            self.input_processor = PromptInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings(), plm=self.transformer)
        else:
            # default input and output processor for out toy task
            self.input_processor = BaseInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        # goes through (embedding_dim, num_label) linear layer
        self.output_processor = BaseOutputProcessor(config=config, embedding_dim=self.embedding_dim, num_labels=self.num_labels, model_name_or_path=model_name_or_path)
    
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

        inputs_embeds, attention_mask = self.input_processor(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # shape : (batch, length, embedding_dim)
        last_hidden_state = outputs.last_hidden_state

        # loss        : (batch, )
        # predictions : (batch, )
        loss, predictions = self.output_processor(last_hidden_state=last_hidden_state, attention_mask=attention_mask, labels=labels)
        
        return loss, predictions
