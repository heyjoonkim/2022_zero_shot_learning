from typing import Tuple

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class BaseOutputProcessor(torch.nn.Module):
    def __init__(self, config, embedding_dim, num_labels, model_name_or_path):
        super(BaseOutputProcessor, self).__init__()

        self.config = config
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.model_name_or_path = model_name_or_path

        # final layer for prediction
        self.score = torch.nn.Linear(self.embedding_dim, self.num_labels, bias=True)
        self.dense = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.dropout = torch.nn.Dropout(0.1)


        if config.pooling_method == "cls":
            self.pooler = self._mean_pooling
        elif config.pooling_method == "mean":
            self.pooler = self._mean_pooling


    def forward(
        self,
        last_hidden_state:torch.Tensor,                # shape : (batch, length, embedding_dim)
        attention_mask:torch.Tensor,        # shape : (batch, length)
        labels:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, _ = attention_mask.shape

        # pooling
        # shape : (batch, length, embedding_dim) -> (batch, embedding_dim)
        last_hidden_state = self.pooler(last_hidden_state, attention_mask)

        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.dense(last_hidden_state)
        last_hidden_state = torch.tanh(last_hidden_state)
        last_hidden_state = self.dropout(last_hidden_state)
        # shape : (batch, num_labels)
        pooled_logits = self.score(last_hidden_state)

       
        ## same code as transformers.GPT2ForSequenceClassification ##
        loss = None
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

    
    def _mean_pooling(self, last_hidden_state, attention_mask):
        ## mean pooling ##
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # shape : (batch, embedding_dim)
        mean_embedding = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return mean_embedding

    def _cls_pooling(self, last_hidden_state, attention_mask):
        return last_hidden_state[:, 0, :]