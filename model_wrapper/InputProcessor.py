

from typing import Tuple

import torch
import copy


class BaseInputProcessor(torch.nn.Module):
    def __init__(self, config, embeddings):
        super(BaseInputProcessor, self).__init__()
        
        self.config = config
        # self.embeddings = torch.nn.Embedding.from_pretrained(embeddings.weight)
        self.embeddings = embeddings
        self.embedding_dim = self.embeddings.embedding_dim

    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, length = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        assert batch_size == input_embeddings.shape[0]
        assert length == input_embeddings.shape[1]

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return input_embeddings, attention_mask





class PromptInputProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings, plm):
        super().__init__(config, embeddings)

        self.config = config

        assert config.plm_layer != -1
        # initialize encoder
        self.encoder = copy.deepcopy(plm.encoder.layer[config.plm_layer])

        assert config.prompt_length > 0, f'Prompt length must be greater than 0, got {self.prompt_length}.'
        self.prompt_length = config.prompt_length
        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, self.embedding_dim)

        # for positional embedding for the encoder
        self.position_embeddings = plm.embeddings.position_embeddings
        self.LayerNorm = plm.embeddings.LayerNorm
        self.dropout = plm.embeddings.dropout

    # from transfomers.RobertaEmbedding
    def _create_position_ids_from_inputs_embeds(self, embeddings):

        input_shape = embeddings.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.config.pad_token_id + 1, sequence_length + self.config.pad_token_id + 1, dtype=torch.long, device=embeddings.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def _get_input_embeddings(self, embeddings):
        position_ids = self._create_position_ids_from_inputs_embeds(embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    # from transformers.modeling_utils.ModeulUtilsMixin
    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int], device) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.
        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids:torch.Tensor,             # shape : (batch, input+prompt length)
        attention_mask:torch.Tensor,        # shape : (batch, input+prompt length)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, length = input_ids.shape


        # shape : (batch, length+prompt, embedding_dim)
        input_embeddings = self.embeddings(input_ids)
        

        # shape : (prompt_length, )
        prompt_token_ids = torch.LongTensor(list(range(self.prompt_length))).to(input_embeddings.device)
        # shape : (batch, prompt_length)
        prompt_token_ids = prompt_token_ids.repeat(batch_size, 1)
        # shape : (batch, prompt_length, embedding_dim)
        prompt_embeddings = self.prompt_embeddings(prompt_token_ids)

        # add new prompt embeddings
        input_embeddings[:, 1:self.prompt_length+1, :] = prompt_embeddings

        # add positional embeddings
        input_embeddings = self._get_input_embeddings(input_embeddings)

        expanded_attention_mask = self.get_extended_attention_mask(attention_mask=attention_mask, input_shape=attention_mask.size(), device=input_embeddings.device)

        outputs = self.encoder(hidden_states=input_embeddings, attention_mask=expanded_attention_mask)
        # shape : (batch, input+prompt_length, embedding_dim)
        outputs_hidden_states = outputs[0]
        # shape : (batch, prompt_length, embedding_dim)
        input_dependent_prompt_embeddings = outputs_hidden_states[:, 1:self.prompt_length+1, :]

        # shape : (batch, length+prompt, embedding_dim)
        final_embeddings = self.embeddings(input_ids)

        # add new prompt embeddings
        final_embeddings[:, 1:self.prompt_length+1, :] = input_dependent_prompt_embeddings

        return final_embeddings, attention_mask

