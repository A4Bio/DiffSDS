import time
import torch
import math
import torch.nn as nn
import argparse
import typing
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
)
from modules import BertEmbeddings, AnglesPredictor, GaussianFourierProjection, SinusoidalPositionEmbeddings
from typing import *
import logging

LR_SCHEDULE = Optional[Literal["OneCycleLR", "LinearWarmup"]]
TIME_ENCODING = Literal["gaussian_fourier", "sinusoidal"]
LOSS_KEYS = Literal["l1", "smooth_l1"]
DECODER_HEAD = Literal["mlp", "linear"]

class BertForDiffusion(BertPreTrainedModel):
    def __init__(
        self,
        config,
        ft_is_angular: List[bool] = [False, True, True, True],
        ft_names: Optional[List[str]] = None,
        time_encoding: TIME_ENCODING = "gaussian_fourier",
        decoder: DECODER_HEAD = "mlp"
    ) -> None:
        """
        dim should be the dimension of the inputs
        """
        super().__init__(config)
        self.config = config
        if self.config.is_decoder:
            raise NotImplementedError
        n_inputs = len(ft_is_angular)
        self.n_inputs = n_inputs

        if ft_names is not None:
            self.ft_names = ft_names
        else:
            self.ft_names = [f"ft{i}" for i in range(n_inputs)]
        assert (
            len(self.ft_names) == n_inputs
        ), f"Got {len(self.ft_names)} names, expected {n_inputs}"
        
        # Needed to project the low dimensional input to hidden dim
        self.inputs_to_hidden_dim = nn.Linear(
            in_features=n_inputs, out_features=config.hidden_size
        )
        self.embeddings = BertEmbeddings(config)
        self.seq_embedding = nn.Sequential(
            nn.Embedding(21, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.mask_embedding = nn.Sequential(
            nn.Embedding(2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.encoder = BertEncoder(config)

        # Set up the network to project token representation to our four outputs
        if decoder == "linear":
            self.token_decoder = nn.Linear(config.hidden_size, n_inputs)
        elif decoder == "mlp":
            self.token_decoder = AnglesPredictor(config.hidden_size, n_inputs)
        else:
            raise ValueError(f"Unrecognized decoder: {decoder}")

        # Set up the time embedder
        if time_encoding == "gaussian_fourier":
            self.time_embed = GaussianFourierProjection(config.hidden_size)
        elif time_encoding == "sinusoidal":
            self.time_embed = SinusoidalPositionEmbeddings(config.hidden_size)
        else:
            raise ValueError(f"Unknown time encoding: {time_encoding}")

        # Initialize weights and apply final processing
        self.init_weights()

        # Epoch counters and timers
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()
    
    @classmethod
    def get_params(self, timesteps):
        f = lambda t: torch.cos(t/self.T * torch.pi/2) 
        
        def alpha_t_s(t,s):
            return f(t)/f(s)

        def sigma_t_s(t,s):
            return torch.clip(1-alpha_t_s(t,s),min=0,max=0.999)
        
        # compute b_t
        self.T = torch.tensor(1000, device=timesteps.device)
        sigma_T = sigma_t_s(self.T,0)
        sigma_T_t = sigma_t_s(self.T,timesteps)
        b_t = sigma_T_t**2/sigma_T**2
        
        # compute alpha_t, sigma_t
        ZERO = torch.tensor(0, device=timesteps.device)
        alpha_t = alpha_t_s(timesteps,ZERO)
        alpha_t = alpha_t_s(timesteps,ZERO)
        sigma_t = sigma_t_s(timesteps, ZERO)
        
        return b_t, alpha_t, sigma_t
        
    @classmethod
    def get_xs_xt(self, angles, coords, unknown_mask, start_idx, end_idx, timesteps):
        # 1. params
        b_s, alpha_s, sigma_s = self.get_params(timesteps-1)
        b_t, alpha_t, sigma_t = self.get_params(timesteps)
        
        dot_z_s = b_s[:,None]*angles
        dot_z_t = b_t[:,None]*angles
        
        noise = torch.randn_like(angles)
        x_s = dot_z_s*alpha_s[:,None]
        x_t = dot_z_t*alpha_t[:,None] + sigma_t[:,None]*noise
        
        return dot_z_s, noise, x_t, coords
    
    
    
    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,  # Tensor of shape batch_length with time indices
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        seqs: torch.Tensor, 
        unknown_mask: torch.Tensor, 
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        t1 = time.time()
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_shape = inputs.size()
        batch_size, seq_length, *_ = input_shape
        logging.debug(f"Detected batch {batch_size} and seq length {seq_length}")

        assert attention_mask is not None

        # If position IDs are not given, auto-generate them
        if position_ids is None:
            # [1, seq_length]
            position_ids = (
                torch.arange(
                    seq_length,
                )
                .expand(batch_size, -1)
                .type_as(timestep)
            )


        assert (
            attention_mask.dim() == 2
        ), f"Attention mask expected in shape (batch_size, seq_length), got {attention_mask.shape}"
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(attention_mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        assert len(inputs.shape) == 3  # batch_size, seq_length, features
        inputs_upscaled = self.inputs_to_hidden_dim(inputs)  # Batch * seq_len * dim

        # Pass through embeddings
        inputs_upscaled = self.embeddings(inputs_upscaled, position_ids=position_ids)
        seq_encoded = self.seq_embedding(seqs).squeeze()
        mask_encoded = self.mask_embedding(unknown_mask.long()).squeeze()

        # timestep is (batch, 1), squeeze to (batch,)
        # embedding gets to (batch, embed_dim) -> unsqueee to (batch, 1, dim)
        time_encoded = self.time_embed(timestep.squeeze(dim=-1)).unsqueeze(1) # [64, 1, 384]
        inputs_with_time = (inputs_upscaled + time_encoded + seq_encoded)* torch.sigmoid(mask_encoded) # [64, 128, 384]
        encoder_outputs = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            # head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0] # [64, 128, 384]
        per_token_decoded = self.token_decoder(sequence_output)
        t2 = time.time()
        # print(t2-t1)
        return per_token_decoded


