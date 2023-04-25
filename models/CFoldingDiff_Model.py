import time
import torch
import math
import torch.nn as nn
import argparse
import typing
from transformers import BertConfig
from modules.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
)
from modules import BertEmbeddings, AnglesPredictor, GaussianFourierProjection, SinusoidalPositionEmbeddings # aaa
from typing import *
import logging
from utils.nerf import TorchNERFBuilder
from utils import beta_schedules
import utils

LR_SCHEDULE = Optional[Literal["OneCycleLR", "LinearWarmup"]]
TIME_ENCODING = Literal["gaussian_fourier", "sinusoidal"]
LOSS_KEYS = Literal["l1", "smooth_l1"]
DECODER_HEAD = Literal["mlp", "linear"]


def rigid_transform_P2Q(P, Q):
    '''
    mapping from P to Q
    P: [batch, N, 3]
    Q: [batch, N, 3]
    Q = (R @ P.permute(0,2,1)).permute(0,2,1) + t
    '''
    # find mean column wise: 3 x 1
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)

    # subtract mean
    Pm = P - centroid_P
    Qm = Q - centroid_Q

    H = Pm.permute(0,2,1) @ Qm

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    d = torch.sign(torch.linalg.det(Vt.permute(0,2,1) @ U.permute(0,2,1)))
    SS = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=1))

    R = (Vt.permute(0,2,1) @ SS) @ U.permute(0,2,1)

    t = -(R @ centroid_P.permute(0,2,1)).permute(0,2,1) + centroid_Q

    return R, t

class CBertForDiffusion(BertPreTrainedModel):
    def __init__(
        self,
        config,
        step_gamma = 0.00001,
        ft_is_angular: List[bool] = [False, True, True, True],
        ft_names: Optional[List[str]] = None,
        time_encoding: TIME_ENCODING = "gaussian_fourier",
        decoder: DECODER_HEAD = "mlp",
        use_grad=1
    ) -> None:
        """
        dim should be the dimension of the inputs
        """
        super().__init__(config)
        self.config = config
        self.use_grad = use_grad
        self.step_gamma = step_gamma
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
        self.distance_to_hidden_dim =  nn.Sequential(
            nn.Linear(in_features=32, out_features=config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.config.num_attention_heads),
        )
        
        
        self.inputs_to_hidden_dim = nn.Linear(
            in_features=n_inputs, out_features=config.hidden_size
        )
        self.data_builder = TorchNERFBuilder(0)
        self.embeddings = BertEmbeddings(config)

        self.num_embedding = nn.Sequential(
            nn.Embedding(128, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.length_embedding = nn.Sequential(
            nn.Linear(32, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
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
    
    def pred_coord(self, coords, start_idx, end_idx, phi, psi, omega, C_1N_1CA, tau, CA_C_1N):
        idx = torch.arange(start_idx.shape[0], device = start_idx.device)
        start_coords = coords[idx, start_idx]
        end_coords = coords[idx, end_idx]
        max_length = (end_idx-start_idx).max()+1

        phi2 = []
        psi2 = []
        omega2 = []
        C_1N_1CA2 = []
        tau2 = []
        CA_C_1N2 = []
        for i in range(max_length.item()):
            idx2 = torch.clamp(start_idx+i,max=phi.shape[1]-1) # next
            phi2.append(phi[idx, idx2])
            psi2.append(psi[idx, idx2])
            omega2.append(omega[idx, idx2])
            C_1N_1CA2.append(C_1N_1CA[idx, idx2])
            tau2.append(tau[idx, idx2])
            CA_C_1N2.append(CA_C_1N[idx, idx2])
        
        phi2 = torch.concat(phi2, dim=1) # C_(i-1),N_i,CA_i,C_i 
        psi2 = torch.concat(psi2, dim=1) # N_i,CA_i,C_i,N_(i+1)
        omega2 = torch.concat(omega2, dim=1) # CA_(i-1),C_(i-1),N_i,CA_i
        C_1N_1CA2 = torch.concat(C_1N_1CA2, dim=1) # C_i,N_(i+1),CA_(i+1)
        tau2 = torch.concat(tau2, dim=1) #  N_(i+1),CA_(i+1),C_(i+1)
        CA_C_1N2 = torch.concat(CA_C_1N2, dim=1) # CA_i,C_i,N_(i+1)

        self.data_builder.set_values(phi2.detach(), psi2.detach(), omega2.detach(), bond_angle_n_ca = C_1N_1CA2.detach(), bond_angle_ca_c = tau2.detach(), bond_angle_c_n = CA_C_1N2.detach(), init_coords=start_coords.detach(), max_length = max_length)

        optimize = False
        if optimize:
            # torch.set_grad_enabled(True)
            ## optimize
            step_gamma = self.step_gamma
            grad = self.data_builder.Gradient(start_coords.detach(), end_coords.detach(), end_idx-start_idx)
            # torch.set_grad_enabled(False)
            phi2 = phi2-grad['phi']*step_gamma
            psi2 = psi2-grad['psi']*step_gamma
            omega2 = omega2-grad['omega']*step_gamma
            C_1N_1CA2 = (C_1N_1CA2-grad['bond_angle_n_ca']*step_gamma).detach()
            tau2 = (tau2-grad['bond_angle_ca_c']*step_gamma).detach()
            CA_C_1N2 = (CA_C_1N2-grad['bond_angle_c_n']*step_gamma).detach()

            phi2 = utils.modulo_with_wrapped_range(phi2, range_min=-torch.pi, range_max=torch.pi)
            phi2 = utils.modulo_with_wrapped_range(phi2, range_min=-torch.pi, range_max=torch.pi)
            omega2 = utils.modulo_with_wrapped_range(omega2, range_min=-torch.pi, range_max=torch.pi)
            C_1N_1CA2 = utils.modulo_with_wrapped_range(C_1N_1CA2, range_min=-torch.pi, range_max=torch.pi)
            tau2 = utils.modulo_with_wrapped_range(tau2, range_min=-torch.pi, range_max=torch.pi)
            CA_C_1N2 = utils.modulo_with_wrapped_range(CA_C_1N2, range_min=-torch.pi, range_max=torch.pi)
            self.data_builder.set_values(phi2, psi2, omega2, bond_angle_n_ca = C_1N_1CA2, bond_angle_ca_c = tau2, bond_angle_c_n = CA_C_1N2, init_coords=start_coords, max_length = max_length)
        

        # trans & rot
        coord = self.data_builder.cartesian_coords()
        batch = start_idx.shape[0]
        coord_reshape = coord.reshape(batch,-1,3,3)
        coord = self.data_builder.transform(coord, coord_reshape[:,0], coord_reshape[idx, end_idx-start_idx], start_coords, end_coords)

        coord = coord.reshape(coord.shape[0],-1,3,3)

        for b in range(start_idx.shape[0]):
            coords[b, start_idx[b]+1:end_idx[b]] = coord[b,1:end_idx[b]-start_idx[b]]
        return coords
    
    def compute_grad(self, input, coords, start_idx, end_idx):
        idx = torch.arange(input.shape[0], device=input.device)
        start_coords = coords[idx,start_idx]
        end_coords = coords[idx,end_idx]
        max_length = (end_idx-start_idx).max()+1
        phi, psi, omega, tau, CA_C_1N, C_1N_1CA = torch.split(input,1, dim=-1)
        
        phi2 = []
        psi2 = []
        omega2 = []
        C_1N_1CA2 = []
        tau2 = []
        CA_C_1N2 = []
        for i in range(max_length.item()):
            idx2 = torch.clamp(start_idx+i,max=phi.shape[1]-1) # next
            phi2.append(phi[idx, idx2])
            psi2.append(psi[idx, idx2])
            omega2.append(omega[idx, idx2])
            C_1N_1CA2.append(C_1N_1CA[idx, idx2])
            tau2.append(tau[idx, idx2])
            CA_C_1N2.append(CA_C_1N[idx, idx2])
        phi2 = torch.concat(phi2, dim=1) # C_(i-1),N_i,CA_i,C_i 
        psi2 = torch.concat(psi2, dim=1) # N_i,CA_i,C_i,N_(i+1)
        omega2 = torch.concat(omega2, dim=1) # CA_(i-1),C_(i-1),N_i,CA_i
        C_1N_1CA2 = torch.concat(C_1N_1CA2, dim=1) # C_i,N_(i+1),CA_(i+1)
        tau2 = torch.concat(tau2, dim=1) #  N_(i+1),CA_(i+1),C_(i+1)
        CA_C_1N2 = torch.concat(CA_C_1N2, dim=1) # CA_i,C_i,N_(i+1)
        
        
        self.data_builder.set_values(
                    phi2.detach(), 
                    psi2.detach(), 
                    omega2.detach(), 
                    bond_angle_n_ca = C_1N_1CA2.detach(), 
                    bond_angle_ca_c = tau2.detach(), 
                    bond_angle_c_n = CA_C_1N2.detach(), 
                    init_coords=start_coords.detach(), 
                    max_length = max_length)
        
        grad = self.data_builder.Gradient(start_coords, end_coords, end_idx-start_idx)
        
        grad = torch.stack([grad['phi'], 
                     grad['psi'], 
                     grad['omega'],
                     grad['bond_angle_ca_c'], 
                     grad['bond_angle_c_n'], 
                     grad['bond_angle_n_ca']], 
                     dim=-1)
        
        full_grad = torch.zeros_like(input)
        # noise_mask = torch.zeros(input.shape[0],input.shape[1], device=input.device)==1
        for i in range(max_length.item()):
            idx2 = torch.clamp(start_idx+i,max=phi.shape[1]-1) 
            full_grad[idx,idx2] = grad[:,i]
            # noise_mask[idx, idx2] = True
        full_grad[idx, start_idx] = 0
        full_grad[idx, end_idx] = 0
        
        full_grad[torch.isnan(full_grad)] = 0
        # noise_mask[idx, start_idx] = False
        # noise_mask[idx, end_idx] = False
        return full_grad
    
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
        
        x_s = dot_z_s*alpha_s[:,None] #+ sigma_s[:,None]*torch.randn_like(angles)
        x_t = dot_z_t*alpha_t[:,None] + sigma_t[:,None]*torch.randn_like(angles)
        
        # mask data
        coords = coords*(~unknown_mask[...,None])
        x_s = x_s*unknown_mask + angles*(~unknown_mask)
        x_t = x_t*unknown_mask + angles*(~unknown_mask)
        dot_z_s = dot_z_s*unknown_mask + b_s[:,None]*angles*(~unknown_mask)
        # 模型预测dot_z_s
        return dot_z_s, x_s, x_t, coords
    
    def forward(
        self,
        inputs: torch.Tensor,
        coords,
        timestep: torch.Tensor,  # Tensor of shape batch_length with time indices
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        seqs: torch.Tensor, 
        unknown_mask: torch.Tensor, 
        start_idx, 
        end_idx,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        t1 = time.time()


        num = end_idx - start_idx
        num_embed = self.num_embedding(num)
        idx = torch.arange(start_idx.shape[0], device = start_idx.device)
        length = torch.norm(coords[idx, end_idx,1,:] - coords[idx, start_idx,1,:], dim=1)
        # length_embed = self.length_embedding(_rbf(length, 32).squeeze())

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
        inputs_with_time = (inputs_upscaled + time_encoded + seq_encoded)* torch.sigmoid(mask_encoded)

        sequence_output = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            # head_mask=head_mask,
            output_attentions=True,
            return_dict=return_dict,
            att_bias = None
        )

        noise = self.token_decoder(sequence_output)
        
        pred = utils.modulo_with_wrapped_range(inputs + noise, -torch.pi, torch.pi)

        t2 = time.time()
        # print(t2-t1)
        return pred

