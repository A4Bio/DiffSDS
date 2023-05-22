import time
import torch
import math
import torch.nn as nn
import argparse
import typing
from transformers import BertConfig
from modules.modeling_bert_DiffSDS import (
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


def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

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

def Slerp(v0, v1, t):
    unit_vec = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
    v0 = unit_vec(v0)
    v1 = unit_vec(v1)
    dot = torch.sum(v0 * v1, dim=-1)
    
    # Calculate initial angle between v0 and v1
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega)
    
    # Angle at timestep t
    s0 = torch.sin((1-t)*omega) / sin_omega
    s1 = torch.sin(t*omega) / sin_omega
    v2 = s0[...,None] * v0 + s1[...,None] * v1
    return v2


class DiffSDS_model(BertPreTrainedModel):
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
        
        self.start_pos_embedding = nn.Embedding(2*128,config.hidden_size)
        self.end_pos_embedding = nn.Embedding(2*128,config.hidden_size)

        # Set up the network to project token representation to our four outputs
        if decoder == "linear":
            self.token_decoder = nn.Linear(config.hidden_size, n_inputs)
        elif decoder == "mlp":
            self.token_decoder = AnglesPredictor(config.hidden_size, n_inputs)
        else:
            raise ValueError(f"Unrecognized decoder: {decoder}")
    
        self.vector_decoder = AnglesPredictor(config.hidden_size, 9)

        # Set up the time embedder
        if time_encoding == "gaussian_fourier":
            self.time_embed = GaussianFourierProjection(config.hidden_size)
        elif time_encoding == "sinusoidal":
            self.time_embed = SinusoidalPositionEmbeddings(config.hidden_size)
        else:
            raise ValueError(f"Unknown time encoding: {time_encoding}")

        # self.vecNN = Structural_module()

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
        
        phi2 = torch.stack(phi2, dim=1) # C_(i-1),N_i,CA_i,C_i 
        psi2 = torch.stack(psi2, dim=1) # N_i,CA_i,C_i,N_(i+1)
        omega2 = torch.stack(omega2, dim=1) # CA_(i-1),C_(i-1),N_i,CA_i
        C_1N_1CA2 = torch.stack(C_1N_1CA2, dim=1) # C_i,N_(i+1),CA_(i+1)
        tau2 = torch.stack(tau2, dim=1) #  N_(i+1),CA_(i+1),C_(i+1)
        CA_C_1N2 = torch.stack(CA_C_1N2, dim=1) # CA_i,C_i,N_(i+1)

        self.data_builder.set_values(phi2.detach(), psi2.detach(), omega2.detach(), bond_angle_n_ca = C_1N_1CA2.detach(), bond_angle_ca_c = tau2.detach(), bond_angle_c_n = CA_C_1N2.detach(), init_coords=start_coords.detach(), max_length = max_length)

        # # trans & rot
        coord = self.data_builder.cartesian_coords()
        batch = start_idx.shape[0]
        coord_reshape = coord.reshape(batch,-1,3,3)
        coord = self.data_builder.transform(coord, coord_reshape[:,0], coord_reshape[idx, end_idx-start_idx], start_coords, end_coords)

        coord = coord.reshape(coord.shape[0],-1,3,3)

        for b in range(start_idx.shape[0]): # keep the start and end as proir
            coords[b, start_idx[b]+1:end_idx[b]] = coord[b,1:end_idx[b]-start_idx[b]]
        return coords
    
    
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
    def get_xs_xt(self, angles, coords, unknown_mask, start_idx, end_idx, timesteps, mode = "vector"):
        if mode == "colddiff":
            # 1. params
            b_t, alpha_t, sigma_t = self.get_params(timesteps)
            # b_s, alpha_s, sigma_s = self.get_params(timesteps-1)
            # SNR_s = (alpha_s**2/sigma_s**2)
            # SNR_t = (alpha_t**2/sigma_t**2)
            # loss_weight = 0.5*(SNR_s-SNR_t)
            # loss_weight = (timesteps/self.T+1)
            loss_weight = None
            z_T = torch.rand_like(angles)
            
            z_t = (1-b_t[:,None])*z_T + b_t[:,None]*angles # 此处b_t恒等于1
            
            x_t = z_t*alpha_t[:,None] + sigma_t[:,None]*torch.randn_like(angles)
            
            # mask data
            coords = coords*(~unknown_mask[...,None])
            x_t = x_t*unknown_mask + angles*(~unknown_mask)
            return angles, x_t, coords, loss_weight
        
    @classmethod
    def diff(self, x_0, timesteps, z_T):
        b_t, alpha_t, sigma_t = self.get_params(timesteps)
        b_t = b_t.reshape(-1,1,1)
        alpha_t = alpha_t.reshape(-1,1,1)
        sigma_t = sigma_t.reshape(-1,1,1)
        z_t = (1-b_t)*z_T + b_t*x_0 # average of Guasian disdribution, 
        x_t = z_t*alpha_t + sigma_t*torch.randn_like(x_0) # alpha_t=1, 
        return x_t
    
    def dihedral(self, v1, v2, v3):
        unit_vec = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        v1 = unit_vec(v1)
        v2 = unit_vec(v2)
        v3 = unit_vec(v3)
        
        n1 = torch.cross(v1, v2, dim=-1)
        n2 = torch.cross(v2, v3, dim=-1)
        x = torch.sum(n1*n2, dim=-1)
        y = torch.sum(torch.cross(n1,n2, dim=-1)*v2, dim=-1)
        return torch.arctan2(y,x)
    
    def angle(self, v1, v2):
        unit_vec = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        v1 = unit_vec(v1)
        v2 = unit_vec(v2)
        cos = torch.sum(v1*v2, dim=-1)
        cos = torch.clamp(cos, -0.99999,0.99999)
        return torch.arccos(cos)
    
    def get_vector(self, C_Ni, Ni_CAi, CAi_Ci, start_name="CA", end_name="CA", start_idx=80, end_idx=81):
        vector_list = []
        for b_idx in range(start_idx.shape[0]):
            s_idx = start_idx[b_idx]
            e_idx = end_idx[b_idx]
            vector = torch.sum(CAi_Ci[b_idx,s_idx:e_idx],dim=0)*1.54 +\
            torch.sum(C_Ni[b_idx,s_idx+1:e_idx+1],dim=0)*1.34 +\
            torch.sum(Ni_CAi[b_idx,s_idx+1:e_idx+1],dim=0)*1.46
            
            if start_name == "C":
                vector = vector - CAi_Ci[b_idx, s_idx]*1.54
            if start_name == 'N':
                vector = vector + Ni_CAi[b_idx, s_idx]*1.46
            if end_name == "C":
                vector = vector + CAi_Ci[b_idx, e_idx]*1.54
            if end_name == "N":
                vector = vector - Ni_CAi[b_idx, e_idx]*1.46
            
            vector_list.append(vector)
        return torch.stack(vector_list)
    
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

        B, N, _ = inputs.shape
        num = end_idx - start_idx
        num_embed = self.num_embedding(num)
        idx = torch.arange(start_idx.shape[0], device = start_idx.device)
        length = torch.norm(coords[idx, end_idx,1,:] - coords[idx, start_idx,1,:], dim=1)
        length_embed = self.length_embedding(_rbf(length, 32)).reshape(B,-1)
        
        seq_length = inputs.size()[1]
        position_ids_start = torch.arange(seq_length, dtype=torch.long, device=inputs.device).view(1, -1)
        position_ids_end = torch.arange(seq_length, dtype=torch.long, device=inputs.device).view(1, -1)
        
        position_ids_start = (position_ids_start - start_idx[:,None])+seq_length
        position_ids_end = (position_ids_end - end_idx[:,None])+seq_length
        
        start_embedding = self.start_pos_embedding(position_ids_start)
        end_embedding = self.end_pos_embedding(position_ids_end)

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
        seq_encoded = self.seq_embedding(seqs).reshape(B, N,-1)
        mask_encoded = self.mask_embedding(unknown_mask.long()).squeeze()

        # timestep is (batch, 1), squeeze to (batch,)
        # embedding gets to (batch, embed_dim) -> unsqueee to (batch, 1, dim)
        time_encoded = self.time_embed(timestep.squeeze(dim=-1)).unsqueeze(1) # [64, 1, 384]
        inputs_with_time = (inputs_upscaled + time_encoded + seq_encoded + length_embed[:,None]+num_embed[:,None]+start_embedding+end_embedding)* torch.sigmoid(mask_encoded)

        sequence_output = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            # head_mask=head_mask,
            output_attentions=True,
            return_dict=return_dict,
            att_bias = None
        )
        first_vector = True
        if first_vector:
            vectors = self.vector_decoder(sequence_output)
            B,N,_ = vectors.shape
            vectors = vectors.reshape(B,-1,3)
            vectors = vectors/torch.norm(vectors, dim=-1, keepdim=True)
            
            C_Ni = vectors[:,0::3]
            Ni_CAi = vectors[:,1::3]
            CAi_Ci = vectors[:,2::3] # CA80_C80*1.54 + C80_N81*1.34 + N81_CA81*1.46
            
            phi = self.dihedral(C_Ni, Ni_CAi, CAi_Ci) # OK
            psi = self.dihedral(Ni_CAi, CAi_Ci, torch.roll(C_Ni, -1, dims=1)) # OK
            omega = self.dihedral(CAi_Ci, torch.roll(C_Ni, -1, dims=1), torch.roll(Ni_CAi, -1, dims=1)) # OK
            tau = self.angle(-torch.roll(Ni_CAi, -1, dims=1), torch.roll(CAi_Ci, -1, dims=1)) # OK
            CA_C_1N = self.angle(-CAi_Ci, torch.roll(C_Ni,-1,dims=1)) # OK
            C_1N_1CA = self.angle(-torch.roll(C_Ni,-1,dims=1), torch.roll(Ni_CAi,-1,dims=1)) # OK
            
            pred = torch.stack([phi, psi, omega, tau, CA_C_1N, C_1N_1CA], dim=-1)
            
            N_N = self.get_vector(C_Ni, Ni_CAi, CAi_Ci, "N", "N", start_idx, end_idx)
            CA_CA = self.get_vector(C_Ni, Ni_CAi, CAi_Ci, "CA", "CA", start_idx, end_idx)
            C_C = self.get_vector(C_Ni, Ni_CAi, CAi_Ci, "C", "C", start_idx, end_idx)
            vectors = torch.stack([N_N, C_C, CA_CA], dim=1)
            
            # overlapping loss
            rand_idx = torch.cat([torch.randint(start_idx[i], end_idx[i],(1,)) for i in range(start_idx.shape[0])]).to(pred.device)
            
            anchor = coords[idx, start_idx,1] + self.get_vector(C_Ni, Ni_CAi, CAi_Ci, "CA", "CA", start_idx, rand_idx)
            
            dist = (anchor[:,None,None] - coords[:,None,:,1,:]).norm(dim=-1)[:,0]
            rel_idx = rand_idx[:,None] - torch.arange(coords.shape[1], device=coords.device)[None]
            select_mask = (~unknown_mask[:,:,0]* attention_mask*(rel_idx>10))
            dist = dist*select_mask + 10000*(1-select_mask)
            dist = dist.min(dim=1)[0]
        else:
            pred = self.token_decoder(sequence_output)
            phi, psi, omega, tau, CA_C_1N, C_1N_1CA = pred.split(1,dim=-1)
            coords_pred = self.pred_coord(coords, start_idx-1, end_idx+1, phi, psi, omega, tau, CA_C_1N, C_1N_1CA)
            idx = torch.arange(start_idx.shape[0], device=coords.device)
            vectors = coords_pred[idx, end_idx] - coords_pred[idx, start_idx]
            
            rand_idx = torch.cat([torch.randint(start_idx[i], end_idx[i],(1,)) for i in range(start_idx.shape[0])]).to(pred.device)
            
            # print(coords[idx,rand_idx,1].shape,  coords[:,None,:,1,:].shape)
            dist = (coords[idx,rand_idx,1][:,None,None] - coords[:,None,:,1,:]).norm(dim=-1)[:,0]
            rel_idx = rand_idx[:,None] - torch.arange(coords.shape[1], device=coords.device)[None]
            select_mask = (~unknown_mask[:,:,0]* attention_mask*(rel_idx>10))
            dist = dist*select_mask + 10000*(1-select_mask)
            dist = dist.min(dim=1)[0]
            
        return pred, vectors, dist


class Structural_module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv2d(1,64, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(64,64, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(64,1, 3, padding=1))
        
    
    def forward(self, vectors):
        B, N, _ = vectors.shape
        q = vectors
        A = torch.einsum("bnd,bmd->bnm", vectors, vectors)
        attn = self.CNN(A.reshape(B,1,N,N)).reshape(B,N,N)
        # attn = torch.softmax(attn, dim=-1)
        out = attn@q
        return out