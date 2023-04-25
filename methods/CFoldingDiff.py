from curses import init_pair
from unittest import result
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
from typing import *
from methods.base_method import Base_method
from utils.main_utils import select_feats2mat
from utils.model_utils import T, cuda
from models import CBertForDiffusion
from transformers import BertConfig
from utils.loss import radian_smooth_l1_loss
from utils import beta_schedules
import utils
from utils.nerf import TorchNERFBuilder
from utils.metrics import LogMetric


class CFoldingDiff(Base_method):
    def __init__(self, args, device, steps_per_epoch, mean_angle, distributed=False):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.distributed = distributed
        self.model = self._build_model()
        
        self.mean_angle = mean_angle
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.builder = TorchNERFBuilder(0)

    def _build_model(self):
        cfg = BertConfig(
            max_position_embeddings=self.args.max_seq_len,
            num_attention_heads=self.args.num_heads,
            hidden_size=self.args.hidden_size,
            intermediate_size=self.args.intermediate_size,
            num_hidden_layers=self.args.num_hidden_layers,
            position_embedding_type=self.args.position_embedding_type,
            hidden_dropout_prob=self.args.dropout_p,
            attention_probs_dropout_prob=self.args.dropout_p,
            use_cache=False,
        )

        model = CBertForDiffusion(cfg,
                                time_encoding='gaussian_fourier',
                                decoder='mlp',
                                ft_is_angular=[True, True, True, True, True, True],
                                ft_names=['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA'],
                                use_grad=self.args.use_grad
                                ).to(self.device)
        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)
        
        return model
    
    def cal_length_loss(self, pred_dot_z_s, mean_angles, start_idxs, end_idxs, coords):
        pred_angles = pred_dot_z_s + torch.from_numpy(mean_angles).to(pred_dot_z_s.device)
        pred_angles = utils.modulo_with_wrapped_range(pred_angles, range_min=-torch.pi, range_max=torch.pi)
        phi, psi, omega, tau, CA_C_1N, C_1N_1CA = pred_angles.split(1,dim=-1)
        len_loss_list = []
        for i in range(phi.shape[0]):
            start_idx = start_idxs[i]
            end_idx = end_idxs[i]
            self.builder.set_values( phi[i,start_idx:end_idx].reshape(1,-1), 
                        psi[i,start_idx:end_idx].reshape(1,-1), 
                        omega[i,start_idx:].reshape(1,-1), 
                        bond_angle_n_ca = C_1N_1CA[i,start_idx:end_idx].reshape(1,-1), 
                        bond_angle_ca_c = tau[i,start_idx:end_idx].reshape(1,-1),
                        bond_angle_c_n = CA_C_1N[i,start_idx:end_idx].reshape(1,-1),
                        init_coords=coords[i,start_idx],
                        max_length = 1000)
            pred_coords = self.builder.cartesian_coords().reshape(-1,3,3)
            
            true_vectors = coords[i, end_idx] - coords[i, start_idx]
            pred_vectors = pred_coords[-1]-pred_coords[0]
            
            len_loss = torch.sum((true_vectors.norm(dim=-1) - pred_vectors.norm(dim=-1))**2, dim=-1).mean()
            len_loss_list.append(len_loss)
        len_loss = torch.stack(len_loss_list).mean()
        return len_loss

    def forward_loss(self, batch):
        angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=self.model.device)
            
        dot_z_s, x_s, x_t, coords = CBertForDiffusion.get_xs_xt(angles, coords, unknown_mask, start_idx, end_idx, timestamps)
        pred_dot_z_s = self.model(x_t, coords, timestamps, attn_mask, position_ids, seqs, unknown_mask, start_idx, end_idx)
        angle_loss = self._get_loss_terms(pred_dot_z_s, dot_z_s, attn_mask, unknown_mask)
        loss = angle_loss.mean()
        len_loss = torch.zeros_like(loss)
        overlap_loss = torch.zeros_like(loss)
        
        return loss, angle_loss, len_loss, overlap_loss
        
    
    def train_one_epoch(self, train_loader):
        LogLoss = LogMetric(torch.zeros(1))
        LogAngle = LogMetric(torch.zeros(6))
        
        self.model.train()
        self.train_loader = train_loader
        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            self.optimizer.zero_grad()
            loss, angle_loss, len_loss, overlap_loss = self.forward_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            LogAngle(angle_loss.detach().cpu(), 1)
            LogLoss(loss.detach().cpu(), 1)
            
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            
        self.scheduler.step()
        train_loss = LogLoss.val.item()/LogLoss.total
        phi, psi, omega, tau, CA_C_1N, C_1N_1CA =  LogAngle.val/LogAngle.total 
        
        return {"train_loss":train_loss.item(),
                "train_phi": phi.item(), 
                "train_psi": psi.item(), 
                "train_omega": omega.item(), 
                "train_tau": tau.item(), 
                "train_CA_C_1N": CA_C_1N.item(), 
                "train_C_1N_1CA": C_1N_1CA.item()}
    

    def valid_one_epoch(self, valid_loader):
        LogLoss = LogMetric(torch.zeros(1))
        LogAngle = LogMetric(torch.zeros(6))
        self.model.eval()
        valid_pbar = tqdm(valid_loader)

        
        for batch in valid_pbar:
            with torch.no_grad():
                loss, angle_loss, len_loss, overlap_loss = self.forward_loss(batch)
                LogAngle(angle_loss.detach().cpu(), 1)
                LogLoss(loss.detach().cpu(), 1)
            
            valid_pbar.set_description('valid loss: {:.4f}'.format(loss.item()))
                
        valid_loss = LogLoss.val.item()/LogLoss.total
        phi, psi, omega, tau, CA_C_1N, C_1N_1CA =  LogAngle.val/LogAngle.total 
        return {
                "valid_loss": valid_loss.item(),
                "valid_phi": phi.item(), 
                "valid_psi": psi.item(), 
                "valid_omega": omega.item(), 
                "valid_tau": tau.item(), 
                "valid_CA_C_1N": CA_C_1N.item(), 
                "valid_C_1N_1CA": C_1N_1CA.item()}



    def test_one_epoch(self, test_loader):
        self.model.eval()
        step_angle_loss2 = []
        for batch in tqdm(test_loader):
            angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=self.model.device)
            
            timestamps = 1000
            pred_angles, step_angle_loss = self.sampling(angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx ,  mode='colddiff')
            step_angle_loss2.extend(step_angle_loss)
            test_all = torch.stack(step_angle_loss)


        test_all = torch.stack(step_angle_loss)
        test_angle_loss = test_all.mean(dim=0)
        return test_angle_loss

    def _get_loss_terms(self, predicted_noise, known_noise, attn_mask, unknown_mask=None):
        """
        Returns the loss terms for the model. Length of the returned list
        is equivalent to the number of features we are fitting to.
        """
        if unknown_mask is None:
            unmask_idx = torch.where(attn_mask)
        else:
            unmask_idx = torch.where(attn_mask*unknown_mask.squeeze())
        self.circle_lambda = 0
        loss_terms = []
        for i in range(known_noise.shape[-1]):
            l = radian_smooth_l1_loss(
                    predicted_noise[unmask_idx[0], unmask_idx[1], i],
                    known_noise[unmask_idx[0], unmask_idx[1], i],
                    circle_penalty=self.circle_lambda,
                )

            loss_terms.append(l)
        return torch.stack(loss_terms)
    
    def sampling(self, angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx, train_dset):
        def to_real_angle(x):
            x = x + torch.from_numpy(train_dset.dset.get_masked_means()).to(results.device)
            x = utils.modulo_with_wrapped_range(x, range_min=-torch.pi, range_max=torch.pi)
            return x
        
        noise = torch.randn_like(angles).to(self.device)
        noise = utils.modulo_with_wrapped_range(noise, -np.pi, np.pi)
        input = noise*(unknown_mask) + angles*(~unknown_mask)
        seqs = 20*(unknown_mask) + seqs*(~unknown_mask)
        coords = coords*(~unknown_mask[...,None])
        b = noise.shape[0]  

        error_list = []
        results = []
        for i in tqdm(
            reversed(range(0, timestamps)), desc="sampling loop time step", total=timestamps
        ):
            input = p_sample(
                    model=self.model,
                    x=input,
                    coords = coords,
                    seqs = seqs,
                    position_ids = position_ids,
                    attn_mask = attn_mask,
                    unknown_mask = unknown_mask,
                    start_idx = start_idx,
                    end_idx = end_idx,
                    t=torch.full((b,), i, device=self.device, dtype=torch.long),  # time vector
                    t_index=i,
                )
            
            input = utils.modulo_with_wrapped_range(input, range_min=-torch.pi, range_max=torch.pi)
            
            error = ((angles - input)*unknown_mask).sum(dim=(0,1))/unknown_mask.sum(dim=(0,1))
            # print(error)
            error_list.append(error)

            results.append(input)
        results = torch.stack(results)[-1,...]
        error_list = torch.stack(error_list)
        
        # means = torch.from_numpy(train_dset.dset.get_masked_means()).to(results.device)
        
        # phi_true = torch.masked_select(angles[:,:,0], unknown_mask[:,:,0])
        # psi_true = torch.masked_select(angles[:,:,1], unknown_mask[:,:,0])
        # omega_true = torch.masked_select(angles[:,:,2], unknown_mask[:,:,0])
        # tau_true = torch.masked_select(angles[:,:,3], unknown_mask[:,:,0])
        # CA_C_1N_true = torch.masked_select(angles[:,:,4], unknown_mask[:,:,0])
        # C_1N_1CA_true = torch.masked_select(angles[:,:,5], unknown_mask[:,:,0])
        
        # phi_pred = torch.masked_select(results[:,:,0], unknown_mask[:,:,0])
        # psi_pred = torch.masked_select(results[:,:,1], unknown_mask[:,:,0])
        # omega_pred = torch.masked_select(results[:,:,2], unknown_mask[:,:,0])
        # tau_pred = torch.masked_select(results[:,:,3], unknown_mask[:,:,0])
        # CA_C_1N_pred = torch.masked_select(results[:,:,4], unknown_mask[:,:,0])
        # C_1N_1CA_pred = torch.masked_select(results[:,:,5], unknown_mask[:,:,0])
        
        # torch.save({"phi_true": phi_true.cpu(), 
        #             "psi_true":psi_true.cpu(),
        #             "omega_true":omega_true.cpu(),
        #             "tau_true":tau_true.cpu(),
        #             "CA_C_1N_true":CA_C_1N_true.cpu(),
        #             "C_1N_1CA_true": C_1N_1CA_true.cpu(),
        #             "phi_pred":phi_pred.cpu(),
        #             "psi_pred":psi_pred.cpu(),
        #             "omega_pred":omega_pred.cpu(),
        #             "tau_pred":tau_pred.cpu(),
        #             "CA_C_1N_pred":CA_C_1N_pred.cpu(),
        #             "C_1N_1CA_pred":C_1N_1CA_pred.cpu(),
        #             "means":means.cpu()
        #             }, "/gaozhangyang/experiments/ProreinBinder/results/cfoldingdiff_angles.pt")
        
        torch.save({"error_curve": error_list.cpu()}, 
                   "/gaozhangyang/experiments/ProreinBinder/results/error_curve_cfolddiff.pt")
        
        results = results + torch.from_numpy(train_dset.dset.get_masked_means()).to(results.device)
        results = utils.modulo_with_wrapped_range(results, range_min=-torch.pi, range_max=torch.pi)
        


        return results, error_list
    
    
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    coords,
    seqs: torch.Tensor,
    position_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    unknown_mask: torch.Tensor,
    start_idx,
    end_idx,
    t: torch.Tensor,
    t_index: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    with torch.no_grad():
        pred_z_s = model(x, coords, t, attn_mask, position_ids, seqs, unknown_mask, start_idx, end_idx)
        b_s, alpha_s, sigma_s = CBertForDiffusion.get_params(t-1)
        pred_x_s = pred_z_s*alpha_s[:,None,None]
    pred_x_s = pred_x_s*unknown_mask + x*(~unknown_mask)
    return pred_x_s