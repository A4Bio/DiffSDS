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
from models import DiffSDS_model
from transformers import BertConfig
from utils.loss import radian_smooth_l1_loss
from utils import beta_schedules
import utils
from utils.metrics import LogMetric

class DiffSDS(Base_method):
    def __init__(self, args, device, steps_per_epoch, mean_angles, distributed=True):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.distributed = distributed
        self.model = self._build_model()
        self.mean_angles = mean_angles
        

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.angle_weights = torch.tensor(list(map(float, self.config["W_angle"].split(",")))).to(device)

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

        model = DiffSDS_model(cfg,
                                time_encoding='gaussian_fourier',
                                decoder='mlp',
                                ft_is_angular=[False, False, False, False, False, False],
                                ft_names=['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA'],
                                use_grad=self.args.use_grad
                                ).to(self.device)
        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)
        return model
    
    def forward_loss(self, batch):
        angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=self.model.device)
            

        x_0, x_t, coords, loss_weight = DiffSDS_model.get_xs_xt(angles, coords, unknown_mask, start_idx, end_idx, timestamps, mode=self.args.mode)
        idx = torch.arange(x_0.shape[0], device = x_0.device)
        true_vectors = coords[idx, end_idx] - coords[idx, start_idx]
    
        pred_x_0, vectors, dist = self.model(x_t, coords, timestamps, attn_mask, position_ids, seqs, unknown_mask, start_idx, end_idx)

        
        angle_loss = self._get_loss_terms_angle(pred_x_0, x_0, attn_mask, unknown_mask, loss_weight)
        angle_loss_reduced = (angle_loss*self.angle_weights).mean()
        len_loss = torch.sum((true_vectors.norm(dim=-1) - vectors.norm(dim=-1))**2, dim=-1).mean()
        overlap_loss = torch.clamp(torch.exp(-0.8*dist) - np.exp(-0.8*3), 0).mean()
        
        loss_final = angle_loss_reduced + len_loss*0.0001 + 10*overlap_loss
        
        return loss_final, angle_loss, len_loss, overlap_loss

    
    def train_one_epoch(self, train_loader):
        self.model.train()
        LogAngle = LogMetric(torch.zeros(6))
        LogLoss = LogMetric(torch.zeros(1))
        LogLen = LogMetric(torch.zeros(1))
        LogOverlap = LogMetric(torch.zeros(1))
        
        
        self.train_loader = train_loader
        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            self.optimizer.zero_grad()
            loss, angle_loss, len_loss, overlap_loss = self.forward_loss(batch)
            loss.backward()
            
            LogAngle(angle_loss.detach().cpu(), 1)
            LogLoss(loss.detach().cpu(), 1)
            LogLen(len_loss.detach().cpu(),1)
            LogOverlap(overlap_loss.detach().cpu(), 1)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            
        self.scheduler.step()
        
        train_loss = LogLoss.val.item()/LogLoss.total
        phi, psi, omega, tau, CA_C_1N, C_1N_1CA =  LogAngle.val/LogAngle.total 
        train_overlap = LogOverlap.val.item()/LogOverlap.total 
        
        return {"train_loss": train_loss,
                "train_phi": phi,
                "train_psi": psi,
                "train_omega": omega,
                "train_omega": omega,
                "train_tau": tau,
                "train_CA_C_1N": CA_C_1N,
                "train_C_1N_1CA": C_1N_1CA,
                "train_overlap": train_overlap}
    

    def valid_one_epoch(self, valid_loader):
        LogAngle = LogMetric(torch.zeros(6))
        LogLoss = LogMetric(torch.zeros(1))
        LogLen = LogMetric(torch.zeros(1))
        LogOverlap = LogMetric(torch.zeros(1))
        
        self.model.eval()
        valid_pbar = tqdm(valid_loader)

        for batch in valid_pbar:
            with torch.no_grad():
                loss, angle_loss, len_loss, overlap_loss = self.forward_loss(batch)
                LogAngle(angle_loss.detach().cpu(), 1)
                LogLoss(loss.detach().cpu(), 1)
                LogLen(len_loss.detach().cpu(),1)
                LogOverlap(overlap_loss.detach().cpu(), 1)
                
            
            valid_pbar.set_description('valid loss: {:.4f}'.format(loss.item()))

            
        valid_loss = LogLoss.val.item()/LogLoss.total
        phi, psi, omega, tau, CA_C_1N, C_1N_1CA =  LogAngle.val/LogAngle.total 
        valid_overlap = LogOverlap.val.item()/LogOverlap.total 
        return {"valid_loss": valid_loss,
                "valid_phi": phi,
                "valid_psi": psi,
                "valid_omega": omega,
                "valid_omega": omega,
                "valid_tau": tau,
                "valid_CA_C_1N": CA_C_1N,
                "valid_C_1N_1CA": C_1N_1CA,
                "valid_overlap": valid_overlap}

    def test_one_epoch(self, test_loader, train_loader):
        self.model.eval()
        step_angle_loss2 = []
        for batch in tqdm(test_loader):
            angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=self.model.device)
            raw_coords = coords.clone()
            timestamps = 1000
            
            pred_angles, step_angle_loss = self.sampling(angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx , train_loader.dataset, mode='colddiff')
            step_angle_loss2.extend(step_angle_loss)
            test_all = torch.stack(step_angle_loss)
            

        test_all = torch.stack(step_angle_loss)
        test_angle_loss = test_all.mean(dim=0)
        return test_angle_loss


    def _get_loss_terms_angle(self, predicted_noise, known_noise, attn_mask, unknown_mask=None, loss_weight = None):
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
            
            if loss_weight is not None:
                l = l*loss_weight[i]

            loss_terms.append(l)
        return torch.stack(loss_terms)
    

    
    def sampling(self, angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx, train_dset, mode):
        noise = torch.randn_like(angles).to(self.device)
        noise = utils.modulo_with_wrapped_range(noise, -np.pi, np.pi)
        input = noise*(unknown_mask) + angles*(~unknown_mask)
        seqs = 20*(unknown_mask) + seqs*(~unknown_mask)
        coords = coords*(~unknown_mask[...,None])
        b = noise.shape[0]  #input[0][attn_mask[0].bool()] == angles[0][attn_mask[0].bool()]

        error_list = []
        results = []
        for i in tqdm(
            reversed(range(0, timestamps)), desc=f"sampling loop time step", total=timestamps
        ):
            input, pred_angles = p_sample(
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
                    noise=noise,
                    mode = mode
                )
            error = ((angles - input)*unknown_mask).sum(dim=(0,1))/unknown_mask.sum(dim=(0,1))
            error_list.append(error)

            results.append(input)
        results = torch.stack(results)[-1,...]
        error_list = torch.stack(error_list)
        
        torch.save({"error_curve": error_list.cpu()}, 
                   "/gaozhangyang/experiments/ProreinBinder/results/error_curve_diffsds.pt")

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
    noise,
    mode
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    if mode == 'denoise':

        with torch.no_grad():
            x_s = model(x, coords, t, attn_mask, position_ids, seqs, unknown_mask, start_idx, end_idx)
    
    if mode=='colddiff':
        with torch.no_grad():
            hat_x_0, _, _ = model(x, coords, t, attn_mask, position_ids, seqs, unknown_mask, start_idx, end_idx)
        # x_noise_s = x - DiffSDS_model.diff(hat_x_0, t, noise) + DiffSDS_model.diff(hat_x_0, t-1, noise)
        # noise = torch.rand_like(noise) # 添加随机扰动
        x_noise_s = DiffSDS_model.diff(hat_x_0, t-1, noise) # t-->t-1
        # x_s = hat_x_0
        x_s = x_noise_s*unknown_mask + x*(~unknown_mask)
    return x_s, hat_x_0
