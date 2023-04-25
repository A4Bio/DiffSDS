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
from models import  BertForDiffusion
from transformers import BertConfig
from utils.loss import radian_smooth_l1_loss
from utils import beta_schedules
import utils



@torch.no_grad()
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
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    # Equation 11 in the paper

    model_mean = sqrt_recip_alphas_t * (
        x
        - betas_t
        * model(x, t, attn_mask, position_ids, seqs, unknown_mask, start_idx, end_idx, betas)
    )


    if t_index == 0:
        model_mean = model_mean*unknown_mask + x*(~unknown_mask)
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        model_mean = model_mean + torch.sqrt(posterior_variance_t) * noise
        model_mean = model_mean*unknown_mask + x*(~unknown_mask)
        return model_mean



class FoldingDiff(Base_method):
    def __init__(self, args, device, steps_per_epoch, mean_angle, distributed=False):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.distributed = distributed
        self.mean_angle = mean_angle
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

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
        
        model = BertForDiffusion(cfg,
                                time_encoding='gaussian_fourier',
                                decoder='mlp',
                                ft_is_angular=[True, True, True, True, True, True],
                                ft_names=['phi', 'psi', 'omega', 'tau', 'CA:C:1N', 'C:1N:1CA'],
                                ).to(self.device)
        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True)
        return model

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_sum, train_weights = 0., 0.
        self.train_loader = train_loader
        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            self.optimizer.zero_grad()
            angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=self.model.device)
            
            dot_z_s, noise, x_t, coords = BertForDiffusion.get_xs_xt(angles, coords, unknown_mask, start_idx, end_idx, timestamps)
        
            predicted_noise = self.model(x_t, timestamps, attn_mask, position_ids, seqs, unknown_mask)

            loss_terms = self._get_loss_terms(predicted_noise, noise, attn_mask, unknown_mask)
            loss = loss_terms.mean()
            
            loss.backward()
            train_sum += loss
            train_weights  += 1
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

            
        self.scheduler.step()
        train_loss = train_sum / train_weights
        return {"train_loss":train_loss}

    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        valid_sum, valid_weights = 0., 0.
        valid_pbar = tqdm(valid_loader)

        with torch.no_grad():
            for batch in valid_pbar:
                angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=self.model.device)
            
                dot_z_s, noise, x_t, coords = BertForDiffusion.get_xs_xt(angles, coords, unknown_mask, start_idx, end_idx, timestamps)
            
                predicted_noise = self.model(x_t, timestamps, attn_mask, position_ids, seqs, unknown_mask)
                
                loss_terms = self._get_loss_terms(predicted_noise, noise, attn_mask, unknown_mask)
                loss = loss_terms.mean()

                valid_sum += loss
                valid_weights  += 1
                valid_pbar.set_description('valid loss: {:.4f}'.format(loss.item()))
                
            valid_loss = valid_sum / valid_weights
        return {"valid_loss":valid_loss}

    def test_one_epoch(self, test_loader):
        self.model.eval()
        test_pbar = tqdm(test_loader)
        test_sum, test_weights = 0., 0.
        with torch.no_grad():
            for batch in test_pbar:
                angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=self.model.device)
            
                dot_z_s, noise, x_t, coords = BertForDiffusion.get_xs_xt(angles, coords, unknown_mask, start_idx, end_idx, timestamps)
            
                predicted_noise = self.model(x_t, timestamps, attn_mask, position_ids, seqs, unknown_mask)
                
                loss_terms = self._get_loss_terms(predicted_noise, noise, attn_mask, unknown_mask)
                
                loss = loss_terms.mean()

                test_sum += loss
                test_weights  += 1

        
        test_loss = test_sum / test_weights
        return {"test_loss":test_loss}

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
    
    def sampling(self, angles, coords, corrupted, known_noise, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx, train_dset):
        noise = torch.randn_like(angles).to(self.device)
        noise = utils.modulo_with_wrapped_range(noise, -np.pi, np.pi)
        input = noise*(unknown_mask) + angles*(~unknown_mask)
        seqs = 20*(unknown_mask) + seqs*(~unknown_mask)
        coords = coords*(~unknown_mask[...,None])
        b = noise.shape[0]  #input[0][attn_mask[0].bool()] == angles[0][attn_mask[0].bool()]

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
                    betas=train_dset.alpha_beta_terms["betas"],
                )
            
            input = utils.modulo_with_wrapped_range(input, range_min=-torch.pi, range_max=torch.pi)

            results.append(input)
        results = torch.stack(results)[-1,...]
        results = results + torch.from_numpy(train_dset.dset.get_masked_means()).to(results.device)
        results = utils.modulo_with_wrapped_range(results, range_min=-torch.pi, range_max=torch.pi)

        return results