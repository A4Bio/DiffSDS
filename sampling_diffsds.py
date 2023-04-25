import torch
import os.path as osp
from parser import create_parser
import json

import warnings
warnings.filterwarnings('ignore')

import torch.backends.cudnn as cudnn
import random 
import numpy as np
from main import Exp
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(111)

# from constants import method_maps
from utils import Recorder
from utils.nerf import TorchNERFBuilder
from utils import *
from utils.angles_and_coords import (
    canonical_distances_and_dihedrals,
    EXHAUSTIVE_ANGLES,
    EXHAUSTIVE_DISTS,
    extract_backbone_coords,
    create_new_chain_nerf
)

import pandas as pd

if __name__ == '__main__':
    args = create_parser()
    args.method = 'DiffSDS'
    config = args.__dict__
    default_params = load_config(osp.join('./configs', args.method + '.py' if args.config_file is None else args.config_file))
    
    config.update(default_params)
    config['batch_size'] = 1000
    config['strict_test'] = True
    config["sampling"] = True
    mode = "colddiff"
    
    
    svpath = "/gaozhangyang/experiments/DiffSDS/results/DiffSDS_sampling/"
    

    mask_location = {}
    exp = Exp(args, distributed=False)
    
    params = torch.load('/gaozhangyang/experiments/DiffSDS/model_zoom/DiffSDS/checkpoint.pth', map_location=torch.device('cuda:0'))
    new_params = {}
    for key, val in params.items():
        new_params[key.replace("module.", "")] = val
    exp.method.model.load_state_dict(new_params)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> sampling  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    for batch in tqdm(exp.test_loader):
        angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=exp.method.model.device)
        raw_coords = coords.clone()
        timestamps = 1000
        for idx, key in enumerate(batch['key']):
            mask_location[key] = (start_idx[idx].item(), end_idx[idx].item())
            
        angles, step_angle_loss = exp.method.sampling(angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx , exp.train_loader.dataset, mode=mode)

        unknown_mask = unknown_mask.squeeze()
        
        phi, psi, omega, tau, CA_C_1N, C_1N_1CA = torch.split(angles, 1, dim=-1)
        
        pred_coords = coords*(~unknown_mask[...,None,None])
        pred_coords2 = exp.method.model.pred_coord(pred_coords.clone(), start_idx, end_idx, phi, psi, omega, C_1N_1CA, tau, CA_C_1N)
        



        for i in range(angles.shape[0]):
            if (attn_mask*unknown_mask)[i].sum()>0:                
                TorchNERFBuilder.sv2pdb(f"{svpath}/pred_{batch['key'][i]}.pdb", pred_coords2.cpu()[i].reshape(-1,3), unknown_mask[i], attn_mask[i])
                TorchNERFBuilder.sv2pdb(f"{svpath}/raw_{batch['key'][i]}.pdb", raw_coords.cpu()[i].reshape(-1,3), unknown_mask[i], attn_mask[i])
    
                

