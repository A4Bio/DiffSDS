import torch
import os.path as osp
from parser import create_parser

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
import json



if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__
    # default_params = load_config(osp.join('./configs', args.method + '.py' if args.config_file is None else args.config_file))
    # config.update(default_params)
    
    # param = json.load(open("/gaozhangyang/experiments/ProreinBinder/results/cfoldingdiff_bacth1024/model_param.json", 'r'))
    
    param = json.load(open("/gaozhangyang/experiments/ProreinBinder/results/CFoldingDiff_rebuttal/model_param.json", 'r'))
    config.update(param)
    
    config['strict_test'] = True
    config['batch_size'] = 1024
    config['method'] = 'CFoldingDiff'
    config['mode'] = "denoise"
    config["sampling"] = True
    mode = config['mode']
    print(config)
    
    
    exp = Exp(args, distributed=False)
    # 统计模型参数数量
    params = torch.nn.utils.parameters_to_vector(exp.method.model.parameters())
    num_params = params.size(0)
    # 转换参数量的单位
    if num_params >= 1e9:
        num_params = f'{num_params / 1e9:.2f}G'
    elif num_params >= 1e6:
        num_params = f'{num_params / 1e6:.2f}M'
    elif num_params >= 1e3:
        num_params = f'{num_params / 1e3:.2f}K'
    else:
        num_params = f'{num_params:.2f}B'

    print(f'模型参数数量：{num_params}')



    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/inpaint/checkpoint/cfolddiff_baseline/checkpoint.pth"))
    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/results/cfoldingdiff_bacth1024/checkpoint.pth", map_location='cuda:0'))
    
    
    state_dict = torch.load("/gaozhangyang/experiments/ProreinBinder/results/CFoldingDiff_rebuttal/checkpoint.pth", map_location='cuda:0')
    new_state_dict = OrderedDict()
    for k in state_dict:
        name = k.replace('module.', '')
        new_state_dict[name] = state_dict[k]
    exp.method.model.load_state_dict(new_state_dict)
    
    
    
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> sampling  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # 需要修改load_data.py里面的## only for sampling_foldingdiff, match duaspace 部分
    for batch in tqdm(exp.test_loader):
        angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx = cuda([batch["angles"], batch['coords'], batch["attn_mask"], batch["position_ids"], batch["t"], batch["seqs"], batch["unknown_mask"], batch["start_idx"], batch["end_idx"]], device=exp.method.model.device)
        raw_coords = coords.clone()
        timestamps = 1000
        
        
        
        angles = exp.method.sampling(angles, coords, attn_mask, position_ids, timestamps, seqs, unknown_mask, start_idx, end_idx , exp.train_loader.dataset)

        unknown_mask = unknown_mask.squeeze()
        
        phi, psi, omega, tau, CA_C_1N, C_1N_1CA = torch.split(angles, 1, dim=-1)
        
        pred_coords = coords*(~unknown_mask[...,None,None])
        pred_coords2 = exp.method.model.pred_coord(pred_coords.clone(), start_idx, end_idx, phi, psi, omega, C_1N_1CA, tau, CA_C_1N)


        # for i in range(angles.shape[0]):
        #     if (attn_mask*unknown_mask)[i].sum()>0:
        #         # TorchNERFBuilder.sv2pdb(f"/gaozhangyang/experiments/ProreinBinder/results/test/empty_{i}.pdb", pred_coords.cpu()[i].reshape(-1,3), unknown_mask[i])
        #         TorchNERFBuilder.sv2pdb(f"/gaozhangyang/experiments/ProreinBinder/results/inpaint_cfoldingdiff_bacth1024/pred_{batch['key'][i]}.pdb", pred_coords2.cpu()[i].reshape(-1,3), unknown_mask[i], attn_mask[i])
        #         TorchNERFBuilder.sv2pdb(f"/gaozhangyang/experiments/ProreinBinder/results/inpaint_cfoldingdiff_bacth1024/raw_{batch['key'][i]}.pdb", raw_coords.cpu()[i].reshape(-1,3), unknown_mask[i], attn_mask[i])
                

