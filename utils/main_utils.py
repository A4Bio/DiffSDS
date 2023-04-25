import os
import logging
import numpy as np
import torch
import random 
import torch.backends.cudnn as cudnn
from .config_utils import Config
import glob
import hashlib


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(filename:str = None):
    '''
    load and print config
    '''
    print('loading config from ' + filename + ' ...')
    configfile = Config(filename=filename)
    config = configfile._cfg_dict
    return config

def md5_all_py_files(dirname: str) -> str:
    """Create a single md5 sum for all given files"""
    # https://stackoverflow.com/questions/36099331/how-to-grab-all-files-in-a-folder-and-get-their-md5-hash-in-python
    fnames = glob.glob(os.path.join(dirname, "*.py"))
    hash_md5 = hashlib.md5()
    for fname in sorted(fnames):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

def len2mask(lens):
    max_len = lens.max()
    mask = torch.arange(max_len, device=lens.device).expand(len(lens), max_len) < lens.unsqueeze(1)
    return mask

def select_feats2mat(mask, raw_feat):
    length = mask.sum(dim=1)
    mask2 = len2mask(length)
    if len(raw_feat.shape)==2:
        new_feat = torch.zeros(mask2.shape[0], mask2.shape[1], device=mask2.device, dtype=raw_feat.dtype)
        new_feat = new_feat.masked_scatter(mask2, torch.masked_select(raw_feat, mask))
    if len(raw_feat.shape)==3:
        new_feat = torch.zeros(mask2.shape[0], mask2.shape[1], raw_feat.shape[2], device=mask2.device, dtype=raw_feat.dtype)
        new_feat = new_feat.masked_scatter(mask2.unsqueeze(-1), torch.masked_select(raw_feat, mask.unsqueeze(-1)))
    return new_feat, mask2