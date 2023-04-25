import os
import nni
import logging
import pickle
import json
import torch
import os.path as osp
from parser import create_parser
import datetime
import json
import wandb
import warnings
warnings.filterwarnings('ignore')


import random 
import numpy as np

def set_seed(seed):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



from utils import Recorder
from utils import *

from methods import  FoldingDiff, CFoldingDiff, DiffSDS


method_maps = {
    'FoldingDiff': FoldingDiff,
    'CFoldingDiff':CFoldingDiff,
    'DiffSDS': DiffSDS
}

# CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 main.py --ex_name ablation/full_feat

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 8 main.py --ex_name dualspace_bacth1024
# CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 main.py --ex_name dualspace_baseline
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 8 main.py --ex_name DiffSDS_bacth1024_epoch100k

class Exp:
    def __init__(self, args, show_params=True, distributed=True):
        self.args = args
        self.config = args.__dict__
        self.distributed = distributed
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            if self.distributed:
                torch.cuda.set_device(self.args.local_rank)
                device = torch.device("cuda", self.args.local_rank)
            else:
                device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        # set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch, self.train_loader.dataset.dset.get_masked_means(), self.distributed)

    def _get_data(self):
        self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config, self.distributed, self.config['sampling'])

    def _save(self, name=''):
        torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
        fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
        state = self.method.scheduler.state_dict()
        pickle.dump(state, fw)

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')))
        fw = open(osp.join(self.checkpoints_path, str(epoch) + '.pkl'), 'rb')
        state = pickle.load(fw)
        self.method.scheduler.load_state_dict(state)

    def train(self):
        recorder = Recorder(self.args.patience, verbose=True)
        for epoch in range(self.args.epoch):
            train_metrics = self.method.train_one_epoch(self.train_loader)
            
            wandb.log(train_metrics)
            
            if epoch % 500 == 0:
                self._save(str(epoch))

            if epoch % self.args.log_step == 0:
                valid_metrics = self.valid()
                
                print_log('Epoch: {}, Steps: {} | Train Loss: {:.4f}  Valid Loss: {}\n'.format(epoch + 1, len(self.train_loader), train_metrics['train_loss'], valid_metrics['valid_loss']))
                
                recorder(valid_metrics['valid_loss'], self.method.model, self.path)
                if recorder.early_stop:
                    print("Early stopping")
                    logging.info("Early stopping")
                    break
            
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path))

    def valid(self):
        valid_metrics = self.method.valid_one_epoch(self.valid_loader)
        wandb.log(valid_metrics)
        return valid_metrics

    def test(self):
        angle_sum = self.method.test_one_epoch(self.test_loader, self.train_loader)
        angle_sum = angle_sum.tolist()
        print_log('Test Loss: {} \n'.format(angle_sum))
        return angle_sum



if __name__ == '__main__':
    # debug: CUDA_VISIBLE_DEVICES="0" python -m debugpy --listen 5698 --wait-for-client -m torch.distributed.launch --nproc_per_node 1 main.py
    torch.distributed.init_process_group(backend='nccl')
    args = create_parser()
    config = args.__dict__
    default_params = load_config(osp.join('./configs', args.method + '.py' if args.config_file is None else args.config_file))
    config.update(default_params)
    config['local_rank'] = 0
    config['sampling'] = False
    
    print(config) 
    
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
    wandb.init(project="ProteinBinder", entity="gaozhangyang", config=config, name=args.ex_name, group=args.ex_name)

    set_seed(111)
    exp = Exp(args)
    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/inpaint/checkpoint/cfolddiff_baseline/checkpoint.pth"))
    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/results/dualspace_bacth1024/checkpoint.pth", map_location=torch.device('cuda:0')))
    
    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/results/dualspace_bacth1024_weighted/checkpoint.pth", map_location=torch.device('cuda:0')))
    
    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/results/NoConstraints_NoWeighted/checkpoint.pth", map_location=torch.device('cuda:0')))
    
    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/results/dualspace_epoch100k_restart_from_bacth1024weighted/checkpoint.pth", map_location=torch.device('cuda:0')))
    
    
    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/results/dualspace_epoch100k_restart_from_bacth1024weighted/checkpoint.pth", map_location=torch.device('cuda:0')))
    

    # exp.method.model.load_state_dict(torch.load("/gaozhangyang/experiments/ProreinBinder/results/CFoldingDiff_rebuttal/checkpoint.pth", map_location='cuda:0'))
    

    # exp.method.model.load_state_dict(torch.load(svpath+'checkpoint.pth'))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    angle_sum = exp.test()
    print("test loss", angle_sum)