import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument("--local_rank", default=-1, type=int, help="Used for DDP, local rank means the process number of each machine")
    
    # dataset parameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument("--nproc_per_node", default=8, type=int, help="Used for DDP, local rank means the process number of each machine")
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--num_workers', default=80, type=int)
    parser.add_argument('--pad', default=128, type=int)
    parser.add_argument('--min_length', default=40, type=int)
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--sampling', default=False, type=bool)
    
    # method parameters
    parser.add_argument('--method', default='DiffSDS', choices=['CFoldingDiff', 'DiffSDS', 'FoldingDiff']) 
    parser.add_argument('--config_file', '-c', default=None, type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--num_heads', default=12, type=int)
    parser.add_argument('--hidden_size', default=384, type=int)
    parser.add_argument('--intermediate_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=16, type=int)
    parser.add_argument('--position_embedding_type', default="relative_key", type=str)
    parser.add_argument('--dropout_p', default=0.1, type=float)
    parser.add_argument('--timesteps', default=1000, type=int)
    parser.add_argument('--use_grad', default=0, type=int)
    parser.add_argument('--mode', default="colddiff", choices=['colddiff', 'denoise']) 
    parser.add_argument("--use_seq", default=True, type=bool)
    parser.add_argument("--strict_test", default=True, type=bool)

    # Training parameters
    parser.add_argument('--epoch', default=10000, type=int, help='end epoch')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--patience', default=10000, type=int)
    parser.add_argument('--search', default=0, type=int)
    
    # Loss params
    parser.add_argument('--W_len', default=0.001, type=float, help='weight of the length loss')
    parser.add_argument('--W_overlap', default=1, type=float, help='weight of the length loss')
    parser.add_argument('--ignore_zero_center', default=True, type=bool)
    parser.add_argument("--W_angle", default="1.0, 1.0, 1.0, 1.0, 1.0, 1.0")
    

    args = parser.parse_args()
    return args