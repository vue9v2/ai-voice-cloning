import torch
import argparse

import os
import sys

sys.path.insert(0, './dlas/codes/')
sys.path.insert(0, './dlas/')

from codes import train as tr
from utils import util, options as option

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='../options/train_vit_latent.yml')
parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=True)
if args.launcher != 'none':
    # export CUDA_VISIBLE_DEVICES for running in distributed mode.
    if 'gpu_ids' in opt.keys():
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
trainer = tr.Trainer()

#### distributed training settings
if args.launcher == 'none':  # disabled distributed training
    opt['dist'] = False
    trainer.rank = -1
    if len(opt['gpu_ids']) == 1:
        torch.cuda.set_device(opt['gpu_ids'][0])
    print('Disabled distributed training.')
else:
    opt['dist'] = True
    init_dist('nccl')
    trainer.world_size = torch.distributed.get_world_size()
    trainer.rank = torch.distributed.get_rank()
    torch.cuda.set_device(torch.distributed.get_rank())

trainer.init(args.opt, opt, args.launcher)
trainer.do_training()