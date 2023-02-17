import torch
import argparse

import os
import sys

# this is some massive kludge that only works if it's called from a shell and not an import/PIP package
# it's smart-yet-irritating module-model loader breaks when trying to load something specifically when not from a shell

sys.path.insert(0, './dlas/codes/')
# this is also because DLAS is not written as a package in mind
# it'll gripe when it wants to import from train.py
sys.path.insert(0, './dlas/')

# for PIP, replace it with:
# sys.path.insert(0, os.path.dirname(os.path.realpath(dlas.__file__)))
# sys.path.insert(0, f"{os.path.dirname(os.path.realpath(dlas.__file__))}/../")

# don't even really bother trying to get DLAS PIP'd
# without kludge, it'll have to be accessible as `codes` and not `dlas`

from codes import train as tr
from utils import util, options as option

# this is effectively just copy pasted and cleaned up from the __main__ section of training.py
# I'll clean it up better

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