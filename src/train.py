import os
import sys
import argparse


"""
if 'BITSANDBYTES_OVERRIDE_LINEAR' not in os.environ:
    os.environ['BITSANDBYTES_OVERRIDE_LINEAR'] = '0'
if 'BITSANDBYTES_OVERRIDE_EMBEDDING' not in os.environ:
    os.environ['BITSANDBYTES_OVERRIDE_EMBEDDING'] = '1'
if 'BITSANDBYTES_OVERRIDE_ADAM' not in os.environ:
    os.environ['BITSANDBYTES_OVERRIDE_ADAM'] = '1'
if 'BITSANDBYTES_OVERRIDE_ADAMW' not in os.environ:
    os.environ['BITSANDBYTES_OVERRIDE_ADAMW'] = '1'
"""


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

import torch
from codes import train as tr
from utils import util, options as option

# this is effectively just copy pasted and cleaned up from the __main__ section of training.py
# I'll clean it up better

def train(yaml, launcher='none'):
    opt = option.parse(yaml, is_train=True)
    if launcher != 'none':
        # export CUDA_VISIBLE_DEVICES for running in distributed mode.
        if 'gpu_ids' in opt.keys():
            gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
            print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    trainer = tr.Trainer()

    #### distributed training settings
    if launcher == 'none':  # disabled distributed training
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

    trainer.init(yaml, opt, launcher)
    trainer.do_training()

if __name__ == "__main__":
    # simple check because I'm brain damaged and forgot I can't modify what a module exports by simply changing the booleans that decide what it exports after the fact
    try:
        import torch_intermediary
        if torch_intermediary.OVERRIDE_ADAM:
            print("Using BitsAndBytes ADAMW optimizations")
    except Exception as e:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='../options/train_vit_latent.yml', nargs='+') # ugh
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    args = parser.parse_args()
    args.opt = " ".join(args.opt) # absolutely disgusting

    train(args.opt, args.launcher)