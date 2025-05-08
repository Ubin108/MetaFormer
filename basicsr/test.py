import argparse
import random
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import (check_resume, make_exp_dirs, mkdir_and_rename, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import parse
from basicsr.utils.nano import psf2otf

import numpy as np
from tqdm import tqdm

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--name',
        default=None,
        help='job launcher')
    import sys
    vv = sys.version_info.minor
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train, name=args.name if args.name is not None and args.name != "" else None)


    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []
    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))


    # define ks for Wiener filters
    ks_params = opt['train'].get('ks', None)
    if not ks_params:
        raise NotImplementedError
    M = ks_params['num']
    ks = torch.logspace(ks_params['start'], ks_params['end'], M)
    ks = ks.view(1,M,1,1,1,1).to("cuda")

    val_conv = opt['val'].get("apply_conv", True)

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        current_iter = resume_state['iter']

    else:
        model = create_model(opt)
        current_iter = 0

    # load psf
    psf = torch.tensor(np.load("./psf.npy")).to("cuda")
    _,psf_h,psf_w,_ = psf.shape
    otf = psf2otf(psf, h=psf_h*3, w=psf_w*3, permute=True)[None]

    dataset_opt = opt['datasets']['val']

    val_set = create_dataset(dataset_opt)
    val_loader = create_dataloader(
        val_set,
        dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'])

    print("Start validation on spatially varying aberrration")
    rgb2bgr = opt['val'].get('rgb2bgr', True)
    use_image = opt['val'].get('use_image', True)
    psnr, others = model.validation(val_loader, current_iter, None, True, rgb2bgr, use_image, psf=otf, ks=ks, val_conv=val_conv)
    print("==================")
    print(f"Test results: PSNR: {psnr:.2f}, SSIM: {others['ssim']:.4f}, LPIPS: {others['lpips']:.4f}\n")


if __name__ == '__main__':
    main()
