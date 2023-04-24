# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: model_train.py
@Author: nkul
@Date: 2023/4/10 下午2:12
"""
import math
import os
import random
import time

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.evaluating import eval_lpips, eval_dists
from utils.parser_helper import root_dir
from utils.ssim import ssim
from math import log10
import lpips
from DISTS_pytorch import DISTS
from utils.util_func import get_device, get_model, loader, get_loss_fn
import warnings
warnings.filterwarnings("ignore")
import logging

# 设置logging的等级以及打印格式
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - [%(levelname)s] %(message)s')
from torch.utils.tensorboard import SummaryWriter


def set_up(seed=3407):
    # 3407 is all you need. 3407 is not a specials number.
    # The seed is set to ensure that our results can be reproduced.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def __adjust_learning_rate(epoch):
    lr = 0.0005 * (0.1 ** (epoch // 10))
    return lr


def __train(model, model_name, train_loader, valid_loader, max_epochs, verbose):
    # step 1: initial
    set_up()
    out_dir = os.path.join(root_dir, 'checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, f'best_{model_name}.pytorch')
    final_ckpt = os.path.join(out_dir, f'final_{model_name}.pytorch')
    start = time.time()
    device = get_device()  # whether using GPU for training
    best_ssim = 0

    # step 2: load model
    net = model.to(device)
    log_info = f"Model parameter number - {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    logging.debug(log_info)

    # step 3: load loss
    criterion = get_loss_fn(model_name, device)
    criterion.to(device)
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
    lpips_fn.to(device)
    dists_fn = DISTS()
    dists_fn.to(device)

    _optimizer = optim.Adam(net.parameters(), lr=1e-4)
    _scheduler = CosineAnnealingLR(_optimizer, max_epochs)

    # step 4: start train
    _log_time = None
    _train_writer = None
    _val_writer = None
    if verbose:
        _log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        _train_writer = SummaryWriter(root_dir + f'/logs/{model_name}/train_{_log_time}')
        _val_writer = SummaryWriter(root_dir + f'/logs/{model_name}/val_{_log_time}')
    for epoch in range(1, max_epochs + 1):
        run_res = {'samples': 0, 'g_loss': 0, 'g_score': 0}
        # update by @nkul
        # alr = __adjust_learning_rate(epoch)
        # optimizer = optim.Adam(net.parameters(), lr=alr)
        # free memory
        for p in net.parameters():
            if p.grad is not None:
                del p.grad
        torch.cuda.empty_cache()
        net.train()
        train_bar = tqdm(train_loader, colour='#75c1c4')
        for _step, (data, target, _) in enumerate(train_bar):
            batch_size = data.size(0)
            run_res['samples'] += batch_size
            real_imgs = target.to(device)
            z = data.to(device)
            fake_imgs = net(z)
            net.zero_grad()
            g_loss = criterion(fake_imgs, real_imgs)
            g_loss.backward()
            _optimizer.step()
            # update by @nkul
            # optimizer.step()
            run_res['g_loss'] += g_loss.item() * batch_size
            train_bar.set_description(
                desc=f"[{epoch}/{max_epochs}] Loss_G: {run_res['g_loss'] / run_res['samples']:.6f}"
            )
            if verbose and _step % 100 == 0:
                _train_writer.add_scalar(tag=f'loss', scalar_value=g_loss, global_step=epoch * len(train_bar) + _step)
        # update by @nkul
        _scheduler.step()

        # step 4.2 staring valid
        # val_res 记录所有batch的总和
        val_res = {'g_loss': 0, 'mse': 0, 'ssim': 0, 'ssims': 0, 'psnr': 0, 'samples': 0, "lpips": 0, 'dists': 0}
        net.eval()
        valid_bar = tqdm(valid_loader, colour='#fda085')
        with torch.no_grad():
            for val_lr, var_hr, inds in valid_bar:
                batch_size = val_lr.size(0)
                val_res['samples'] += batch_size
                lr = val_lr.to(device)
                hr = var_hr.to(device)
                sr = net(lr)
                g_loss = criterion(sr, hr)
                # loss
                val_res['g_loss'] += g_loss.item() * batch_size
                # mae & mse
                _this_batch_mse = ((sr - hr) ** 2).mean()
                _this_batch_mae = abs(sr - hr).mean()
                val_res['mse'] += _this_batch_mse * batch_size
                batch_ssim = ssim(sr, hr)
                val_res['ssims'] += batch_ssim * batch_size
                val_res['psnr'] = 10 * log10(1 / (val_res['mse'] / val_res['samples']))
                val_res['ssim'] = val_res['ssims'] / val_res['samples']
                # lpips
                val_res['lpips'] += eval_lpips(sr, hr, lpips_fn)
                # dists
                val_res['dists'] += eval_dists(sr, hr, dists_fn)
                _avg_lpips = val_res['lpips'] / val_res['samples']
                _avg_dists = val_res['dists'] / val_res['samples']
                valid_bar.set_description(
                    desc=f"[Predicting in Valid set] PSNR: {val_res['psnr']:.6f} dB; SSIM: {val_res['ssim']:.6f}; "
                         f"LPIPS: {_avg_lpips:.6f}; DISTS: {_avg_dists:.6f}; ")
        this_psnr = val_res['psnr']
        this_ssim = val_res['ssim'].item()
        this_lpips = val_res['lpips'] / val_res['samples']
        this_dists = val_res['dists'] / val_res['samples']
        if this_ssim > best_ssim:
            best_ssim = this_ssim
            print(
                f'Update SSIM ===> '
                f'PSNR: {this_psnr:.6f} dB; SSIM: {this_ssim:.6f}; LPIPS: {this_lpips:.6f}; DISTS: {this_dists:.6f};')
            torch.save(net.state_dict(), best_ckpt)

        if verbose:
            _val_writer.add_scalar(tag=f'loss', scalar_value=val_res['g_loss'] / val_res['samples'],
                                   global_step=epoch * len(train_bar))
            _val_writer.add_scalar(tag=f'ssim', scalar_value=this_ssim,
                                   global_step=epoch * len(train_bar))
            _val_writer.add_scalar(tag=f'psnr', scalar_value=this_psnr,
                                   global_step=epoch * len(train_bar))
            _val_writer.add_scalar(tag=f'lpips', scalar_value=this_lpips,
                                   global_step=epoch * len(train_bar))

    # step5: save final ckpt
    print(f'All epochs done. Running cost is {(time.time() - start) / 60:.1f} min.')
    torch.save(net.state_dict(), final_ckpt)
    if verbose:
        _val_writer.close()
        _train_writer.close()


def __train_gan(_net_g, _net_d, model_name, train_loader, valid_loader, max_epochs, verbose):
    # step 1: initial
    set_up()
    out_dir = os.path.join(root_dir, 'checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, f'best_{model_name}.pytorch')
    final_ckpt = os.path.join(out_dir, f'final_{model_name}.pytorch')
    start = time.time()
    device = get_device()  # whether using GPU for training
    best_ssim = 0

    # step 2: load model
    # load network
    net_g = _net_g.to(device)
    net_d = _net_d.to(device)
    log_info = f"netG parameter number: {sum(p.numel() for p in net_g.parameters() if p.requires_grad)}"
    print(log_info)
    log_info = f"netD parameter number: {sum(p.numel() for p in net_d.parameters() if p.requires_grad)}"
    print(log_info)

    # step 3: load loss
    criterion_g = get_loss_fn('DeepHiC').to(device)
    criterion_d = torch.nn.BCELoss().to(device)

    # optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=0.0001)
    optimizer_d = optim.Adam(net_d.parameters(), lr=0.0001)
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
    lpips_fn.to(device)
    dists_fn = DISTS()
    dists_fn.to(device)

    # step 4: start train
    _log_time = None
    _train_writer = None
    _val_writer = None
    if verbose:
        _log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        _train_writer = SummaryWriter(root_dir + f'/logs/{model_name}/train_{_log_time}')
        _val_writer = SummaryWriter(root_dir + f'/logs/{model_name}/val_{_log_time}')
    for epoch in range(1, max_epochs + 1):
        run_res = {'samples': 0, 'g_loss': 0, 'd_loss': 0, 'g_score': 0, 'd_score': 0}
        net_g.train()
        net_d.train()
        train_bar = tqdm(train_loader, colour='#75c1c4')
        for _step, (data, target, _) in enumerate(train_bar):
            batch_size = data.size(0)
            run_res['samples'] += batch_size
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = target.to(device)
            z = data.to(device)
            fake_img = net_g(z)

            ############################
            # (2) Train discriminator
            ###########################
            net_d.zero_grad()
            real_out = net_d(real_img)
            fake_out = net_d(fake_img)
            d_loss_real = criterion_d(real_out, torch.ones_like(real_out))
            d_loss_fake = criterion_d(fake_out, torch.zeros_like(fake_out))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward(retain_graph=True)
            optimizer_d.step()

            ############################
            # (2) Train generator
            ###########################
            net_g.zero_grad()
            g_loss = criterion_g(fake_img, real_img)
            g_loss.backward()
            optimizer_g.step()

            run_res['g_loss'] += g_loss.item() * batch_size
            run_res['d_loss'] += d_loss.item() * batch_size
            run_res['d_score'] += real_out.mean().item() * batch_size
            run_res['g_score'] += fake_out.mean().item() * batch_size
            train_bar.set_description(
                desc=f"[{epoch}/{max_epochs}] Loss_G: {run_res['g_loss'] / run_res['samples']:.6f}"
            )

            if verbose and _step % 100 == 0:
                _train_writer.add_scalar(tag=f'loss', scalar_value=g_loss, global_step=epoch * len(train_bar) + _step)

        # step 4.2 staring valid
        # val_res 记录所有batch的总和
        val_res = {'g_loss': 0, 'd_loss': 0, 'g_score': 0, 'd_score': 0, 'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0,
                   'samples': 0, 'lpips': 0, 'dists': 0}
        net_g.eval()
        net_d.eval()
        valid_bar = tqdm(valid_loader, colour='#fda085')
        with torch.no_grad():
            for val_lr, var_hr, inds in valid_bar:
                batch_size = val_lr.size(0)
                val_res['samples'] += batch_size
                lr = val_lr.to(device)
                hr = var_hr.to(device)
                sr = net_g(lr)
                g_loss = criterion_g(sr, hr)
                # loss
                val_res['g_loss'] += g_loss.item() * batch_size
                # mae & mse
                _this_batch_mse = ((sr - hr) ** 2).mean()
                _this_batch_mae = abs(sr - hr).mean()
                val_res['mse'] += _this_batch_mse * batch_size
                batch_ssim = ssim(sr, hr)
                val_res['ssims'] += batch_ssim * batch_size
                val_res['psnr'] = 10 * log10(1 / (val_res['mse'] / val_res['samples']))
                val_res['ssim'] = val_res['ssims'] / val_res['samples']
                # lpips
                val_res['lpips'] += eval_lpips(sr, hr, lpips_fn)
                # dists
                val_res['dists'] += eval_dists(sr, hr, dists_fn)
                _avg_lpips = val_res['lpips'] / val_res['samples']
                _avg_dists = val_res['dists'] / val_res['samples']
                valid_bar.set_description(
                    desc=f"[Predicting in Valid set] PSNR: {val_res['psnr']:.6f} dB; SSIM: {val_res['ssim']:.6f}; "
                         f"LPIPS: {_avg_lpips:.6f}; DISTS: {_avg_dists:.6f}; ")

        this_psnr = val_res['psnr']
        this_ssim = val_res['ssim'].item()
        this_lpips = val_res['lpips'] / val_res['samples']
        this_dists = val_res['dists'] / val_res['samples']
        if this_ssim > best_ssim:
            best_ssim = this_ssim
            print(
                f'Update SSIM ===> '
                f'PSNR: {this_psnr:.6f} dB; SSIM: {this_ssim:.6f}; LPIPS: {this_lpips:.6f}; DISTS: {this_dists:.6f};')
            torch.save(net_g.state_dict(), best_ckpt)
        if verbose:
            _val_writer.add_scalar(tag=f'loss', scalar_value=val_res['g_loss'] / val_res['samples'],
                                   global_step=epoch * len(train_bar))
            _val_writer.add_scalar(tag=f'ssim', scalar_value=this_ssim,
                                   global_step=epoch * len(train_bar))
            _val_writer.add_scalar(tag=f'psnr', scalar_value=this_psnr,
                                   global_step=epoch * len(train_bar))
            _val_writer.add_scalar(tag=f'lpips', scalar_value=this_lpips,
                                   global_step=epoch * len(train_bar))

    # step5: save final ckpt
    print(f'All epochs done. Running cost is {(time.time() - start) / 60:.1f} min.')
    torch.save(net_g.state_dict(), final_ckpt)

    if verbose:
        _train_writer.close()
        _val_writer.close()


def model_train(_model_name, _train_file, _valid_file, _max_epochs, _batch_size, _verbose=True):
    # 1) get model
    net_g, _padding, net_d = get_model(_model_name)

    # 2) get loader
    train_loader = loader(_train_file, 'Train', _padding, True, _batch_size)
    valid_loader = loader(_valid_file, 'Valid', _padding, False, _batch_size)

    # 3) train
    if net_d is not None:
        __train_gan(net_g, net_d, _model_name, train_loader, valid_loader, _max_epochs, _verbose)
    else:
        __train(net_g, _model_name, train_loader, valid_loader, _max_epochs, _verbose)
