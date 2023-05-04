# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: model_predict.py
@Author: nkul
@Date: 2023/4/10 下午2:12
"""
import os
import time

import numpy as np
from tqdm import tqdm
import torch

from utils.evaluating import eval_lpips, eval_dists
from utils.io_helper import together
from utils.ssim import ssim
from math import log10
import lpips
from DISTS_pytorch import DISTS
from utils.util_func import get_model, loader, get_device
import warnings
warnings.filterwarnings("ignore")
import logging
from utils.config import set_log_config, root_dir
set_log_config()


def __save_data(data, file):
    np.savez_compressed(file, hic=data)
    logging.debug(f'Saving file - {file}')


def __data_info(data):
    sizes = data['sizes'][()]
    return sizes


def __model_predict(model, _loader, ckpt_file):
    device = get_device()
    lpips_fn = lpips.LPIPS(net='alex', verbose=False)
    lpips_fn.to(device)
    dists_fn = DISTS()
    dists_fn.to(device)
    net = model.to(device)
    log_info = f"Model parameter number - {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    logging.debug(log_info)
    net.load_state_dict(
        torch.load(ckpt_file, map_location=torch.device('cpu'))
    )
    res_data = []
    res_inds = []
    net.eval()
    val_res = {'g_loss': 0, 'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'lpips': 0, 'dists': 0, 'samples': 0}
    predict_bar = tqdm(_loader, colour='#178069', desc="Predicting:")
    with torch.no_grad():
        for batch in predict_bar:
            lr, hr, inds = batch
            batch_size = lr.size(0)
            val_res['samples'] += batch_size
            lr = lr.to(device)
            hr = hr.to(device)
            sr = net(lr)
            batch_mse = ((sr - hr) ** 2).mean()
            val_res['mse'] += batch_mse * batch_size
            batch_ssim = ssim(sr, hr)
            val_res['ssims'] += batch_ssim * batch_size
            val_res['psnr'] = 10 * log10(1 / (val_res['mse'] / val_res['samples']))
            val_res['ssim'] = val_res['ssims'] / val_res['samples']
            val_res['lpips'] += eval_lpips(sr, hr, lpips_fn)
            val_res['dists'] += eval_dists(sr, hr, dists_fn)
            _avg_lpips = val_res['lpips'] / val_res['samples']
            _avg_dists = val_res['dists'] / val_res['samples']
            predict_bar.set_description(
                desc=f"[Predicting in Test set] PSNR: {val_res['psnr']:.6f} dB; SSIM: {val_res['ssim']:.6f}; "
                     f"LPIPS: {_avg_lpips:.6f}; DISTS: {_avg_dists:.6f}; ")
            predict_data = sr.to('cpu').numpy()
            predict_data[predict_data < 0] = 0  # no Negative Number
            res_data.append(predict_data)
            res_inds.append(inds.numpy())

    # concatenate data
    res_data = np.concatenate(res_data, axis=0)
    res_inds = np.concatenate(res_inds, axis=0)
    res_hic = together(res_data, res_inds, tag='Reconstructing: ')
    this_ssim = val_res['ssim']
    this_psnr = val_res['psnr']
    this_lpips = val_res['lpips'] / val_res['samples']
    this_dists = val_res['dists'] / val_res['samples']
    logging.debug(f'SSIM:{this_ssim:.6f}; PSNR:{this_psnr:.6f}; LPIPS:{this_lpips:.6f}; DISTS:{this_dists:.6f};')
    return res_hic


def model_predict(model_name, predict_file,  _batch_size, ckpt):
    # 1) Load Model
    model, _padding, _, = get_model(model_name)

    # 2) Load File
    logging.debug(f'Loading predict data - {predict_file}')
    # Load Predict Data
    in_dir = os.path.join(root_dir, 'data')
    predict_file_path = os.path.join(in_dir, predict_file)
    predict_data_np = np.load(predict_file_path, allow_pickle=True)
    predict_loader = loader(predict_file, 'Predict', _padding, False, _batch_size)

    # 3) Load ckpt
    best_ckpt_file = os.path.join(root_dir, 'checkpoints', ckpt)

    # 4) Predict
    start = time.time()
    res_hic = __model_predict(model, predict_loader, best_ckpt_file)
    end = time.time()
    logging.debug(f'Model running cost is {(end - start):.6f} s.')

    # 5） return, put save code in main func as multiprocess must be created in main
    sizes = __data_info(predict_data_np)
    return res_hic, sizes
