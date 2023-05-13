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
from DISTS_pytorch import DISTS

from .evaluating import eval_dists, eval_ssim, eval_psnr
from .io_helper import together, spread_matrix
from .util_func import get_model, loader, get_device
from .config import set_log_config, root_dir

import warnings
warnings.filterwarnings("ignore")
import logging
set_log_config()


def __save_data(data, compact, size, file):
    data = spread_matrix(data, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hic=data, compact=compact)
    logging.debug(f'Saving file - {file}')


def __data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes


def __data_name(path):
    path = path.split('/')[-1]
    return path[:-4]


def __model_predict_without_target(model, _loader, ckpt_file):
    device = get_device()
    net = model.to(device)
    log_info = f"Model parameter number - {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    logging.debug(log_info)
    net.load_state_dict(
        torch.load(ckpt_file, map_location=torch.device('cpu'))
    )
    res_data = []
    res_inds = []
    net.eval()
    predict_bar = tqdm(_loader, colour='#178069', desc="Predicting:")
    with torch.no_grad():
        for batch in predict_bar:
            lr, inds = batch
            lr = lr.to(device)
            sr = net(lr)
            predict_bar.set_description(
                desc=f"[Predicting in Test set]")
            predict_data = sr.to('cpu').numpy()
            predict_data[predict_data < 0] = 0  # no Negative Number
            res_data.append(predict_data)
            res_inds.append(inds.numpy())

    # concatenate data
    res_data = np.concatenate(res_data, axis=0)
    res_inds = np.concatenate(res_inds, axis=0)
    res_hic = together(res_data, res_inds, tag='Reconstructing: ')
    return res_hic


def __model_predict(model, _loader, ckpt_file):
    device = get_device()
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
    val_res = {'ssims': 0, 'psnrs': 0, 'dists': 0, 'samples': 0}
    predict_bar = tqdm(_loader, colour='#178069', desc="Predicting:")
    with torch.no_grad():
        for batch in predict_bar:
            lr, hr, inds = batch
            batch_size = lr.size(0)
            val_res['samples'] += batch_size
            lr = lr.to(device)
            hr = hr.to(device)
            sr = net(lr)

            batch_ssim = eval_ssim(sr, hr)
            val_res['ssims'] += batch_ssim * batch_size
            val_res['psnrs'] += eval_psnr(sr, hr) * batch_size
            val_res['dists'] += eval_dists(sr, hr, dists_fn)
            _avg_dists = val_res['dists'] / val_res['samples']
            predict_bar.set_description(
                desc=f"[Predicting in Test set] PSNR: {val_res['psnrs']/val_res['samples']:.6f} dB;"
                     f"SSIM: {val_res['ssims']/val_res['samples']:.6f};  DISTS: {_avg_dists:.6f}; ")
            predict_data = sr.to('cpu').numpy()
            predict_data[predict_data < 0] = 0  # no Negative Number
            res_data.append(predict_data)
            res_inds.append(inds.numpy())

    # concatenate data
    res_data = np.concatenate(res_data, axis=0)
    res_inds = np.concatenate(res_inds, axis=0)
    res_hic = together(res_data, res_inds, tag='Reconstructing: ')
    this_ssim = val_res['ssims'] / val_res['samples']
    this_psnr = val_res['psnrs'] / val_res['samples']
    this_dists = val_res['dists'] / val_res['samples']
    logging.debug(f'SSIM:{this_ssim:.6f}; PSNR:{this_psnr:.6f}; ; DISTS:{this_dists:.6f};')
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
    predict_loader, has_target = loader(predict_file, 'Predict', _padding, False, _batch_size)

    # 3) Load ckpt
    best_ckpt_file = os.path.join(root_dir, 'checkpoints', ckpt)

    # 4) Predict
    start = time.time()
    if has_target:
        res_hic = __model_predict(model, predict_loader, best_ckpt_file)
    else:
        res_hic = __model_predict_without_target(model, predict_loader, best_ckpt_file)

    end = time.time()
    logging.debug(f'Model running cost is {(end - start):.6f} s.')

    # 5） return, put save code in main func as multiprocess must be created in main
    indices, compacts, sizes = __data_info(predict_data_np)

    out_dir = os.path.join(root_dir, 'predict')
    data_name = __data_name(predict_file)

    # 6) save data
    def save_data_n(_key):
        __file = os.path.join(out_dir, f'Predict_{model_name}_{data_name}_chr{_key}.npz')
        __save_data(res_hic[_key], compacts[_key], sizes[_key],  __file)

    for key in sizes.keys():
        save_data_n(key)
