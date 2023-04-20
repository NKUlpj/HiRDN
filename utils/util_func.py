# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: util_func.py
@Author: nkul
@Date: 2023/4/10 下午2:07
"""


import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import compared_models.HiCARN_1 as HiCARN
import compared_models.HiCNN as HiCNN
import compared_models.DeepHiC as DeepHiC
import compared_models.HiCSR as HiCSR
# from models.HiRDN import HiRDN
import models.HiRDN as HiRDN
from utils.parser_helper import root_dir


# get model by name
def get_model(_model_name):
    _padding = False
    _netG = None
    _netD = None

    if _model_name == 'HiRDN':
        _netG = HiRDN.HiRDN()

    elif _model_name == 'HiCARN':
        _netG = HiCARN.Generator(num_channels=64)

    elif _model_name == 'HiCNN':
        _netG = HiCNN.Generator()
        _padding = True

    elif _model_name == 'HiCSR':
        _padding = True
        _netG = HiCSR.Generator()
        _netD = HiCSR.Discriminator()

    elif _model_name == 'DeepHiC':
        _padding = False
        _netG = DeepHiC.Generator(1, in_channel=1, res_block_num=5)
        _netD = DeepHiC.Discriminator(in_channel=1)
    else:
        raise NotImplementedError('Model {} is not implemented'.format(_model_name))
    return _netG, _padding, _netD


# get data loader
def loader(file_name, loader_type='train', padding=False, shuffle=True, batch_size=64):
    __data_dir = os.path.join(root_dir, 'data')
    __file = os.path.join(__data_dir, file_name)
    __file_np = np.load(__file)

    __input_np = __file_np['data']
    __input_tensor = torch.tensor(__input_np, dtype=torch.float)
    if padding:
        __input_tensor = F.pad(__input_tensor, (6, 6, 6, 6), mode='constant')
    __target_np = __file_np['target']
    __target_tensor = torch.tensor(__target_np, dtype=torch.float)

    __inds_np = __file_np['inds']
    __inds_tensor = torch.tensor(__inds_np, dtype=torch.int)
    print(f"{loader_type} Set Size: {__input_tensor.size()}")
    __dataset = TensorDataset(__input_tensor, __target_tensor, __inds_tensor)
    __data_loader = DataLoader(__dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return __data_loader


def get_device():
    _has_cuda = torch.cuda.is_available()
    _device = torch.device('cuda:0' if _has_cuda else 'cpu')
    print("CUDA available? ", _has_cuda)
    if not _has_cuda:
        print("GPU acceleration is strongly recommended")
    return _device
