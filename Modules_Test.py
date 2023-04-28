# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Modules_Test.py
@Author: nkul
@Date: 2023/4/10 下午6:14
"""
import os.path

import numpy as np
import torch

from thop import profile

from compared_models import HiCSR, DeepHiC
from models.Attention import LKA

if __name__ == '__main__':
    # _input = torch.randn(1, 1, 64, 64)
    # _netG = DeepHiC.Generator(1, in_channel=1, res_block_num=5)
    # _netD = DeepHiC.Discriminator(in_channel=1)
    # # net = LKA(channels=64)
    # # # net = PRMLayer(groups=64)
    # _output = _netD(_input)
    # print(_output.shape)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    print(cur_dir)
