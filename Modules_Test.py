# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Modules_Test.py
@Author: nkul
@Date: 2023/4/10 下午6:14
"""
import numpy as np
import torch

from models.HiRDN import HiRDN
from models.BottleNeck import BottleNeck
from models.Attention import *
from models.UBlock import UBlock
from thop import profile

from utils.visualization import plot_hic_matrix

if __name__ == '__main__':
    _input = torch.randn(10, 64, 64, 64)
    # net = HiRDN()
    # # net = LKA(channels=64)
    net = UBlock(in_channels=1, hidden_channels=16, out_channels=1)
    # # net = PRMLayer(groups=64)
    _output = net(_input)
    print(_output.shape)
    # # input = torch.randint(4 ** 5, (1, 996))
    macs, params = profile(net, inputs=(_input,))
    print(f"macs={macs/1000000}, params={params}")
