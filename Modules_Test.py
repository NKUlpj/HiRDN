# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Modules_Test.py
@Author: nkul
@Date: 2023/4/10 下午6:14
"""
import numpy as np
import torch

from thop import profile

from models.Attention import LKA

if __name__ == '__main__':
    _input = torch.randn(1, 52, 64, 64)
    # net = HiRDN()
    net = LKA(channels=64)
    # # net = PRMLayer(groups=64)
    _output = net(_input)
    print(_output.shape)
    # # input = torch.randint(4 ** 5, (1, 996))
    macs, params = profile(net, inputs=(_input,))
    print(f"macs={macs/1000000}, params={params}")
    log_info = f"Model parameter number: {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    print(log_info)

