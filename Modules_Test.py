# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Modules_Test.py
@Author: nkul
@Date: 2023/4/10 下午6:14
"""
import torch
from models.HiRDN import HiRDN
from models.BottleNeck import BottleNeck


if __name__ == '__main__':
    _input = torch.randn(1, 64, 64, 64)
    # net = HiRDN()
    net = BottleNeck(channels=64, ratio=2)
    log_info = f"Model parameter number: {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    print(log_info)
    _output = net(_input)
    print(_output.shape)
