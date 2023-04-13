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
from models.Attention import *
from models.UBlock import UBlock
from thop import profile


if __name__ == '__main__':
    # _input = torch.randn(10, 1, 64, 64)
    # net = HiRDN()
    # # net = LKA(channels=64)
    # # net = UBlock(in_channels=1, hidden_channels=64, out_channels=1)
    # # net = PRMLayer(groups=64)
    # log_info = f"Model parameter number: {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    # print(log_info)
    # _output = net(_input)
    # print(_output.shape)
    # # input = torch.randint(4 ** 5, (1, 996))
    # macs, params = profile(net, inputs=(_input,))
    # print(f"macs={macs}, params={params}")
    _dict = torch.load('./Datasets_NPZ/checkpoints/best_HiRDN_T.pytorch')
    # print(_dict.keys())
    print(_dict['hidb_group.3.c2_r.scale1.scale'], _dict['hidb_group.3.c2_r.scale2.scale'])



