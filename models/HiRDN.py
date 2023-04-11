# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: HiRDN.py
@Author: nkul
@Date: 2023/4/10 下午1:56
"""
import torch
import torch.nn as nn
from .Common import *
from .Config import get_config
from .HiDB import HiDB


class HiRDN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, mode='T') -> None:
        super(HiRDN, self).__init__()
        _config = get_config(mode)
        hidden_channels = _config['HiRDN'][0]
        _block_num = _config['HiRDN'][1]

        self.fea_conv = conv_layer(in_channels, hidden_channels, kernel_size=3)
        self.hidb_group = nn.ModuleList()

        for _ in range(_block_num):
            self.hidb_group.append(
                HiDB(channels=hidden_channels, mode=mode)
            )
        # self.hidb1 = HiDB(channels=hidden_channels, mode=mode)
        # self.hidb2 = HiDB(channels=hidden_channels, mode=mode)
        # self.hidb3 = HiDB(channels=hidden_channels, mode=mode)
        # self.hidb4 = HiDB(channels=hidden_channels, mode=mode)
        # self.hidb5 = HiDB(channels=hidden_channels, mode=mode)
        # self.hidb6 = HiDB(channels=hidden_channels, mode=mode)
        self.c = conv_block(hidden_channels * _block_num, hidden_channels, kernel_size=1, act_type='lrelu')
        self.LR_conv = conv_layer(hidden_channels, hidden_channels, kernel_size=3)
        self.exit = conv_block(hidden_channels, out_channels, kernel_size=3, stride=1, act_type='lrelu')

    def forward(self, x):
        out_fea = self.fea_conv(x)
        x1 = out_fea.clone()
        cat_arr = []
        for hidb in self.hidb_group:
            x1 = hidb(x1)
            cat_arr.append(x1.clone())
        x1 = torch.cat(cat_arr, dim=1)

        # out_b1 = self.hidb1(out_fea)
        # out_b2 = self.hidb2(out_b1)
        # out_b3 = self.hidb3(out_b2)
        # out_b4 = self.hidb4(out_b3)
        # out_b5 = self.hidb5(out_b4)
        # out_b6 = self.hidb6(out_b5)
        # x1 = torch.cat([out_b1, out_b2, out_b3, out_b4, out_b5, out_b6], dim=1)
        out_b = self.c(x1)
        out_lr = self.LR_conv(out_b) + out_fea
        output = self.exit(out_lr)
        return output
