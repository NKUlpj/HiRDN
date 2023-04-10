# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN 
@File ：HiFM.py
@Author ：nkul
@Date ：2023/4/10 下午1:46 
"""
import torch
import torch.nn as nn

from .Config import get_config
from .LKA import ChannelWiseSpatialAttention
from .CA import CA


class HiFM(nn.Module):
    def __init__(self, channels, k=2, mode='T') -> None:
        super(HiFM, self).__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size= self.k, stride=self.k),
            nn.Upsample(scale_factor=self.k, mode='nearest')
        )
        _config = get_config(mode)
        _hifm = _config['HiFM']
        self.conv_group = nn.ModuleList()
        if _hifm == 'CA':
            # print("HiFM using CA")
            self.conv_group.append(
                CA(channels=channels * 2)
            )
        elif _hifm == 'CWSA':
            # print("HiFM using CWSA")
            self.conv_group.append(
                ChannelWiseSpatialAttention(channels=channels * 2)
            )
        self.conv_group.append(
            nn.Conv2d(channels * 2, channels // 2, 1, padding='same')
        )

    def forward(self, x):
        tl = self.net(x)
        tl = x - tl
        x = torch.cat((x, tl), 1)
        for conv in self.conv_group:
            x = conv(x)
        return x
