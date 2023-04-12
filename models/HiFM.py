# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: HiFM.py
@Author: nkul
@Date: 2023/4/10 下午1:46
"""
import torch
import torch.nn as nn

from .Config import get_config
from .Common import get_attn_by_name


class HiFM(nn.Module):
    def __init__(self, channels, mode, k=2) -> None:
        super(HiFM, self).__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size= self.k, stride=self.k),
            nn.Upsample(scale_factor=self.k, mode='nearest')
        )
        self.conv_group = nn.ModuleList()
        _config = get_config(mode)
        _attn_name = _config['HiFM']
        _attn = get_attn_by_name(_attn_name, channels * 2)
        if _attn is not None:
            self.conv_group.append(_attn)
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
