# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN
@File ：HiCBAM.py
@Author ：nkul
@Date ：2023/4/10 下午4:12
"""
import torch
import torch.nn as nn
from .ECA import ECA
from .ESA import ESA


class HiCBAM(nn.Module):
    """
    Input: B * C * H * W
    Out:   B * C * H * W
    """
    def __init__(self, channels) -> None:
        super(HiCBAM, self).__init__()
        self.channel_attention = ECA(channels)
        self.spatial_attention = ESA(conv=nn.Conv2d, channels=channels)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out
