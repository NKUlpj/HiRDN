# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN
@File ：LKA.py
@Author ：nkul
@Date ：2023/4/10 下午1:46
see Reference [Visual Attention Network]
"""
import torch
import torch.nn as nn
from .CA import CA


# 大核远距离的空间注意力模块
class AttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 7, stride=1, padding=9, groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class ChannelWiseSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(channels)
        self.ca = CA(channels=channels)
        self.proj_2 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        r = self.proj_1(x)
        r = self.activation(r)
        r1 = self.ca(r)
        r2 = self.spatial_gating_unit(r)
        r = torch.cat([r1, r2], 1)
        r = self.proj_2(r)
        return x + r
