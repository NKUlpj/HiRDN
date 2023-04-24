# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: Attention.py
@Author: nkul
@Date: 2023/4/12 下午3:05
Attention Module
"""
import torch.nn as nn


class PA(nn.Module):
    def __init__(self, channels, reduction=8):
        super(PA, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding='same', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1, padding='same', bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


# Basic Channel Attention
class CA(nn.Module):
    def __init__(self, channels, reduction=8) -> None:
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y * x


class HiCBAM(nn.Module):
    """
    Input: B * C * H * W
    Out:   B * C * H * W
    """

    def __init__(self, channels) -> None:
        super(HiCBAM, self).__init__()
        self.channel_attention = CA(channels)
        self.pixel_attention = PA(channels)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.pixel_attention(out)
        return out


class LKA(nn.Module):
    def __init__(self, channels):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, 9, padding='same', groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 15, stride=1, padding='same', groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn
