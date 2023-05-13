# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: Component.py
@Author: nkul
@Date: 2023/4/24 上午11:14 
ALL IN ONE
"""


import torch
import torch.nn as nn

from models.Common import *


class Scale(nn.Module):
    """
    Input x [B, C, H, W]
    return lambda * x
    """
    def __init__(self, init_value=1) -> None:
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class ResidualUnit(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=2, bias=True) -> None:
        super(ResidualUnit, self).__init__()
        r'''
        in:         [C * W * H]
        reduction:  [C * W * H]   ---> [C/2 * W * H]
        expansion:  [C/2 * W * H] ---> [C * W * H]
        out:        [C * W * H]
        '''
        hidden_channels = channels // reduction
        self.reduction = nn.Conv2d(channels, hidden_channels, kernel_size, padding='same', bias=bias)
        self.expansion = nn.Conv2d(hidden_channels, channels, kernel_size, padding='same', bias=bias)
        self.scale1 = Scale(1)
        self.scale2 = Scale(1)

    def forward(self, x):
        x1 = self.reduction(x)
        x1 = self.expansion(x1)
        return self.scale1(x) + self.scale2(x1)


class HiFM(nn.Module):

    def __init__(self, channels, mode, k=2) -> None:
        super(HiFM, self).__init__()
        r'''
        in:         [C * W * H]
        out:        [C/2 * W * H]
        '''
        self.k = k
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.k, stride=self.k),
            nn.Upsample(scale_factor=self.k, mode='nearest')
        )
        if mode != 'T':
            self.attn = get_attn_by_name('LKA', channels * 2)
        self.out = nn.Conv2d(channels * 2, channels // 2, 1, padding='same')

    def forward(self, x):
        tl = self.net(x)
        tl = x - tl
        x = torch.cat((x, tl), 1)
        if hasattr(self, 'attn') and self.attn is not None:
            x = self.attn(x)
        x = self.out(x)
        return x


class HiDB(nn.Module):
    def __init__(self, channels, mode) -> None:
        super(HiDB, self).__init__()
        r'''
        in:         [C * W * H]
        out:        [C * W * H]
        '''
        hidden_channels = channels // 2

        self.c1_r = ResidualUnit(channels)
        self.c2_r = ResidualUnit(channels)
        self.c3_r = ResidualUnit(channels)

        self.c4 = conv_layer(channels, hidden_channels, 3)
        self.act = get_act_fn('gelu')

        self.c = conv_layer(hidden_channels * 2, channels, 1)
        if mode != 'T':
            self.attn = get_attn_by_name('HiConvMod', channels)

        self.hifm = HiFM(channels, mode=mode)

    def forward(self, x):
        distilled_c1 = self.act(self.hifm(x))

        r_c1 = self.act(self.c1_r(x))
        r_c2 = self.act(self.c2_r(r_c1))
        r_c3 = self.act(self.c3_r(r_c2))
        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, r_c4], dim=1)
        out = self.c(out)

        if hasattr(self, 'attn') and self.attn is not None:
            out = self.attn(out)

        return out

