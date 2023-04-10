# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN 
@File ：HiDB.py
@Author ：nkul
@Date ：2023/4/10 下午1:49 
"""

import torch
import torch.nn as nn

from .Config import get_config
from .HiCBAM import HiCBAM
from .RU import ResidualUnit
from .HiFM import HiFM
from .Common import *
from .ConvMod import ConvMod
from .ESA import ESA


class HiDB(nn.Module):
    def __init__(self, channels, mode='T') -> None:
        super(HiDB, self).__init__()
        self.dc = self.distilled_channels = channels // 2
        self.rc = self.remaining_channels = channels

        self.c1_r = ResidualUnit(channels, channels, mode=mode)
        self.c2_r = ResidualUnit(channels, channels, mode=mode)
        self.c3_r = ResidualUnit(channels, channels, mode=mode)

        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = get_act_fn('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 2, channels, 1)

        _config = get_config(mode)
        _hidb = _config['HiDB']
        if _hidb == 'HiCBAM':
            # print("HiDB using HiCBAM")
            self.attn = HiCBAM(channels=channels)
        elif _hidb == 'ConvMod':
            # print("HiDB using ConvMod")
            self.attn = ConvMod(channels=channels)
        else:
            # print("HiDB using ESA")
            self.attn = ESA(channels, nn.Conv2d)
        self.hifm = HiFM(channels, mode=mode)

    def forward(self, x):
        distilled_c1 = self.act(self.hifm(x))
        r_c1 = (self.c1_r(x))
        r_c1 = self.act(r_c1 + x)

        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, r_c4], dim=1)
        out_fused = self.attn(self.c5(out))
        return out_fused
