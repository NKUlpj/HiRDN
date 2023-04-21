# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: HiDB.py
@Author: nkul
@Date: 2023/4/10 下午1:49
"""

import torch.nn as nn
from .Config import get_config
from .RU import ResidualUnit
from .HiFM import HiFM
from .Common import *


class HiDB(nn.Module):
    def __init__(self, channels, mode) -> None:
        super(HiDB, self).__init__()
        self.dc = self.distilled_channels = channels // 2
        self.rc = self.remaining_channels = channels

        self.c1_r = ResidualUnit(channels, channels, mode=mode)
        self.c2_r = ResidualUnit(channels, channels, mode=mode)
        self.c3_r = ResidualUnit(channels, channels, mode=mode)

        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = get_act_fn('lrelu', neg_slope=0.05)
        self.conv_group = nn.ModuleList()
        self.conv_group.append(conv_layer(self.dc * 2, channels, 1))
        _config = get_config(mode)
        _attn_name = _config['HiDB']
        _attn = get_attn_by_name(_attn_name, channels)
        if _attn is not None:
            self.conv_group.append(_attn)
        self.hifm = HiFM(channels, mode=mode)

    def forward(self, x):
        distilled_c1 = self.act(self.hifm(x))

        # r_c1 = (self.c1_r(x))
        # r_c1 = self.act(r_c1 + x)
        r_c1 = self.act(self.c1_r(x))

        # r_c2 = (self.c2_r(r_c1))
        # r_c2 = self.act(r_c2 + r_c1)
        r_c2 = self.act(self.c2_r(r_c1))

        # r_c3 = (self.c3_r(r_c2))
        # r_c3 = self.act(r_c3 + r_c2)
        r_c3 = self.act(self.c3_r(r_c2))

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, r_c4], dim=1)
        for conv in self.conv_group:
            out = conv(out)
        # out_fused = self.attn(self.c5(out))
        return out
