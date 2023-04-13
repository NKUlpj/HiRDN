# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: RU.py
@Author: nkul
@Date: 2023/4/10 下午1:41
"""
import torch
import torch.nn as nn

from .Config import get_config
from .Scale import Scale


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, mode, bias=True) -> None:
        super(ResidualUnit, self).__init__()
        '''
        in:         [C * W * H]
        reduction:  [C * W * H]   ---> [C/2 * W * H]
        expansion:  [C/2 * W * H] ---> [C * W * H]
        out:        [C * W * H]
        '''
        _config = get_config(mode)
        _ru = _config['RU']
        hidden_channels = out_channels // _ru[0]
        _kernel_size = _ru[1]
        self.reduction = nn.Conv2d(in_channels, hidden_channels, _kernel_size, padding='same', bias=bias)
        self.conv_group = nn.ModuleList()
        for _ in range(_ru[2]):
            self.conv_group.append(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding='same', bias=bias)
            )
        self.expansion = nn.Conv2d(hidden_channels, out_channels, _kernel_size, padding='same', bias=bias)
        # self.act = nn.LeakyReLU(inplace=True)
        self.scale1 = Scale(1)
        self.scale2 = Scale(2)

    def forward(self, x):
        x1 = self.reduction(x)
        x0 = torch.zeros_like(x1)
        for conv in self.conv_group:
            x0, x1 = x1,  conv(x0 + x1)
        res = self.expansion(x0 + x1)
        return self.scale1(res) + self.scale2(x)  # residual is outside
