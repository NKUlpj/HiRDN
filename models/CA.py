# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: CA.py
@Author: nkul
@Date: 2023/4/10 下午1:45
"""

import torch.nn as nn


# channel attention
class CA(nn.Module):
    def __init__(self, channels, reduction=16) -> None:
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
