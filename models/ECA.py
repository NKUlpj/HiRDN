# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN
@File ：ECA.py
@Author ：nkul
@Date ：2023/4/10 下午4:10
See Reference ECA-NET
"""
import torch
import torch.nn as nn
import math


# channel attention
class ECA(nn.Module):
    def __init__(self, channels, b=1, gama=1) -> None:
        super(ECA, self).__init__()
        kernel_size = int(abs(math.log(channels, 2) + b) / gama)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # if kernel_size % 2 == 0:
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        # replace mlp with conv_1d
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False,
            padding=padding
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        x0 = self.avg_pool(x).view([b, 1, c])
        x0 = self.conv(x0).view([b, c, 1, 1])
        x1 = self.max_pool(x).view([b, 1, c])
        x1 = self.conv(x1).view([b, c, 1, 1])
        return self.sigmod(x0 + x1) * x
