# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Scale.py
@Author: nkul
@Date: 2023/4/10 下午4:26
"""
import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, init_value=1) -> None:
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale
