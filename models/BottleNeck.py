# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: BottleNeck.py.py
@Author: nkul
@Date: 2023/4/11 下午6:41 
"""
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(self, channels, ratio):
        super(BottleNeck, self).__init__()
        hidden_channels = channels // ratio
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding='same', groups=channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, 3, padding='same', groups=channels)
        )

    def forward(self, x):
        return self.net(x)
