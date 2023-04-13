# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: UBlock.py
@Author: nkul
@Date: 2023/4/13 下午12:12
"""


import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding='same')
        )

    def forward(self, x):
        return self.net(x) + x


class UBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 1 * h * w ==> c * h * w
        self.layer_1l = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding='same'),
            nn.ReLU(inplace=True)
        )
        # c * h * w => c * h/2 * w/2 (32)
        self.layer_2l = nn.Sequential(
            nn.MaxPool2d(2, 2),  # size/2
            DenseBlock(hidden_channels),
            nn.ReLU(inplace=True)
        )
        # c * h/2 * w/2 => c * h/4 * w/4 (16)
        self.layer_3l = nn.Sequential(
            nn.AvgPool2d(2, 2),
            DenseBlock(hidden_channels),
            nn.ReLU(inplace=True)
        )
        # c * h/4 * w/4 => c * h/8 * w/8 (8)
        self.layer_4 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            DenseBlock(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.layer_3r_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_3r_dense = nn.Sequential(
            # reduce channels, kernel size 3 or 1 ?
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding='same'),  # should be 3 ? todo
            nn.ReLU(inplace=True),  # need relu here ? todo
            DenseBlock(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.layer_2r_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_2r_dense = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding='same'),
            nn.ReLU(inplace=True),  # need relu here ? todo
            DenseBlock(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.layer_1r_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_1r = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding='same'),
            nn.ReLU(inplace=True),  # need relu here ? todo
            DenseBlock(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding='same'),  # kernel size 3 ? todo
            nn.Sigmoid()  # need sigmod here ? todo
        )

    def forward(self, x):
        x1 = self.layer_1l(x)
        x2 = self.layer_2l(x1)
        x3 = self.layer_3l(x2)

        x4 = self.layer_4(x3)

        x5 = self.layer_3r_up(x4)
        x5 = self.layer_3r_dense(torch.cat([x5, x3], 1))

        x6 = self.layer_2r_up(x5)
        x6 = self.layer_2r_dense(torch.cat([x6, x2], 1))

        x7 = self.layer_1r_up(x6)
        x7 = self.layer_1r(torch.cat([x7, x1], 1))
        return x7
