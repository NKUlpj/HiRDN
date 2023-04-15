# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: UBlock.py
@Author: nkul
@Date: 2023/4/13 下午12:12
"""


import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DeConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, padding='same')
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class UBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 1 * h * w ==> c * h * w
        self.layer_1l = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding='same')
        )
        # c * h * w => c * h/2 * w/2 (32)
        self.layer_2l = nn.Sequential(
            nn.MaxPool2d(2, 2),  # size/2
            DoubleConv(hidden_channels, hidden_channels * 2)
        )
        # c * h/2 * w/2 => c * h/4 * w/4 (16)
        self.layer_3l = nn.Sequential(
            nn.AvgPool2d(2, 2),
            DoubleConv(hidden_channels * 2, hidden_channels * 4)
        )
        # c * h/4 * w/4 => c * h/8 * w/8 (8)
        self.layer_4 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, padding='same')
        )

        self.layer_3r_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_3r_dense = nn.Sequential(
            DeConv(hidden_channels * 8, hidden_channels * 4, hidden_channels * 2)
        )

        self.layer_2r_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_2r_dense = nn.Sequential(
            DeConv(hidden_channels * 4, hidden_channels * 2, hidden_channels)
        )

        self.layer_1r_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_1r = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding='same'),
            nn.Sigmoid()
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
