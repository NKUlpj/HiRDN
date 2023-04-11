# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: ESA.py
@Author: nkul
@Date: 2023/4/10 下午4:09
# Code copy from https://github.com/njulj/RFDN/blob/master/block.py
"""
import torch.nn as nn
import torch.nn.functional as F


class ESA(nn.Module):
    r"""
    CCA Layer, spatial attention
    """
    def __init__(self, channels, conv):
        super(ESA, self).__init__()
        f = channels // 4
        self.conv1 = conv(channels, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return m * x
