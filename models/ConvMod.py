# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN 
@File ：ConvMod.py
@Author ：nkul
@Date ：2023/4/10 下午1:49
see Reference Conv2Former
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvMod(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_channels = channels
        layer_scale_init_value = 1e-6
        self.norm = LayerNorm(hidden_channels, eps=1e-6, data_format='channels_first')
        self.a = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 11, padding=5, groups=hidden_channels)
        )
        self.v = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.proj = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((hidden_channels)), requires_grad=True)

    def forward(self, x):
        r = self.norm(x)
        a = self.a(r)
        r = a * self.v(r)
        r = self.proj(r)
        return x + self.layer_scale.unsqueeze(-1).unsqueeze(-1) * r

