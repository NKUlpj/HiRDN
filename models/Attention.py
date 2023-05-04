# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: Attention.py
@Author: nkul
@Date: 2023/4/12 下午3:05
Attention Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    why rewrite F.layer_norm ?
    see https://github.com/facebookresearch/ConvNeXt/issues/112
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


# Former-Style Spatial Attention
class ConvMod(nn.Module):
    r"""Conv2Former
    https://arxiv.org/abs/2211.11943
    https://github.com/HVision-NKU/Conv2Former
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm(channels, eps=1e-6, data_format='channels_first')
        self.a = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 13, padding='same', groups=channels)  # n - k + 2p + 1 = n
        )
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        r = self.norm(x)
        a = self.a(r)
        r = a * self.v(r)
        r = self.proj(r)
        return r


class PA(nn.Module):
    def __init__(self, channels, reduction=4):
        super(PA, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding='same', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1, padding='same', bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


# Basic Channel Attention
class CA(nn.Module):
    def __init__(self, channels, reduction=4) -> None:
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


class HiConvMod(nn.Module):
    def __init__(self, channels) -> None:
        super(HiConvMod, self).__init__()
        self.spatial_attention = ConvMod(channels)
        self.pixel_attention = PA(channels)

    def forward(self, x):
        out = self.pixel_attention(x)
        out = self.spatial_attention(out)
        return out


class HiCBAM(nn.Module):
    """
    Input: B * C * H * W
    Out:   B * C * H * W
    """

    def __init__(self, channels) -> None:
        super(HiCBAM, self).__init__()
        self.channel_attention = CA(channels)
        self.pixel_attention = PA(channels)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.pixel_attention(out)
        return out


class LKA(nn.Module):
    r"""
    https://arxiv.org/abs/2202.09741
    """
    def __init__(self, channels):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, 9, padding='same', groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 15, stride=1, padding='same', groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn
