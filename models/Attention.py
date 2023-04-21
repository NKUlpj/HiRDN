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
import math
from torch.distributions.normal import Normal


# Rewrite Layer Norm, see the issue in ConvNeXt Repo
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


class PA(nn.Module):
    def __init__(self, channels, reduction=8):
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


# Enhancer Chanel Attention
# See ref ECA-NET
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


class ESA(nn.Module):
    r"""
    CCA Layer, spatial attention
    Code copy from https://github.com/njulj/RFDN/blob/master/block.py
    """

    def __init__(self, channels, conv=nn.Conv2d):
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


# Former-Style Spatial Attention
# See ref Conv2Former
class ConvMod(nn.Module):
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
        return x + r


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
    def __init__(self, channels):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, 7, padding='same', groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 15, stride=1, padding='same', groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class ChannelWiseSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = ConvMod(channels)
        self.ca = CA(channels=channels)
        self.proj_2 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        r = self.proj_1(x)
        r = self.activation(r)
        r1 = self.ca(r)
        r2 = self.spatial_gating_unit(r)
        r = torch.cat([r1, r2], 1)
        r = self.proj_2(r)
        return x + r


class NonLocalAttention(nn.Module):
    def __init__(self, channels, reduction=2, res_scale=1):
        super(NonLocalAttention, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.PReLU()
        )

    def forward(self, x):
        x_embed_1 = self.conv1(x)
        x_embed_2 = self.conv2(x)
        x_embed_3 = self.conv3(x)

        n, c, h, w = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(n, h * w, c)
        x_embed_2 = x_embed_2.view(n, c, h * w)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_embed_3 = x_embed_3.view(n, -1, h * w).permute(0, 2, 1)
        x_res = torch.matmul(score, x_embed_3)
        return x_res.permute(0, 2, 1).view(n, -1, h, w) + self.res_scale * x



