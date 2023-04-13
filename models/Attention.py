# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: Attention.py
@Author: nkul
@Date: 2023/4/12 下午3:05
Attention Module
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.normal import Normal


# Rewrite Layer Norm, see issue ConvNeXt
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
        layer_scale_init_value = 1e-2
        self.norm = LayerNorm(channels, eps=1e-6, data_format='channels_first')
        self.a = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 11, padding=5, groups=channels)  # n - k + 2p + 1 = n
        )
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x):
        r = self.norm(x)
        a = self.a(r)
        r = a * self.v(r)
        r = self.proj(r)
        return x + self.layer_scale.unsqueeze(-1).unsqueeze(-1) * r


class HiCBAM(nn.Module):
    """
    Input: B * C * H * W
    Out:   B * C * H * W
    """
    def __init__(self, channels) -> None:
        super(HiCBAM, self).__init__()
        self.channel_attention = ECA(channels)
        self.spatial_attention = ESA(conv=nn.Conv2d, channels=channels)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class LKA(nn.Module):
    def __init__(self, channels):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 7, stride=1, padding=9, groups=channels, dilation=3)
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
        self.spatial_gating_unit = LKA(channels)
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


class PRMLayer(nn.Module):
    def __init__(self, groups=52, mode='dot_product'):
        super(PRMLayer, self).__init__()
        self.mode = mode
        self.groups = groups
        self.max_pool = nn.AdaptiveMaxPool2d(1, return_indices=True)
        self.weight = nn.Parameter(torch.zeros(1, self.groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, self.groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.one = nn.Parameter(torch.ones(1, self.groups, 1))
        self.zero = nn.Parameter(torch.zeros(1, self.groups, 1))
        self.theta = nn.Parameter(torch.rand(1, 2, 1, 1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        b, c, h, w = x.size()
        position_mask = self.get_position_mask(x, b, h, w, self.groups)  # batch * group, 2, 64, 64
        # Similarity function
        query_value, query_position = self.get_query_position(x, self.groups)  # shape [b*num,2,1,1]
        # print(query_position.float()/h)
        query_value = query_value.view(b*self.groups, -1, 1)
        x_value = x.view(b * self.groups, -1, h * w)
        similarity_max = self.get_similarity(x_value, query_value, mode=self.mode)
        similarity_gap = self.get_similarity(x_value, self.gap(x).view(b * self.groups, -1, 1), mode=self.mode)
        similarity_max = similarity_max.view(b, self.groups, h * w)
        distance = abs(position_mask - query_position)
        distance = distance.type(query_value.type())
        # distance = torch.exp(-distance * self.theta)
        distribution = Normal(0, self.scale)
        distance = distribution.log_prob(distance * self.theta).exp().clone()
        distance = (distance.mean(dim=1)).view(b, self.groups, h * w)
        # print_dis = distance.mean(dim=0).mean(dim=0).view(h, w)
        # np.savetxt(time.perf_counter().__str__()+'.txt', print_dis.detach().cpu().numpy())
        similarity_max = similarity_max*distance
        similarity_gap = similarity_gap.view(b, self.groups, h*w)
        similarity = similarity_max*self.zero+similarity_gap*self.one
        context = similarity - similarity.mean(dim=2, keepdim=True)
        std = context.std(dim=2, keepdim=True) + 1e-5
        context = (context/std).view(b, self.groups, h, w)
        # affine function
        context = context * self.weight + self.bias
        context = context.view(b*self.groups, 1, h, w)\
            .expand(b*self.groups, c//self.groups, h, w).reshape(b, c, h, w)
        value = x*self.sig(context)
        return value

    @staticmethod
    def get_position_mask(x, b, h, w, number):
        mask = (x[0, 0, :, :] != 2020).nonzero()
        mask = (mask.reshape(h, w, 2)).permute(2, 0, 1).expand(b*number, 2, h, w)
        return mask

    def get_query_position(self, query, groups):
        b, c, h, w = query.size()
        value = query.view(b*groups, c//groups, h, w)
        value = value.sum(dim=1, keepdim=True)
        max_value, max_position = self.max_pool(value)
        t_position = torch.cat((max_position//w, max_position % w), dim=1)
        t_value = value[torch.arange(b*groups), :, t_position[:, 0, 0, 0], t_position[:, 1, 0, 0]]
        t_value = t_value.view(b, c, 1, 1)
        return t_value, t_position

    @staticmethod
    def get_similarity(query, key_value, mode='dot_product'):
        if mode == 'dot_product':
            similarity = torch.matmul(key_value.permute(0, 2, 1), query).squeeze(dim=1)
        elif mode == 'l1norm':
            similarity = -(abs(query - key_value)).sum(dim=1)
        elif mode == 'cosine':
            similarity = torch.cosine_similarity(query, key_value, dim=1)
        else:
            similarity = torch.matmul(key_value.permute(0, 2, 1), query)
        return similarity


class AttentionProducing(nn.Module):
    def __init__(self):
        super(AttentionProducing, self).__init__()
