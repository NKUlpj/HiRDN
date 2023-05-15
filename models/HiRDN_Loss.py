# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: HiRDN_Loss.py
@Author: nkul
@Date: 2023/4/24 下午12:00 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from DISTS_pytorch import DISTS
import warnings

warnings.filterwarnings("ignore")


class LossL(nn.Module):
    """
    Loss_L = [r1 * vgg(3) + r2 * vgg(8) + r3 * vgg(15)] + alpha * dists_loss + beta * MS_SSIM_L1_LOSS
    MS_SSIM_L1_LOSS = beta1 * loss_ms_ssim + (1 - beta1) * gaussian_l1
    """
    def __init__(self, device):
        super().__init__()
        vgg = vgg16(pretrained=True)
        # vgg = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.device = device
        self.loss_weights = [0.004, 1.5e-06, 1.5e-06, 1]
        loss_networks = []
        for layer in [3, 8, 15]:
            loss_network = nn.Sequential(*list(vgg.features)[:layer]).eval()
            for param in loss_network.parameters():
                param.requires_grad = False
            loss_networks.append(loss_network)
        self.loss_networks = loss_networks
        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        # self.l1_loss = nn.L1Loss()
        self.ms_ssim_l1_loss = MS_SSIM_L1_LOSS(device=device, alpha=0.15)
        self.dists_loss = DISTS()

    def forward(self, out_images, target_images):
        perception_loss = 0
        for idx, _loss_network in enumerate(self.loss_networks):
            _loss_network.to(self.device)
            _out_feat = _loss_network(out_images.repeat([1, 3, 1, 1]))
            _target_feat = _loss_network(target_images.repeat([1, 3, 1, 1]))
            perception_loss += self.loss_weights[idx] * self.mse_loss(_out_feat, _target_feat)
        ms_ssim_l1_loss = self.ms_ssim_l1_loss(out_images, target_images)
        dists_loss = self.dists_loss(out_images, target_images, require_grad=True, batch_average=True)
        return ms_ssim_l1_loss + perception_loss + 0.008 * dists_loss


class MS_SSIM_L1_LOSS(nn.Module):
    """
    Some Code from https://github.com/psyrocloud/MS-SSIM_L1_LOSS
    Paper "Loss Functions for Image Restoration With Neural Networks"
    """
    def __init__(self, device, data_range=1.0, k=(0.01, 0.03), alpha=0.84, compensation=1, channel=1):
        super(MS_SSIM_L1_LOSS, self).__init__()
        gaussian_sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
        self.channel = channel
        self.DR = data_range
        self.C1 = (k[0] * data_range) ** 2
        self.C2 = (k[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._f_special_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._f_special_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 1, 0, :, :] = self._f_special_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._f_special_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.to(device=device)

    @staticmethod
    def _f_special_gauss_1d(size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _f_special_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma ([float]): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._f_special_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel
        mu_x = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)
        mu_y = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mu_x2
        sigma_y2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - mu_y2
        sigma_xy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - mu_xy

        # l(j), cs(j) in MS-SSIM
        _l = (2 * mu_xy + self.C1) / (mu_x2 + mu_y2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigma_xy + self.C2) / (sigma_x2 + sigma_y2 + self.C2)

        if self.channel == 3:
            l_m = _l[:, -1, :, :] * _l[:, -2, :, :] * _l[:, -3, :, :]
        else:
            l_m = _l[:, -1, :, :]

        # l_m = _l[:, -1, :, :] * _l[:, -2, :, :] * _l[:, -3, :, :]
        p_ics = cs.prod(dim=1)

        loss_ms_ssim = 1 - l_m * p_ics  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
