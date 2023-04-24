# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: HiRDN_Loss.py
@Author: nkul
@Date: 2023/4/10 下午1:58
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from DISTS_pytorch import DISTS
import warnings

from models.Config import get_config

warnings.filterwarnings("ignore")


class GeneratorLoss(nn.Module):
    def __init__(self, device, mode='T'):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        # vgg = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.device = device
        self.loss_weights = get_config(mode)['Loss']
        loss_networks = []
        for layer in [3, 8, 15]:
            # for layer in [1, 15, 25]:
            loss_network = nn.Sequential(*list(vgg.features)[:layer]).eval()
            for param in loss_network.parameters():
                param.requires_grad = False
            loss_networks.append(loss_network)
        self.loss_networks = loss_networks
        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        self.l1_loss = nn.L1Loss()
        self.dists_loss = DISTS()
        self.ms_ssim_l1_loss = MS_SSIM_L1_LOSS(device)

    def forward(self, out_images, target_images):
        perception_loss = 0
        for idx, _loss_network in enumerate(self.loss_networks):
            _loss_network.to(self.device)
            _out_feat = _loss_network(out_images.repeat([1, 3, 1, 1]))
            _target_feat = _loss_network(target_images.repeat([1, 3, 1, 1]))
            # _this_per_loss = self.mse_loss(_out_feat, _target_feat)
            # print(idx, _this_per_loss)
            perception_loss += self.loss_weights[idx] * self.mse_loss(_out_feat, _target_feat)
        # image_loss = self.l1_loss(out_images, target_images)
        # TODO
        ms_ssim_l1_loss = self.ms_ssim_l1_loss(out_images.repeat([1, 3, 1, 1]), target_images.repeat([1, 3, 1, 1]))
        # print(0.001 * ms_ssim_l1_loss)
        dists_loss = self.dists_loss(out_images, target_images, require_grad=True, batch_average=True)
        # print(image_loss, dists_loss)
        # return self.loss_weights[-2] * dists_loss + self.loss_weights[-1] * image_loss + perception_loss
        # return self.loss_weights[-1] * image_loss + perception_loss
        return 0.001 * ms_ssim_l1_loss + perception_loss + self.loss_weights[-2] * dists_loss


class MS_SSIM_L1_LOSS(nn.Module):
    # TODO alpha
    def __init__(self, device, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3 * len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3 * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3 * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.to(device=device)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()
