# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: HiRDN_Loss.py
@Author: nkul
@Date: 2023/4/10 下午1:58
"""

import torch.nn as nn
from torchvision.models.vgg import vgg16
from DISTS_pytorch import DISTS
import warnings

from models.Config import get_config
warnings.filterwarnings("ignore")


class GeneratorLoss(nn.Module):
    def __init__(self, device, mode='T'):
        super(GeneratorLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        vgg = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.device = device
        self.loss_weights = get_config(mode)['Loss']
        loss_networks = []
        for layer in [1, 15, 25]:
            loss_network = nn.Sequential(*list(vgg.features)[:layer]).eval()
            for param in loss_network.parameters():
                param.requires_grad = False
            loss_networks.append(loss_network)
        self.loss_networks = loss_networks
        self.mse_loss = nn.MSELoss(reduce=True, size_average=True)
        self.l1_loss = nn.L1Loss()
        self.dists_loss = DISTS()

    def forward(self, out_images, target_images):
        perception_loss = 0
        for idx, _loss_network in enumerate(self.loss_networks):
            _loss_network.to(self.device)
            _out_feat = _loss_network(out_images.repeat([1, 3, 1, 1]))
            _target_feat = _loss_network(target_images.repeat([1, 3, 1, 1]))
            perception_loss += self.loss_weights[idx] * self.mse_loss(_out_feat, _target_feat)
        image_loss = self.l1_loss(out_images, target_images)
        dists_loss = self.dists_loss(out_images, target_images, require_grad=True, batch_average=True)
        # dists_loss: 0.3756; image_loss: .0298; perception_loss: 0.0016;
        return self.loss_weights[-2] * dists_loss + self.loss_weights[-1] * image_loss + perception_loss
