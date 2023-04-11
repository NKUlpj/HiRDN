# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: HiRDN_Loss.py
@Author: nkul
@Date: 2023/4/10 下午1:58
"""
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from DISTS_pytorch import DISTS
import warnings
warnings.filterwarnings("ignore")


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        # vgg = vgg16(pretrained=True)
        vgg = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        loss_networks = []
        loss_network_weights = [0.0005, 0.0003, 0.0002]
        for layer in [1, 15, 25]:
            loss_network = nn.Sequential(*list(vgg.features)[:layer]).eval()
            for param in loss_network.parameters():
                param.requires_grad = False
            loss_networks.append(loss_network)
        self.loss_networks = loss_networks
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.loss_network_weights = loss_network_weights
        self.dists_loss = DISTS()

    def forward(self, out_images, target_images):
        perception_loss = 0
        for idx, _loss_network in enumerate(self.loss_networks):
            _loss_network.to(torch.device('cuda:0'))
            _out_feat = _loss_network(out_images.repeat([1, 3, 1, 1]))
            _target_feat = _loss_network(target_images.repeat([1, 3, 1, 1]))
            perception_loss += self.loss_network_weights[idx] * self.mse_loss(_out_feat.reshape(
                _out_feat.size(0), -1), _target_feat.reshape(_target_feat.size(0), -1))
        image_loss = self.l1_loss(out_images, target_images)
        dists_loss = self.dists_loss(out_images, target_images, require_grad=True, batch_average=True)
        return 0.01 * dists_loss + image_loss + perception_loss
