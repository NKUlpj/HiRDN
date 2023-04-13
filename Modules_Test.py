# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Modules_Test.py
@Author: nkul
@Date: 2023/4/10 下午6:14
"""
import numpy as np
import torch
from models.HiRDN import HiRDN
from models.BottleNeck import BottleNeck
from models.Attention import *
from models.UBlock import UBlock
from thop import profile

from utils.visualization import plot_hic_matrix

if __name__ == '__main__':
    _input = torch.randn(10, 1, 64, 64)
    # net = HiRDN()
    # # net = LKA(channels=64)
    # # net = UBlock(in_channels=1, hidden_channels=64, out_channels=1)
    # # net = PRMLayer(groups=64)
    # log_info = f"Model parameter number: {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    # print(log_info)
    # _output = net(_input)
    # print(_output.shape)
    # # input = torch.randint(4 ** 5, (1, 996))
    # macs, params = profile(net, inputs=(_input,))
    # print(f"macs={macs}, params={params}")
    _dict = torch.load('./Datasets_NPZ/checkpoints/best_HiRDN.pytorch')
    _dict_keys = list(_dict.keys())
    # print(_dict.keys())
    net = UBlock(in_channels=1, out_channels=1,hidden_channels=52)
    w = net.state_dict()
    keys = list(w.keys())
    for i in range(34):
        w[keys[i]] = _dict[_dict_keys[-(34-i)]]
    net.load_state_dict(w)
    _input = np.load('./Experimental/hr.npz', allow_pickle=True)['hic']
    plot_hic_matrix(_input,'1.png')
    _input = np.expand_dims(_input, 2)
    _input = torch.from_numpy(_input.transpose([2, 0, 1])).unsqueeze(0).float()
    print(_input.shape)

    _out = net(_input) * _input
    _out = _out.squeeze(0).detach().numpy().transpose(1, 2, 0)[:,:,0]
    np.savetxt('out.txt', _out)

    plot_hic_matrix(_out, '2.png')
    # print(_dict['hidb_group.3.c2_r.scale1.scale'], _dict['hidb_group.3.c2_r.scale2.scale'])



