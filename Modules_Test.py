# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN 
@File ：Modules_Test.py
@Author ：nkul
@Date ：2023/4/10 下午6:14 
"""
import torch
from models.HiRDN import HiRDN


if __name__ == '__main__':
    _input = torch.randn(1, 1, 64, 64)
    net = HiRDN()
    _output = net(_input)
    print(_output.shape)
