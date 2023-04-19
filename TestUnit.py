# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: TestUnit.py
@Author: nkul
@Date: 2023/4/19 下午3:35 
"""
import torch


_dict = torch.load('./Datasets_NPZ/checkpoints/best_HiRDN_T_02.pytorch')
for i in range(6):
    print(_dict[f'hidb_group.{i}.c3_r.scale1.scale'], _dict[f'hidb_group.{i}.c3_r.scale2.scale'])

