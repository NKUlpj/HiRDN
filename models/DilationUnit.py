# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: DilationUnit.py
@Author: nkul
@Date: 2023/4/13 下午2:48 
"""

import torch.nn as nn


class DilationUnit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.conv(x)
