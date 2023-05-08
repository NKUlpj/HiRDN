# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: Schedule.py
@Author: nkul
@Date: 2023/5/5 下午8:00 
"""
from utils.model_train import model_train


models = ['HiCARN', 'DeepHiC']
train_file = 'GM12878_c40_s40_b301_train.npz'
valid_file = 'GM12878_c40_s40_b301_valid.npz'
max_epochs = 50
batch_size = 64
verbose = True
for model_name in models:
    model_train(model_name, train_file, valid_file, max_epochs, batch_size, verbose)
