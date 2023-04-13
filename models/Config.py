# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Config.py.py
@Author: nkul
@Date: 2023/4/10 下午4:34
"""


# Model_T [Tiny]
__T = {
    'HiRDN': [48, 4],
    'HiDB': 'None',
    'HiFM': 'None',
    'RU': [2, 3, 0],  # reduction_ratio, kernel size, block num
    'Loss': [0.01, 0.0001, 0.0003, 0.01, 1]  # perception_loss, dists_loss, image_loss
}

# Model_S [Small]
__S = {
    'HiRDN': [52, 6],
    'HiDB': 'HiCBAM',
    'HiFM': 'CA',
    'RU': [2, 1, 3]
}

# Model_L [Large]
__L = {
    'HiRDN': [52, 6],
    'HiDB': 'ConvMod',
    'HiFM': 'CWSA',
    'RU': [2, 1, 3]
}


__model_dict = {
    'S': __S,
    'T': __T
}


def get_config(mode):
    if mode not in __model_dict:
        raise NotImplementedError('HiRDN_[{:s}] is not found'.format(mode))
    return __model_dict[mode]

# HiCARN
#       SSIM:0.931695; PSNR:37.093705; LPIPS:0.035611; Predict: 15.406859
