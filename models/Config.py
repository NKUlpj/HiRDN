# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Config.py.py
@Author: nkul
@Date: 2023/4/10 下午4:34
"""

# Model_T [Tiny]
__T0 = {
    'HiRDN': [48, 4],
    'HiDB': 'ESA',
    'HiFM': 'None',
    'RU': [2, 1, 2]  # reduction_ratio, kernel size, block num
}

__T = {
    'HiRDN': [48, 4],
    'HiDB': 'None',
    'HiFM': 'None',
    'RU': [4, 3, 0]  # reduction_ratio, kernel size, block num
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


def get_config(mode):
    if mode == 'S':
        return __S
    elif mode == 'T':
        return __T
    print('Error, No this Mode!')
    exit()

# HiCARN
#       SSIM:0.931695; PSNR:37.093705; LPIPS:0.035611                                           ; Predict: 15.406859
#       SSIM:0.932070; PSNR:37.036517; LPIPS:0.028442; DISTS:0.106718 [Model_T0] Train:583.8 min; Predict: 20.731886s
