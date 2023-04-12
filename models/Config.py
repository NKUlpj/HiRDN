# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: Config.py.py
@Author: nkul
@Date: 2023/4/10 下午4:34
"""

# Model_T [Tiny]
__T_m1 = {
    'HiRDN': [48, 4],
    'HiDB': 'ESA',
    'HiFM': 'None',
    'RU': [2, 1, 2]  # reduction_ratio, kernel size, block num
}


__L1 = {
    'HiRDN': [52, 6],
    'HiDB': 'ConvMod',
    'HiFM': 'CWSA',
    'RU': [4, 3, 0]  # reduction_ratio, kernel size, block num
}

__T0 = {
    'HiRDN': [48, 4],
    'HiDB': 'None',
    'HiFM': 'None',
    'RU': [4, 3, 0],  # reduction_ratio, kernel size, block num
    'Loss': [0.0005, 0.0003, 0.0002, 0.01]
}

__T = {
    'HiRDN': [48, 4],
    'HiDB': 'None',
    'HiFM': 'None',
    'RU': [4, 3, 0],  # reduction_ratio, kernel size, block num
    'Loss': [0.0005, 0.003, 0.02, 0.1]  # No effect
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
# ---
#       SSIM:0.932070; PSNR:37.036517; LPIPS:0.028442; DISTS:0.106718 [Model_T_m1]
#       Train:583.8 min; Predict: 20.731886s T0

#       SSIM:0.932949; PSNR:37.122573; LPIPS:0.027259; DISTS:0.103415; Predict: 27.302058s L1
