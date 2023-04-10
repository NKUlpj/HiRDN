# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN 
@File ：Config.py.py
@Author ：nkul
@Date ：2023/4/10 下午4:34 
"""

# Model_T [Tiny]
__T = {
    'HiRDN': 5,
    'HiDB': 'ESA',
    'HiFM': 'None',
    'RU': [2, 3, 0]  # reduction_ratio, kernel size, block num
}

# Model_S [Small]
__S = {
    'HiRDN': 6,
    'HiDB': 'HiCBAM',
    'HiFM': 'CA',
    'RU': [2, 3, 2]
}

# Model_L [Large]
__L = {
    'HiRDN': 6,
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
