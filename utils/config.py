# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: config.py
@Author: nkul
@Date: 2023/4/27 上午11:22 
"""
import logging

__log_level = logging.DEBUG
__log_format = '%(asctime)s - [%(levelname)s] %(message)s'


def set_log_config():
    logging.basicConfig(level=__log_level, format=__log_format)
