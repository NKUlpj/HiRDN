# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: predict.py
@Author: nkul
@Date: 2023/4/10 下午2:00
"""
import multiprocessing
import os
import sys

import numpy as np

from utils.parser_helper import model_predict_parser
from model_predict import model_predict
import logging
from utils.config import set_log_config, root_dir
set_log_config()


def __save_data(data, file, verbose=False):
    np.savez_compressed(file, hic=data)
    if verbose:
        logging.debug(f'Saving file:{file}')


if __name__ == '__main__':
    args = model_predict_parser().parse_args(sys.argv[1:])
    model_name = args.model
    predict_file = args.predict_file
    batch_size = args.batch_size
    ckpt = args.ckpt
    res_hic, sizes = model_predict(model_name, predict_file,  batch_size, ckpt)
    out_dir = os.path.join(root_dir, 'predict')

    # 6) save data
    def save_data_n(_key):
        __file = os.path.join(out_dir, f'Predict_{model_name}_{predict_file}_chr{_key}.npz')
        __save_data(res_hic[_key], __file)

    if multiprocessing.cpu_count() > 23:
        pool_num = 23
    else:
        pool_num = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(processes=pool_num)
    logging.debug(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
    # below multiprocess must be run in main func
    for key in sizes.keys():
        pool.apply_async(save_data_n, (key,))
    pool.close()
    pool.join()
