# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: down_sample.py
@Author: nkul
@Date: 2023/4/10 下午12:53

A tools to down sample data from high resolution data.
"""

import numpy as np
import multiprocessing
import time
import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.parser_helper import *
from utils.io_helper import down_sampling
import logging
from utils.config import set_log_config, root_dir
set_log_config()


def down_sample(in_file, _low_res, _ratio):
    data = np.load(in_file, allow_pickle=True)
    hic = data['hic']
    down_hic = down_sampling(hic, _ratio)
    chr_name = os.path.basename(in_file).split('_')[0]
    out_file = os.path.join(os.path.dirname(
        in_file), f'{chr_name}_{_low_res}.npz')
    np.savez_compressed(out_file, hic=down_hic, ratio=_ratio)
    logging.debug(f'Saving file:{out_file}')


if __name__ == '__main__':
    args = data_down_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    ratio = args.ratio

    pool_num = 23 if multiprocessing.cpu_count(
    ) > 23 else multiprocessing.cpu_count() - 2

    data_dir = os.path.join(root_dir, 'mat', cell_line)
    in_files = [os.path.join(data_dir, f)
                for f in os.listdir(data_dir) if f.find(high_res) >= 0]

    logging.debug(f'Generating {low_res} files from {high_res} files by {ratio}x down_sampling.')
    start = time.time()
    for file in in_files:
        down_sample(file, low_res, ratio)
    '''
    # I'm not sure why using multithreading on my computer will cause problems
    # But, synchronized code does not spend too much time [8 min]
    # It will be ok
    logging.debug(f'Start a multiprocess pool with process_num = {pool_num}')
    pool = multiprocessing.Pool(pool_num)
    for file in in_files:
        pool.apply_async(down_sample, (file, low_res, ratio))
    pool.close()
    pool.join()
    '''
    logging.debug(f'All down_sampling processes done. Running cost is {(time.time()-start)/60:.1f} min.')
