# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: split.py
@Author: nkul
@Date: 2023/4/10 下午1:00
"""

import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.parser_helper import *
from utils.io_helper import divide
import numpy as np
import multiprocessing
import time
import logging
from utils.config import set_log_config
set_log_config()


def data_divider(
        _n,
        h_file,
        d_file,
        _chunk=32,
        _stride=32,
        _bound=301,
        lr_cutoff=100,
        hr_cutoff=255):
    high_data = np.load(h_file, allow_pickle=True)['hic']
    down_data = np.load(d_file, allow_pickle=True)['hic']
    full_size = high_data.shape[0]

    # Clamping
    high_data = np.minimum(hr_cutoff, high_data)
    down_data = np.minimum(lr_cutoff, down_data)

    # Rescaling
    # modify by nkul, Different processing with DeepHiC
    high_data = high_data / 255
    down_data = down_data / 255
    div_d_hic, div_index = divide(down_data, _n, _chunk, _stride, _bound)
    div_h_hic, _ = divide(high_data, _n, _chunk, _stride, _bound, verbose=True)
    return _n, div_d_hic, div_h_hic, div_index, full_size


if __name__ == '__main__':
    args = data_divider_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    dataset = args.dataset
    chunk = args.chunk
    stride = args.stride
    bound = args.bound

    chr_list = set_dict[dataset]
    postfix = cell_line.lower() if dataset == 'all' else dataset

    logging.debug(f'Going to read {high_res} and {low_res} data')

    pool_num = 23 if multiprocessing.cpu_count(
    ) > 23 else multiprocessing.cpu_count() - 2

    data_dir = os.path.join(root_dir, 'mat', cell_line)
    out_dir = os.path.join(root_dir, 'data')
    mkdir(out_dir)

    start = time.time()
    '''
    # I'm not sure why using multithreading on my computer can cause problems
    # But, synchronized code does not spend too much time[1 min]
    # It will be ok
    pool = multiprocessing.Pool(processes=pool_num)
    logging.debug(f'Start a multiprocess pool with processes = {pool_num} for generating data')
    results = []
    for n in chr_list:
        high_file = os.path.join(data_dir, f'chr{n}_{high_res}.npz')
        down_file = os.path.join(data_dir, f'chr{n}_{low_res}.npz')
        kwargs = {'_chunk': chunk, '_stride': stride, '_bound': bound}
        res = pool.apply_async(
            data_divider, (n, high_file, down_file,), kwargs)
        results.append(res)
    pool.close()
    pool.join()
    '''
    results = []
    for n in chr_list:
        high_file = os.path.join(data_dir, f'chr{n}_{high_res}.npz')
        down_file = os.path.join(data_dir, f'chr{n}_{low_res}.npz')
        # kwargs = {'_chunk': chunk, '_stride': stride, '_bound': bound}
        res = data_divider(n, high_file, down_file, chunk, stride, bound)
        results.append(res)

    logging.debug(f'All data generated. Running cost is {(time.time()-start)/60:.1f} min.')
    data = np.concatenate([r[1] for r in results])
    target = np.concatenate([r[2] for r in results])
    inds = np.concatenate([r[3] for r in results])
    sizes = {r[0]: r[4] for r in results}

    filename = f'{cell_line}_c{chunk}_s{stride}_b{bound}_{postfix}.npz'
    split_file = os.path.join(out_dir, filename)
    np.savez_compressed(
        split_file,
        data=data,
        target=target,
        inds=inds,
        sizes=sizes)
    logging.debug(f'Saving file:{split_file}')

