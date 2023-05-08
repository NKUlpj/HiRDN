# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: split_matrix.py
@Author: nkul
@Date: 2023/4/10 下午1:00
"""

import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils.parser_helper import split_matrix_parser, mkdir
from utils.io_helper import divide
import numpy as np
import time
import logging
from utils.config import set_log_config, root_dir
set_log_config()


def __data_divider(
        _n,
        d_file,
        _chunk=32,
        _stride=32,
        _bound=301,
        lr_cutoff=100):
    down_data = np.load(d_file, allow_pickle=True)['hic']
    full_size = down_data.shape[0]

    # Clamping
    down_data = np.minimum(lr_cutoff, down_data)

    # Rescaling
    # modify by nkul, Different processing with DeepHiC
    down_data = down_data / 255
    div_d_hic, div_index = divide(down_data, _n, _chunk, _stride, _bound)
    return _n, div_d_hic, div_index, full_size


def __get_chr_num(file_name):
    file_name = file_name.split('_')
    return file_name[3:]


if __name__ == '__main__':
    args = split_matrix_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    chunk = args.chunk
    stride = args.stride
    bound = args.bound

    data_dir = os.path.join(root_dir, 'mat', cell_line)
    out_dir = os.path.join(root_dir, 'data')
    mkdir(out_dir)

    start = time.time()
    results = []
    matrix_files = os.listdir(data_dir)
    for file in matrix_files:
        down_file = os.path.join(data_dir, file)
        res = __data_divider(__get_chr_num(file), down_file, chunk, stride, bound)
        results.append(res)

    logging.debug(f'All data generated. Running cost is {(time.time()-start)/60:.1f} min.')
    data = np.concatenate([r[1] for r in results])
    inds = np.concatenate([r[2] for r in results])
    sizes = {r[0]: r[3] for r in results}

    filename = f'{cell_line}_c{chunk}_s{stride}_b{bound}_predict.npz'
    split_file = os.path.join(out_dir, filename)
    np.savez_compressed(
        split_file,
        data=data,
        inds=inds,
        sizes=sizes)
    logging.debug(f'Saving file:{split_file}')
