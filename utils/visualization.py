# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: visualization.py
@Author: nkul
@Date: 2023/4/10 下午4:19
"""
import logging
import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import matplotlib.pyplot as plt
import numpy as np

from parser_helper import model_visual_parser
# from config import set_log_config
# set_log_config()


def __plot_hic(matrix, start, end, percentile, name, cmap, color_bar=False):
    data2d = matrix[start:end, start:end]
    v_max = np.percentile(data2d, percentile)
    fig, ax = plt.subplots()
    im = ax.imshow(data2d, interpolation="nearest", vmax=v_max, vmin=0, cmap=cmap)
    if name is None or name == "":
        name = f"{start} - {end}"
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
    if color_bar:
        fig.colorbar(im, ax=ax)
    plt.savefig(name + '.png')
    logging.info(f'Save pic to {name}.png')
    plt.show()


if __name__ == '__main__':
    logging.disable(logging.DEBUG)
    args = model_visual_parser().parse_args(sys.argv[1:])
    _file = args.file
    _s = int(args.start)
    _e = int(args.end)
    _p = int(args.percentile)
    _n = args.name
    _m = np.load(_file)['hic']
    _c = args.cmap
    __plot_hic(_m, _s, _e, _p, _n, _c, color_bar=True)
