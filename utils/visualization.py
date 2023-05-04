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
import seaborn as sns
from colormap import Color, Colormap
import numpy as np

from parser_helper import model_visual_parser
from config import set_log_config
set_log_config()


def __plot_hic(matrix_data, v_max, colors=None):
    r"""
    Function for plot Hi-C heat-map
    Params:
        matrix_data - contact matrix
        v_max - make all records that lager v_max to be max
        colors - color bar
    Returns:
        A Figure
    """
    if colors is None:
        colors = ['white', 'red']
    red_list = list()
    green_list = list()
    blue_list = list()
    for color in colors:
        col = Color(color).rgb
        red_list.append(col[0])
        green_list.append(col[1])
        blue_list.append(col[2])
    c = Colormap()
    d = {'blue': blue_list, 'green': green_list, 'red': red_list}
    my_cmap = c.cmap(d)
    fig, ax = plt.subplots(figsize=(4, 4))  # [宽， 高]
    ax.set_facecolor('w')
    sns.heatmap(
        matrix_data.T,
        vmax=v_max,
        vmin=0.00,
        xticklabels=[],
        yticklabels=[],
        cmap=my_cmap,
        cbar=False)


def plot_hic(matrix, start, end, _percentile=95, name=None):
    assert end - start <= 400, "distance boundary interested too large[<= 400 is recommended]"
    __matrix = matrix[start:end, start:end]
    v_max = np.percentile(__matrix, _percentile)
    __plot_hic(__matrix, v_max)
    plt.tight_layout()
    if name is not None:
        plt.title(name)
        plt.savefig(f'{name}.png')
        logging.debug(f'Save img to {name}.png')
    else:
        plt.show()


if __name__ == '__main__':
    args = model_visual_parser().parse_args(sys.argv[1:])
    _file = args.file
    _s = int(args.start)
    _e = int(args.end)
    _p = int(args.percentile)
    _n = args.name
    _m = np.load(_file)['hic']
    plot_hic(_m, _s, _e, _p, _n)

