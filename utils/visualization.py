# -*- coding: UTF-8 -*-
"""
@Project: HiRDN
@File: visualization.py
@Author: nkul
@Date: 2023/4/10 下午4:19
"""

import matplotlib.pyplot as plt
import seaborn as sns
from colormap import Color, Colormap
import numpy as np


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
        vmin=0.05,
        xticklabels=[],
        yticklabels=[],
        cmap=my_cmap,
        cbar=False)


def __norm(x):
    # 归一化
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    return x


def plot_hic(matrix, start, end, _percentile=95, name=None):
    _matrix = matrix[start:end, start:end]
    v_max = np.percentile(_matrix, _percentile)
    __plot_hic(_matrix, v_max)
    plt.tight_layout()
    if name is not None:
        plt.title(name)
        plt.savefig(f'{name}.png')
    plt.show()


def plot_hic_matrix(_matrix, name):
    v_min = np.min(_matrix)
    _matrix = _matrix + abs(v_min)
    v_max = np.max(_matrix)
    print(v_min, v_max)
    __plot_hic(_matrix, 0.8, colors=['white', 'black'])
    # plt.show()
    plt.savefig(f'{name}')
