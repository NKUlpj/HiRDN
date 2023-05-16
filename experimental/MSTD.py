# -*- coding: UTF-8 -*-
"""
@Project: HiRDN 
@File: MSTD.py
@Author: nkul
@Date: 2023/5/15 下午3:09 
@GitHub: https://github.com/nkulpj
Code from https://github.com/zhanglabtools/MSTD/tree/master
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colormap import Color, Colormap


def _domain_only_diagonal(data, win_n, distance):
    dsize = data.shape[0]
    # step1.1
    p_density = np.zeros(dsize)
    den_dict = {}
    for ip in range(dsize):
        begin_i = ip - win_n + 1
        end_i = ip + win_n - 1
        if (begin_i <= 0) | (end_i >= dsize - 1):
            if begin_i < 0:
                begin_i = 0

            if end_i > dsize - 1:
                end_i = dsize - 1
            p_density[ip] = np.mean(data[begin_i:ip + 1, :][:, ip:end_i + 1])
            den_dict[ip] = p_density[ip]
        else:
            p_density[ip] = p_density[ip - 1] + (
                    -np.sum(data[begin_i - 1:ip, ip - 1]) - np.sum(data[begin_i - 1, ip:end_i])
                    + np.sum(data[ip, ip:end_i + 1]) + np.sum(data[begin_i:ip, end_i])) / (win_n * win_n)
            den_dict[ip] = p_density[ip] + np.random.random(1) / 1000
    # step1.2
    max_step = 100
    ndp_dict = {}
    ass_dict = {}
    for ip in np.arange(0, dsize):
        step = None
        for step in np.arange(0, max(ip, dsize - ip)):
            if ip - step >= 0:
                up_point = p_density[ip - step]
                if up_point > p_density[ip]:
                    ass_dict[ip] = ip - step
                    break
            if ip + step <= dsize - 1:
                down_point = p_density[ip + step]
                if down_point > p_density[ip]:
                    ass_dict[ip] = ip + step
                    break
            if step > max_step:
                ass_dict[ip] = ip
                break
        ndp_dict[ip] = step

        # boundaries DF
    start = {}
    end = {}
    center = {}
    thr_den = np.percentile(p_density, 20)
    point_assign = {}
    for temp in den_dict:
        point_assign[temp] = 0
    # class_num=1
    join_num = 0
    # centers=[]
    for item in den_dict:
        den = den_dict[item]
        dist = ndp_dict[item]
        if (den > thr_den) & (dist > distance):
            join_num = join_num + 1
            point_assign[item] = join_num
            # class_num=class_num+1
            start[join_num] = item
            end[join_num] = item
            center[join_num] = item
            # centers.append(item)
            ass_dict[item] = item
    clues = pd.DataFrame({'Start': start, 'End': end, 'Cen': center}, columns=['Start', 'End', 'Cen'])

    old_join_num = 0
    new_join_num = join_num
    while old_join_num != new_join_num:
        old_join_num = join_num
        for item in den_dict:
            if ndp_dict[item] <= distance:
                if ass_dict[item] == item:
                    continue
                f_class = point_assign[ass_dict[item]]
                if f_class != 0:
                    mclass = point_assign[item]
                    if mclass == 0:
                        temp = center[f_class]
                        if den_dict[item] > den_dict[temp] / 5:
                            # 判断此点是否在类别范围
                            item_class = clues[(item > clues['Start']) & (clues['End'] > item)].values
                            if len(item_class) != 0:
                                point_assign[item] = point_assign[ass_dict[item_class[0][2]]]
                            else:
                                # print item
                                point_assign[item] = point_assign[ass_dict[item]]
                                if item < clues.loc[point_assign[item], 'Start']:
                                    clues.loc[point_assign[item], 'Start'] = item
                                else:
                                    clues.loc[point_assign[item], 'End'] = item
                            join_num = join_num + 1
        new_join_num = join_num

    step = 3
    for clu in clues.index[:-1:1]:
        left = clues.loc[clu, 'End']
        right = clues.loc[clu + 1, 'Start']
        if (left - step >= 0) & (right + step <= dsize - 1):
            if left == right - 1:
                loca = np.argmin(p_density[left - step:right + step])
                new_bound = left - step + loca
                clues.loc[clu, 'End'] = new_bound
                clues.loc[clu + 1, 'Start'] = new_bound
    return clues


# Data=matrix_data
def _generate_density_con(data, win, thr, mdhd):
    dsize = data.shape
    m_density = np.zeros(dsize)
    den_dict = {}
    if dsize[0] == dsize[1]:
        for i in range(dsize[0]):
            for j in range(dsize[1]):
                if i - j > mdhd * 4:
                    begin_i = i - win[0]
                    begin_j = j - win[1]
                    end_i = i + win[0]
                    end_j = j + win[1]
                    if (begin_i < 0) | (begin_j < 0) | (end_i > dsize[0] - 1) | (end_j > dsize[1] - 1):
                        if begin_i < 0:
                            begin_i = 0
                        if begin_j < 0:
                            begin_j = 0
                        if end_i > dsize[0] - 1:
                            end_i = dsize[0] - 1
                        if end_j > dsize[1] - 1:
                            end_j = dsize[1] - 1
                        m_density[i, j] = np.mean(data[begin_i:end_i, begin_j:end_j]) + np.random.random(1) / 1000.0
                    else:
                        m_density[i, j] = m_density[i, j - 1] + (-np.sum(data[begin_i:end_i, begin_j - 1])
                                                                 + np.sum(data[begin_i:end_i, end_j - 1])) / (
                                                      4 * win[0] * win[1])
                    if data[i, j] > thr:
                        den_dict[(i, j)] = m_density[i, j]
    else:
        for i in range(dsize[0]):
            for j in range(dsize[1]):
                begin_i = i - win[0]
                begin_j = j - win[1]
                end_i = i + win[0]
                end_j = j + win[1]
                if (begin_i < 0) | (begin_j < 0) | (end_i > dsize[0] - 1) | (end_j > dsize[1] - 1):
                    if begin_i < 0:
                        begin_i = 0
                    if begin_j < 0:
                        begin_j = 0
                    if end_i > dsize[0] - 1:
                        end_i = dsize[0] - 1
                    if end_j > dsize[1] - 1:
                        end_j = dsize[1] - 1
                    m_density[i, j] = np.mean(data[begin_i:end_i, begin_j:end_j]) + np.random.random(1) / 1000.0
                else:
                    m_density[i, j] = m_density[i, j - 1] + (-np.sum(data[begin_i:end_i, begin_j - 1])
                                                             + np.sum(data[begin_i:end_i, end_j - 1])) / (
                                                  4 * win[0] * win[1])
                if data[i, j] > thr:
                    den_dict[(i, j)] = m_density[i, j]
    return m_density, den_dict


def _find_high_points_v2(den_dict, ratio=1):
    dis = 50
    ndp_dict = {}
    ass_dict = {}
    for item in den_dict:
        # item=ass_dict[item]; item
        ndp_dict[item] = np.linalg.norm((dis, dis * ratio))
        ass_dict[item] = item
        for step in np.arange(1, dis + 1, 1):
            step_point = [(item[0] + st, item[1] + ra) for st in np.arange(-step, step + 1) for ra in
                          np.arange(-step * ratio, step * ratio + 1)
                          if (abs(st) == step or ratio * (step - 1) < abs(ra) <= ratio * step)]
            step_point = [point for point in step_point if point in den_dict]
            distance_index = [(np.linalg.norm(((item[0] - temp[0]) * ratio, item[1] - temp[1])), temp) for temp in
                              step_point if den_dict[temp] > den_dict[item]]
            distance_index.sort()
            for ind in distance_index:
                if den_dict[ind[1]] > den_dict[item]:
                    ndp_dict[item] = ind[0]
                    ass_dict[item] = ind[1]
                    break
            if len(distance_index) > 0:
                break
    return ndp_dict, ass_dict


def _assign_class(den_dict, ndp_dict, ass_dict, thr_den, thr_dis):
    locs = ['upper', 'bottom', 'left', 'right', 'cen_x', 'cen_y']

    point_assign = {}
    for temp in den_dict:
        point_assign[temp] = 0
        # class_num=1
    join_num = 0
    boundaries = pd.DataFrame()
    center = dict()

    for item in den_dict:
        den = den_dict[item]
        dist = ndp_dict[item]
        # value=den*dist
        bound = list()
        if (den >= thr_den) and (dist >= thr_dis):
            join_num = join_num + 1
            point_assign[item] = join_num
            center[join_num] = item
            # class_num=class_num+1
            bound.append(item[0])
            bound.append(item[0] + 1)
            bound.append(item[1])
            bound.append(item[1] + 1)
            bound.append(item[0])
            bound.append(item[1])
            # for k in range(len(locs)):
            #   if (k<2) | (k==4):
            #       bound.append(item[0])
            #   else:
            #       bound.append(item[1])
            ass_dict[item] = item
            bound = pd.DataFrame(bound)
            boundaries = pd.concat([boundaries, bound.T], axis=0)
    boundaries.columns = locs
    boundaries.index = np.arange(1, len(boundaries) + 1)

    thr_den1 = np.percentile(pd.Series(den_dict), 5)
    # Al=len(DEN_Dict)
    old_join_num = 0
    new_join_num = join_num
    while old_join_num != new_join_num:
        old_join_num = join_num
        for item in den_dict:
            if ndp_dict[item] < thr_dis:
                if ass_dict[item] == item:
                    continue
                f_class = point_assign[ass_dict[item]]
                if f_class != 0:
                    # print item
                    mclass = point_assign[item]
                    if mclass == 0:
                        if den_dict[item] > thr_den1:
                            # 判断此点是否在类别范围
                            item_class = boundaries[
                                ((item[0] > boundaries['upper']) & (boundaries['bottom'] > item[0]) &
                                 (item[1] > boundaries['left']) & (boundaries['right'] > item[1]))].values
                            if len(item_class) > 0:
                                if len(item_class) > 1:
                                    print(item_class)
                                point_assign[item] = point_assign[ass_dict[(item_class[0][4], item_class[0][5])]]
                            else:
                                # print item, 2
                                x1 = boundaries.loc[f_class, 'upper']
                                x2 = boundaries.loc[f_class, 'bottom']
                                x3 = boundaries.loc[f_class, 'left']
                                x4 = boundaries.loc[f_class, 'right']
                                # print x1,x2,x3,x4
                                # 更新前确认不能有任何重合
                                point_assign[item] = f_class
                                if item[0] < x1:
                                    sub_bound = boundaries[boundaries['bottom'] <= x1]
                                    if np.all(((sub_bound['right'] <= x3) | (sub_bound['left'] >= x4) |
                                               (item[0] >= sub_bound['bottom']))):
                                        # print (item)
                                        boundaries.loc[f_class, 'upper'] = item[0]
                                elif item[0] > x2:
                                    sub_bound = boundaries[boundaries['upper'] >= x2]
                                    if np.all(((sub_bound['left'] >= x4) | (sub_bound['right'] <= x3)) |
                                              (item[0] <= sub_bound['upper'])):
                                        # print (item)
                                        boundaries.loc[f_class, 'bottom'] = item[0]
                                if item[1] < x3:
                                    sub_bound = boundaries[boundaries['right'] <= x3]
                                    if np.all((sub_bound['bottom'] <= x1) | (sub_bound['upper'] >= x2) |
                                              (item[1] >= sub_bound['right'])):
                                        # print (item)
                                        boundaries.loc[f_class, 'left'] = item[1]
                                elif item[1] > x4:
                                    sub_bound = boundaries[boundaries['left'] >= x4]
                                    if np.all((sub_bound['bottom'] <= x1) | (sub_bound['upper'] >= x2) |
                                              (item[1] <= sub_bound['left'])):
                                        # print (item)
                                        boundaries.loc[f_class, 'right'] = item[1]
                            join_num = join_num + 1
            new_join_num = join_num
    return boundaries, point_assign, center


def _def_str_mou_on_c_hic(matrix_data, win_n=5, thr_dis=15):
    # matrix size
    mat_size = matrix_data.shape
    print("Matrix size:" + str(mat_size[0]) + '*' + str(mat_size[1]))
    ratio = int(matrix_data.shape[1] // matrix_data.shape[0])
    # computing density threshold
    if ratio == 1:
        # point_num=matrix_data.shape[0] * (2000000/reso)*2
        point_num = matrix_data.shape[0] * thr_dis * 8
    else:
        point_num = matrix_data.shape[0] * 200 / 6

    # print "Effective points:"+str(point_num)
    percent = 1 - point_num / float(matrix_data.shape[0] * matrix_data.shape[1])
    thr = np.percentile(matrix_data, percent * 100)
    win = (win_n, win_n * ratio)
    #    if np.max(matrix_data.shape)<3000:
    #        thr=0
    #    elif np.max(matrix_data.shape)<5000:
    #        thr=np.percentile(matrix_data,95)
    #    else:
    #        thr=np.percentile(matrix_data,99)
    m_density, den_dict = _generate_density_con(matrix_data, win, thr, thr_dis)
    # step 2.2

    ndp_dict, ass_dict = _find_high_points_v2(den_dict, ratio)

    if ratio == 1:
        thr_den = np.percentile(pd.Series(den_dict), 90)
    else:
        thr_den = np.percentile(pd.Series(den_dict), 20)

    # step 2.3
    boundaries, point_assign, centers = _assign_class(den_dict, ndp_dict, ass_dict, thr_den, thr_dis)
    return boundaries


def _return_clusters(df, centers, corr_bound):
    start = {}
    end = {}
    # flag
    old_item = 0
    item = None
    i = None
    for i, item in enumerate(df['point_assign']):
        if item != old_item:
            if old_item != 0:
                end[old_item] = i
            if item != 0 and (item not in start):
                start[item] = i
            old_item = item
    if old_item != 0:
        end[item] = i
    clu_count = 1
    i = 0
    while i < len(corr_bound) and clu_count < len(centers):
        item = corr_bound[i]
        if centers[clu_count - 1] < item < centers[clu_count]:
            if (start[clu_count + 1] - item <= 10) and start[clu_count + 1] - end[clu_count] < 2:
                end[clu_count] = item
                start[clu_count + 1] = item
            clu_count = clu_count + 1
            i = i + 1
        elif item > centers[clu_count]:
            clu_count = clu_count + 1
        else:
            i = i + 1
    cluster = pd.DataFrame({'Start': start, 'End': end}, columns=['Start', 'End'])
    return cluster


def _plot_hic(matrix_data, v_max):
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
    d = {'blue': blue_list,
         'green': green_list,
         'red': red_list}
    my_cmap = c.cmap(d)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('w')
    ax.grid(b=None)
    sns.heatmap(matrix_data.T, vmax=v_max, xticklabels=100, yticklabels=100, cmap=my_cmap, cbar=False)


def density_distance(df, centers):
    x_center = [df.loc[centers[line], 'density'] for line in centers]
    y_center = [df.loc[centers[line], 'distance'] for line in centers]
    fig, ax = plt.subplots(figsize=(6, 6))
    t_df = df.loc[df['point_assign'] == 0]
    index = t_df.index
    x = [t_df.loc[line, 'density'] for line in index]
    y = [t_df.loc[line, 'distance'] for line in index]
    plt.plot(x, y, '.', color='k', markersize=10)
    colors = sns.color_palette("Set1", n_colors=15)
    for i in range(len(centers)):
        t_df = df.loc[df['point_assign'] == i + 1]
        index = t_df.index
        x = [t_df.loc[line, 'density'] for line in index]
        y = [t_df.loc[line, 'distance'] for line in index]
        plt.plot(x, y, '.', color=colors[i % len(colors)], markersize=10)
        plt.plot(x_center[i], y_center[i], 'o', color=colors[i % len(colors)], markersize=16)
    ax.set_facecolor('w')
    ax.grid(b=None)
    plt.show()


def _show_diagonal_result(cluster, matrix_data, thr):
    # matrix_data[matrix_data>thr]=thr
    _plot_hic(matrix_data, thr)
    for i in range(len(cluster)):
        start = cluster.loc[i + 1, 'Start']
        end = cluster.loc[i + 1, 'End']
        # x=[start+0.5,start+0.5,end+0.5]
        # y=[start+0.5,end+0.5,end+0.5]
        x = [start + 0.5, start + 0.5, end + 0.5, end + 0.5, start + 0.5]
        y = [start + 0.5, end + 0.5, end + 0.5, start + 0.5, start + 0.5]
        plt.plot(x, y, '-', color='k', lw=3)
    plt.grid(b=None)
    plt.show()


def _show_chic_cluster_result2(results, matrix_data, thr):
    _plot_hic(matrix_data, thr)
    for i in results.index:
        upper = results.loc[i, 'upper']
        bottom = results.loc[i, 'bottom']
        left = results.loc[i, 'left']
        right = results.loc[i, 'right']
        y_loc = [upper, upper, bottom, bottom, upper]
        x_loc = [left, right, right, left, left]
        plt.plot(x_loc, y_loc, '-', color='k', lw=2.5)
    plt.grid(b=None)
    plt.show()


def _show_chic_cluster_result3(results, matrix_data):
    colors = ['white', 'green', 'blue', 'red']
    if np.max(matrix_data.shape) < 3000:
        thr = np.percentile(matrix_data, 99.5)
    else:
        thr = np.percentile(matrix_data, 99.9)
    matrix_data[matrix_data > thr] = thr
    print(thr)
    red_list = list()
    green_list = list()
    blue_list = list()
    # ['darkblue','seagreen','yellow','gold','coral','hotpink','red']
    for color in colors:
        col = Color(color).rgb
        red_list.append(col[0])
        green_list.append(col[1])
        blue_list.append(col[2])
    c = Colormap()
    d = {'blue': blue_list,
         'green': green_list,
         'red': red_list}
    my_cmap = c.cmap(d)
    plt.subplots(figsize=(8, 8))
    sns.heatmap(matrix_data.T, xticklabels=100, yticklabels=1000, cmap=my_cmap, cbar=False)

    for i in results.index:
        upper = results.loc[i, 'upper']
        bottom = results.loc[i, 'bottom']
        left = results.loc[i, 'left']
        right = results.loc[i, 'right']
        x_loc = [upper, upper, bottom, bottom, upper]
        y_loc = [left, right, right, left, left]
        plt.plot(x_loc, y_loc, '-', color='k', lw=2.5)
    plt.grid(b=None)
    plt.show()


def mask_near(matrix_data, mdhd):
    # mask 2M values
    dsize = matrix_data.shape
    trial_m = np.tril(matrix_data)

    for i in range(dsize[0]):
        for j in range(dsize[1]):
            if (0 <= i - j) & (i - j <= mdhd * 4):
                trial_m[i, j] = 0
    return trial_m


def mstd(matrix_file, output_file, mdhd=5, symmetry=1, window=5, visualization=0):
    if symmetry == 1:
        print("#########################################################################")
        print("Step 0 : File Read ")
        print("#########################################################################")
        matrix_data = np.load(matrix_file, allow_pickle=True)['hic']
        matrix_data[np.isnan(matrix_data)] = 0
        thr = np.percentile(matrix_data, 99.99)
        matrix_data[matrix_data > thr] = thr
        print("Step 0 : Done !!")
        print("#########################################################################")
        print("Step 1: define domain Only diagonal line")
        print("#########################################################################")
        clues = _domain_only_diagonal(matrix_data, window, mdhd)
        clues.to_csv(output_file, sep='\t', index=False)
        if visualization == 1:
            thr = np.percentile(matrix_data, 99.5)
            sns.set_style("ticks")
            _show_diagonal_result(clues, matrix_data, thr)

    if symmetry == 2:
        print("#########################################################################")
        print("Step 0 : File Read ")
        print("#########################################################################")
        matrix_data = np.loadtxt(matrix_file)
        thr = np.percentile(matrix_data, 99.99)
        matrix_data[matrix_data > thr] = thr
        print("Step 0 : Done !!")
        print("#########################################################################")
        print("Step 2: define structure moudle on all points or Capture Hi-C")
        print("#########################################################################")
        boundaries = _def_str_mou_on_c_hic(matrix_data, window, mdhd)
        boundaries.to_csv(output_file, index=False, sep='\t')

        if visualization == 1:
            ratio = matrix_data.shape[1] // matrix_data.shape[0]
            if ratio == 1:
                trial_m = mask_near(matrix_data, mdhd)
                thr = np.percentile(trial_m, 99.5)
                sns.set_style("white")
                _show_chic_cluster_result2(boundaries, trial_m.T, thr)
            else:
                sns.set_style("white")
                _show_chic_cluster_result3(boundaries, matrix_data)


if __name__ == '__main__':
    Matrix_file = f'/home/nkul/Desktop/SR/100/r100_chr4.npz'
    Output_file = f'./TAD_r100_out'
    mstd(Matrix_file, Output_file, mdhd=10, symmetry=1, window=10, visualization=0)

