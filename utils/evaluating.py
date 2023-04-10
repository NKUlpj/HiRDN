# -*- coding: UTF-8 -*-
"""
@Project ：HiRDN
@File ：evaluating.py
@Author ：nkul
@Date ：2023/4/10 下午2:01
"""


from torchmetrics.functional import mean_absolute_error
from torchmetrics.functional import mean_squared_error
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio


def eval_mae(x, y):
    return mean_absolute_error(x, y).item()


def eval_mse(x, y):
    return mean_squared_error(x, y).item()


def eval_ssim(x, y):
    _ssim = structural_similarity_index_measure(x, y).item()
    return _ssim


def eval_psnr(x, y):
    _psnr = peak_signal_noise_ratio(x, y).item()
    if _psnr < 0:
        _psnr = 0
    return _psnr


def __array_norm_0(x):
    r"""
    norm data to [0, 1]
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def __array_norm_1(x):
    r"""
    norm data to [-1, 1]
    """
    return __array_norm_0(x) * 2 - 1


def eval_lpips(x, y, loss_fn):
    r"""
    https://github.com/richzhang/PerceptualSimilarity
    """
    # x = __array_norm_1(x)
    # y = __array_norm_1(y)
    _lpips = loss_fn(x, y).sum().item()
    if _lpips < 0:
        _lpips = 0
    return _lpips


def eval_dists(x, y, dists_fn):
    r"""
    https://github.com/dingkeyan93/DISTS
    """
    # x = __array_norm_0(x)
    # y = __array_norm_0(y)
    _dists = dists_fn(x, y).sum().item()
    if _dists < 0:
        _dists = 0
    return _dists
