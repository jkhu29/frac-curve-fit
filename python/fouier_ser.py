# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 17:54
import numpy as np


def get_coff(x, y, N):
    """
    :param x: 自变量 Numpy行向量
    :param y: 函数 Numpy行向量
    :param N: Fourier级数拟合项数
    :return: Fourier级数系数 Numpy行向量 0项为常数项
    """
    T = x[-1] - x[0]
    h = T / (len(x) - 1)
    An = np.zeros(N+1)
    An[0] = np.sum((y[0:-1] + y[1:])) / T / 2 * h
    for i in range(N):
        y2 = np.cos(i * np.pi * x / T)
        An[i] = np.sum((y[0:-1] * y2[0:-1] + y[1:] * y2[1:]) / 2 * h) / T * 2
    return An


def get_func(x, An):
    """
    :param x: 自变量 Numpy行向量
    :param An: Fourier级数系数 Numpy行向量
    :return: Fourier级数
    """
    T = x[-1] - x[0]
    y = np.ones_like(x) * An[0]
    for i in range(1, len(An)):
        y += np.cos(i * np.pi * x / T) * An[i]
    return y
