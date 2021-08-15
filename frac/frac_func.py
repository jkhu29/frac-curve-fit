import numpy as np
import torch
import random


def wrs_fun(x, a, b, N):
    """
    :param x: 自变量 Numpy行向量
    :param a: Weierstrass函数参数λ (1,∞)实数
    :param b: Weierstrass函数参数α (0,1]实数
    :param N: 级数项数 正整数
    :return: Weierstrass函数 Numpy行向量
    """
    x = torch.from_numpy(x)
    if torch.cuda.is_available():
        x = x.to('cuda')
    y = x * 0

    for i in range(1, N+1):
        y += a ** (-b * i) * torch.sin(x * a ** i)

    if torch.cuda.is_available():
        y = y.cpu()
    y = y.numpy()
    return y


def bes_fun(x, a, b):
    """
    :param x: 自变量 Numpy行向量
    :param a: Besicovitch函数参数λj 一维类等比数组
    :param b: Besicovitch函数参数α (0,1]实数
    :return: Besicovitch函数 Numpy行向量
    """
    x = torch.from_numpy(x)
    if torch.cuda.is_available():
        x = x.to('cuda')
    N = len(a)
    y = x * 0

    for i in range(1, N+1):
        y += a[i-1] ** (-b * i) * torch.sin(x * a[i-1] ** i)

    if torch.cuda.is_available():
        y = y.cpu()
    y = y.numpy()
    return y


def rand_frac(x, C):
    """
    :param x: 自变量 Numpy行向量
    :param C: 随机数范围 正实数
    :return: 随机分形函数 Numpy行向量
    """
    M = np.size(x)
    y = x * 0

    for i in range(1, M):
        y[i] = y[i-1] + C * (random.random() - 0.5)

    return y
