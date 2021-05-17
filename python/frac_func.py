# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 15:39
import numpy as np


def wrs_func(x, a, b, N):
    """
    :param x: 自变量 Numpy行向量
    :param a: Weierstrass函数参数λ (1,∞)实数
    :param b: Weierstrass函数参数α (0,1]实数
    :param N: 级数项数 正整数
    :return: Weierstrass函数 Numpy行向量
    """
    y = np.zeros_like(x)
    for i in range(1, N+1):
        y += a ** (-b * i) * np.sin(x * a ** i)
    return y


def bes_func(x, a, b):
    """
    :param x: 自变量 Numpy行向量
    :param a: Besicovitch函数参数λj 一维类等比数组
    :param b: Besicovitch函数参数α (0,1]实数
    :return: Besicovitch函数 Numpy行向量
    """
    y = np.zeros_like(x)

    for i in range(len(x)):
        y += a[i] ** (-b * (i+1)) * np.sin(x * a[i] ** (i+1))
    return y


def rand_func(x, C):
    """
    :param x: 自变量 Numpy行向量
    :param C: 随机数范围 正实数
    :return: 随机分形函数 Numpy行向量
    """
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = y[i-1] + C * (np.random.random() - 0.5)
    return y
