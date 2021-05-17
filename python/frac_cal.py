# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 16:24
import numpy as np
from numeric import gamma, diff, ceil


def frac_int(x, y, v):
    """
    :param x: 自变量 Torch/Numpy行向量
    :param y: 函数 Torch/Numpy行向量
    :param v: 分数阶积分阶数 正实数
    :return: y的分数阶积分 Torch/Numpy行向量
    """
    assert v > 0, "The order of the fractional integral must be positive."
    h = (x[-1] - x[0]) / (len(x) - 1)
    Iy = np.zeros_like(y)
    Iy[1] = y[1] / 2 / v * h ** (v+1)
    for i in range(2, len(x)):
        L1 = np.ones(i - 1) * (i + 1)
        L2 = np.linspace(1, i, i)
        y1 = ((L1 * h - L2[1:i] * h) ** (v-1) * y[1:i] + (L1 * h - L2[0:i-1] * h) ** (v-1) * y[0:i-1]) / 2 * h
        y2 = (y[i] + y[i-1]) / 2 / v * h ** v
        Iy[i] = np.sum(y1) + y2
    Iy /= gamma(v)
    return Iy


def frac_diff(x, y, u):
    """
    :param x: 自变量 Numpy行向量
    :param y: 函数 Numpy行向量
    :param u: 分数阶微分阶数 实数
    :return: y的分数阶微分 Numpy行向量
    """
    if u <= 0:
        return frac_int(x, y, -u)
    else:
        n = ceil(u)
        y = frac_int(x, y, n-u)
        for i in range(n):
            y = diff(x, y, method=2)
        return y
