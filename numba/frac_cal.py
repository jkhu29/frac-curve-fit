# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 16:24
import numpy as np
from numba import njit, prange
from numeric import gamma, diff, ceil


@njit(parallel=True)
def frac_int(x, y, v):
    """
    :param x: 自变量 Numpy行向量
    :param y: 函数 Numpy行向量
    :param v: 分数阶积分阶数 正实数
    :return: y的分数阶积分 Numpy行向量
    """
    assert v > 0, "The order of the fractional integral must be positive."
    h = (x[-1] - x[0]) / (len(x) - 1)
    Iy = np.zeros_like(y)
    Iy[1] = y[1] / 2 / v * h ** (v+1)
    for i in prange(2, len(x)):
        L1 = np.ones(i - 1, dtype=np.int64) * (i + 1)
        L2 = np.linspace(1, i, i)
        y1 = ((L1 * h - L2[1:i] * h) ** (v-1) * y[1:i] + (L1 * h - L2[0:i-1] * h) ** (v-1) * y[0:i-1]) / 2 * h
        y2 = (y[i] + y[i-1]) / 2 / v * h ** v
        Iy[i] = np.sum(y1) + y2
    Iy /= gamma(v)
    return Iy


@njit(parallel=True)
def frac_diff(x, y, u):
    """
    :param x: 自变量 Numpy行向量
    :param y: 函数 Numpy行向量
    :param u: 分数阶微分阶数 实数
    :return: y的分数阶微分 Numpy行向量
    :note: about double-for see -> https://github.com/numba/numba/issues/4116
    """
    if u <= 0:
        return frac_int(x, y, -u)
    else:
        n = ceil(u)
        y = frac_int(x, y, n-u)
        k = len(y)
        y_all = np.zeros((n+1, k), dtype=np.float64)
        y_all[0, :] = y
        for i in prange(1, n+1):
            y = diff(x, y_all[i-1, :], method=2)
            for j in prange(k):
                y_all[i, j] = y[j]
            # when you use `y = function(y)`, numba will be confused
            # it'll raise "Use of reduction variable y in an unsupported reduction function."
        return y_all[-1, :]
