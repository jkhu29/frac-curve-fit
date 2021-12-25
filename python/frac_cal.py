# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 16:24
import math
import numpy as np
from numeric import gamma, diff, ceil


def frac_int(x, y, v):
    """
    :param x: 自变量 Numpy行向量
    :param y: 应变量 Numpy行向量
    :param v: 分数阶积分阶数 正实数
    :return : y的分数阶积分 Numpy行向量
    """
    assert v > 0, "The order of the fractional integral must be positive."
    length = x.shape[0]
    # h = (x[-1] - x[0]) / (length - 1)
    h = (np.max(x) - np.min(x)) / (length - 1)
    if isinstance(h, np.ndarray):
        # TODO(JiaKui Hu): about h[i](💫)
        for i in range(len(h)):
            h[i] = (np.max(x[:, i]) - np.min(x[:, i])) / (length - 1)

    Iy = np.zeros_like(x)
    Iy2 = np.zeros_like(y)
    if isinstance(h, np.ndarray):
        for i in range(len(h)):
            Iy[1, i] = y[1] / 2 / v * (math.pow(h[i], v+1))
    else:
        Iy2[1] = y[1] / 2 / v * h ** (v+1)

    if isinstance(h, np.ndarray):
        for j in range(len(h)):
            for i in range(2, length):
                L1 = np.ones(i - 1) * (i + 1)
                L2 = np.linspace(1, i, i)
                y1 = (L1 * h[j] - L2[1:i] * h[j]) ** (v-1) * y[1:i] / 2 * h[j] + \
                     (L1 * h[j] - L2[0:i-1] * h[j]) ** (v-1) * y[0:i-1] / 2 * h[j]
                y2 = (y[i] + y[i-1]) / 2 / v * h[j] ** v
                Iy[i, j] = np.sum(y1) + y2
        for i in range(length):
            # TODO(JiaKui Hu): simple add?(💫💫💫)
            Iy2[i] = np.sum(Iy[i, :])
    else:
        for i in range(2, length):
            L1 = np.ones(i - 1) * (i + 1)
            L2 = np.linspace(1, i, i)
            y1 = ((L1 * h - L2[1:i] * h) ** (v-1) * y[1:i] + (L1 * h - L2[0:i-1] * h) ** (v-1) * y[0:i-1]) / 2 * h
            y2 = (y[i] + y[i-1]) / 2 / v * h ** v
            Iy2[i] = np.sum(y1) + y2

    Iy2 /= gamma(v)
    return Iy2


def frac_diff(x, y, u):
    """
    :param x: 自变量 Numpy行向量
    :param y: 应变量 Numpy行向量
    :param u: 分数阶微分阶数 实数
    :return : y的分数阶微分 Numpy行向量
    """
    if u <= 0:
        return frac_int(x, y, -u)
    else:
        n = ceil(u)
        y = frac_int(x, y, n-u)
        for _ in range(n):
            y = diff(x, y, method=2)
        return y
