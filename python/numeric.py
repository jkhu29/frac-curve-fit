# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 14:19
import math
import numpy as np


def gamma(x, error=1e-5):
    if x < 0:
        return gamma(x + 1, error) / x
    elif x > 1:
        return (x - 1) * gamma(x - 1, error)
    elif np.abs(1.0 - x) < 1e-5:
        return 1.
    elif np.abs(0.5 - x) < 1e-5:
        return np.sqrt(np.pi)

    res, temp, check = 0., 1., 0.
    i: int = 1
    while np.abs((check - temp) / temp) > error:
        check = temp
        temp *= i / (x - 1 + i)
        i += 1
    res = temp * pow(i, x-1)
    return res


def L2norm(x, y):
    length = x.shape[0]
    h = (x[-1] - x[0]) / (len(x) - 1)
    if isinstance(h, np.ndarray):
        for i in range(len(h)):
            h[i] = (np.max(x[:, i]) - np.min(x[:, i])) / (length - 1)
    y = y ** 2
    if isinstance(h, np.ndarray):
        L2 = np.zeros_like(h)
        for i in range(len(h)):
            L2[i] = np.sum(y[0:-2] + y[1:-1]) / 2 * h[i]
    else:
        L2 = np.sum(y[0:-2] + y[1:-1]) / 2 * h
    return np.sqrt(L2)


def diff(x, y, method=2):
    assert method in [2, 3], "method should be 2 or 3"
    length = x.shape[0]
    h = (x[-1] - x[0]) / (length - 1)
    Dy = np.zeros_like(x)
    Dy2 = np.zeros_like(y)
    if method == 2:
        if isinstance(h, np.ndarray):
            for i in range(len(h)):
                temp = (y[2:] - y[0:-2]) / 2 / h[i]  # shape: (length-2, )
                Dy[:, i] = np.hstack((0, temp, (y[-1] - y[-2]) / h[i]))
            for i in range(length):
                Dy2[i] = np.sum(Dy[i, :])
            return Dy2
        else:
            temp = (y[2:] - y[0:-2]) / 2 / h
            Dy = np.hstack((0, temp, (y[-1] - y[-2]) / h))
            return Dy
    elif method == 3:
        Dy = (y[2:] * 3 - y[1:-1] * 4 + y[0:-2]) / 2 / h
        return np.hstack((0, (y[-1] - y[0]) / h, Dy))


def ceil(x):
    x_int = int(x)
    if x_int < x:
        return x_int + 1
    else:
        return x_int
