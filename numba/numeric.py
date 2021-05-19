# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 14:19
import numpy as np
from numba import njit, prange


@njit(parallel=True)
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
    for i in prange(1000000):
        if np.abs((check - temp) / temp) > error:
            check = temp
            temp *= i / (x - 1 + i)
        else:
            d = i
            break
    res = temp * pow(d, x-1)
    return res


@njit
def L2norm(x, y):
    h = (x[-1] - x[0]) / (len(x) - 1)
    y = y ** 2
    L2 = np.sum(y[0:-2] + y[1:-1]) / 2 * h
    return np.sqrt(L2)


@njit
# about np.hstack()
# see https://stackoverflow.com/questions/62173972/numpy-hstack-alternative-for-numba-njit
def diff(x, y, method=2):
    assert method in [2, 3], "method should be 2 or 3"
    h = (x[-1] - x[0]) / (len(x) - 1)
    if method == 2:
        Dy = (y[2:] - y[0:-2]) / 2 / h
        return np.hstack((np.array([0.]), Dy, np.array([(y[-1] - y[-2]) / h])))
    elif method == 3:
        Dy = (y[2:] * 3 - y[1:-1] * 4 + y[0:-2]) / 2 / h
        return np.hstack((np.array([0.]), np.array([(y[-1] - y[0]) / h]), Dy))


@njit
def ceil(x):
    x_int = int(x)
    if x_int < x:
        return x_int + 1
    else:
        return x_int
