# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 14:02
import numpy as np
from numba import njit


@njit
def MSELoss(x1, x2):
    assert len(x1) == len(x2), "The length of the input vectors should be equal, but get " + str(
        len(x1)) + " and " + str(len(x2))
    return np.sum((x1 - x2) ** 2) / len(x1)


@njit
def RMSELoss(x1, x2):
    return np.sqrt(MSELoss(x1, x2))


@njit
def MAELoss(x1, x2):
    assert len(x1) == len(x2), "The length of the input vectors should be equal, but get " + str(
        len(x1)) + " and " + str(len(x2))
    return np.sum(np.abs(x1, x2)) / len(x1)
