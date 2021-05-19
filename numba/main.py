# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 18:44
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from frac_func import wrs_func
from frac_cal import frac_diff
from fouier_ser import *
from numeric import *


@njit
def main():
    N: int = 20
    M: int = 10 ** 5
    # let M bigger in `./numba/main.py`
    x = np.linspace(0, 1, M+1)
    a = 5
    b = 0.5

    y = wrs_func(x, a, b, N)
    Iy = frac_diff(x, y, -0.5)
    An = get_coff(x, Iy, 5000)
    F = get_func(x, An)
    f = frac_diff(x, F, 0.5)

    print(L2norm(x, f-y))
    return x, f, y


if __name__ == "__main__":
    x, f, y = main()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.ylim(-1, 1)
    plt.plot(x, y, ls='-', lw=0.3, color='black')
    plt.subplot(1, 2, 2)
    plt.ylim(-1, 1)
    plt.plot(x, f, ls='-', lw=0.3, color='blue')
    plt.show()
