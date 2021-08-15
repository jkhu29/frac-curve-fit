import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from frac_func import wrs_fun
from frac_cal import frac_int, frac_dif, test_frac_int
import four_ser as fs
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


def mse(x, y):
    return ((x - y) ** 2).mean()


def rmse(x, y):
    return mse(x, y) ** 0.5


def mae(x, y):
    return (np.abs(x-y)).mean()


N: int = 20
M: int = 10 ** 4 + 1
x = np.linspace(0, 1, M)
a = 5
b = 0.5

for b in range(1, 10, 1):
    b = b / 10
    print(b)
    y = wrs_fun(x, a, b, N)
    Iy = frac_int(x, y, 1-b)
    # t = interpolate.splrep(x, Iy)
    # f = interpolate.interp1d(x, Iy, kind="cubic")  # 3次样条插值

    # xx = np.linspace(0, 1, 2*10**4 + 1)  # 插值测试集的横坐标
    # yy = wrs_fun(xx, a, b, N)  # 插值测试集纵坐标
    # tt = frac_dif(xx, f(xx), 0.5, yy)  # 插值结果纵坐标
    #
    # _f = interpolate.interp1d(x, y, kind="cubic")  # 直接插值解析式
    # _tt = _f(xx)  # 直接插值纵坐标
    An = fs.get_coff(x, Iy, 1000)
    F = fs.get_func(x, An)
    tt = frac_dif(x, F, 1-b, y)
    _tt = fs.get_func(x, fs.get_coff(x, y, 1000))

    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # axins = inset_axes(ax, width="30%", height="30%", loc='lower left', bbox_to_anchor=(0.3, 0.1, 1, 1),bbox_transform=ax.transAxes)
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, Iy, ls='-', lw=0.5, color='black')
    # plt.savefig('./int_wrs_fun.jpg')

    # ax.plot(xx, yy, ls='-', lw=0.3, color='black')
    # ax.plot(xx, tt, ls='-', lw=0.3, color='blue')
    # ax.plot(xx, _tt, ls='-', lw=0.3, color='red')
    # axins.plot(xx, y, ls='-', lw=1, color='black')
    # axins.plot(xx, tt, ls='-', lw=1, color='blue')
    # axins.plot(xx, _tt, ls='-', lw=1, color='red')

    # 调整子坐标系的显示范围
    # axins.set_xlim(0.485, 0.495)
    # axins.set_ylim(0.07, 0.13)
    # axins.set_xlim(0.488, 0.490)
    # axins.set_ylim(0.09, 0.13)

    # axins.set_xlim(0.95, 1.1)
    # axins.set_ylim(-0.8, -0.15)

    # plt.show()
    # fig.savefig('./extend.jpg')
    print('=====MSE====')
    print(mse(y, tt))
    print(mse(y, _tt))
    print('=====RMSE====')
    print(rmse(y, tt))
    print(rmse(y, _tt))
    print('=====MAE=====')
    print(mae(y, tt))
    print(mae(y, _tt))
