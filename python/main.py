# ——*——coding:utf-8——*——
# author: hhhfccz time:2021/5/17 18:44
import numpy as np
import matplotlib.pyplot as plt

from frac_func import *
from frac_cal import *
from fouier_ser import *
from numeric import *
from loss import * 


# init for wrs_func
N: int = 20
M: int = 10 ** 4
x = np.linspace(0, 1, M+1)
a = 5
b = 0.5

y = wrs_func(x, a, b, N)    # 维尔斯特拉斯函数数据生成
# y = rand_func(x, b)       # 随机分形数据生成

###########
# y-x 回归
###########
Iy = frac_diff(x, y, -0.5)  # 得到分数阶积分
# An = get_coff(x, Iy, 5000)  # 得到傅里叶级数系数，解析式
# F = get_func(x, An)         # 解析式对应的 (x, y)
# f = frac_diff(x, F, 0.5)    # 分数阶微分
f = frac_diff(x, Iy, 0.5)    # 分数阶微分
print(y)
print(f)
print("distance: {}".format(L2norm(x, f-y)))
print("MSE: {}".format(MSELoss(f, y)))

# plt.figure()
# plt.subplot(1, 2, 1)
# # plt.ylim(-1, 1)
# plt.plot(x, y, ls='-', lw=0.5, color='black')
# plt.subplot(1, 2, 2)
# # plt.ylim(-1, 1)
# plt.plot(x, f, ls='-', lw=0.5, color='blue')
# plt.show()

############
# y-x 自回归
############
before = 1
after = 1
# f(y[n-before], ..., y[n-1], y[n+1], ..., y[n+after]) = y[n]
y2 = np.zeros((before+after+1, len(x)))
for i in range(before, len(x)-after):
    y2[0, i] = y[i-1]
    y2[1, i] = y[i+1]
    y2[2, i] = y[i]
y2 = y2[:, before:-after]

x = y2[0:-1, :].T  # shape: (num, before+after)
y = y2[2, :].T     # shape: (num, )

Iy = frac_diff(x, y, -0.5)  # 得到分数阶积分
f = frac_diff(x, Iy, 0.5)    # 分数阶微分
print("\n")
print(y)
print(f)
print("distance: {}".format(L2norm(x, f-y)))
print("MSE: {}".format(MSELoss(f, y)))
