import math
import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)


def test_frac_int(x, y, v):
    N = np.size(x)
    t = np.linspace(0, N, N) ** v
    t = -t[0:-1] + t[1:]
    # print(t)
    Iy = y * 0
    Iy[1] = y[1] / 2 / v * (x[1] - x[0]) ** (1 + v)
    l1 = np.insert((y[0:-1] + y[1:])[::-1] / 2 * t, 0, 0)
    # print(l1)
    for i in range(2, N):
        Iy[i] = np.sum(l1[0:i]) / (i + 1) ** v * x[i] ** v
    Iy /= math.gamma(v + 1)
    return Iy


def frac_int(x, y, v):
    """
    :param x: 自变量 Numpy行向量
    :param y: 函数 Numpy行向量
    :param v: 分数阶积分阶数 正实数
    :return: y的分数阶积分 Numpy行向量
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = np.size(x)
    h = (x[M - 1] - x[0]) / (M - 1)
    y = torch.from_numpy(y).to(device)

    Iy = y * 0
    Iy[1] = y[1] / 2 / v * h ** v
    for i in range(2, M):
        L1 = torch.linspace(i + 1, i + 1, i - 1).to(device)
        L2 = torch.linspace(1, i, i).to(device)
        y1 = (torch.mul((L1 * h - L2[1:i] * h)
                        ** (v - 1), y[1:i])
              + torch.mul((L1 * h - L2[0:i - 1] * h) ** (v - 1), y[0:i - 1])) / 2 * h
        y2 = (y[i] + y[i - 1]) / 2 / v * h ** v
        Iy[i] = torch.sum(y1) + y2
    Iy = Iy / math.gamma(v)
    Iy = Iy.cpu()
    Iy = Iy.numpy()
    return Iy


def frac_dif(x, y, u, ans):
    """
    :param x: 自变量 Numpy行向量
    :param y: 函数 Numpy行向量
    :param u: 分数阶微分阶数 [0,1)实数
    :param ans: 真实值
    :return: y的分数阶微分 Numpy行向量
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    M = np.size(x)
    h = x[1] - x[0]
    v = 1 - u

    y = frac_int(x, y, v)
    # y = test_frac_int(x, y, v)
    y = torch.from_numpy(y)
    if torch.cuda.is_available():
        y = y.to('cuda')
    dy = (y[2:] - y[:-2]) / 2 / h
    y = torch.hstack((torch.tensor(0).to(device), dy, torch.tensor(ans[-1]).to(device)))
    if torch.cuda.is_available():
        y = y.cpu()
    y = y.numpy()
    return y
