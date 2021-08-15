import numpy as np
import torch
from math import pi


def get_coff(x, y, N):
    """
    :param x: 自变量 Numpy行向量
    :param y: 函数 Numpy行向量
    :param N: Fourier级数拟合项数
    :return: Fourier级数系数 Numpy行向量 0项为常数项
    """
    M = np.size(x)
    h = (x[M - 1] - x[0]) / (M - 1)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    An = torch.zeros(N + 1)
    T = x[M - 1] - x[0]
    if torch.cuda.is_available():
        x = x.to('cuda')
        y = y.to('cuda')
        An = An.to('cuda')

    An[0] = torch.sum((y[0:M - 1] + y[1:M])) / T / 2 * h
    for n in range(1, N + 1):
        y2 = torch.cos(n * pi * x / T)
        An[n] = torch.sum((torch.mul(y[0:M - 1], y2[0:M - 1]) + torch.mul(y[1:M], y2[1:M])) / 2 * h) / T * 2
    if torch.cuda.is_available():
        An = An.cpu()
    An = An.numpy()
    return An


def get_func(x, An):
    """
    :param x: 自变量 Numpy行向量
    :param An: Fourier级数系数 Numpy行向量
    :return: Fourier级数
    """
    N = np.size(An)
    M = np.size(x)
    x = torch.from_numpy(x)
    An = torch.from_numpy(An)
    T = (x[M - 1] - x[0])
    if torch.cuda.is_available():
        x = x.to('cuda')
        An = An.to('cuda')

    y = x * 0 + An[0]
    for n in range(1, N):
        y += torch.cos(n * pi * x / T) * An[n]

    if torch.cuda.is_available():
        y = y.cpu()
    y = y.numpy()
    return y


def get_extended(x, An, T=1, device=torch.device('cpu')):
    """
    :param x: 自变量 Numpy行向量
    :param An: Fourier级数系数 Numpy行向量
    :param T: 周期 正实数
    :param device: torch.device
    :return: Fourier级数
    """

    N = np.size(An)
    x = torch.from_numpy(x).to(device)
    An = torch.from_numpy(An).to(device)

    y = x * 0 + An[0]
    for n in range(1, N):
        y += torch.cos(n * pi * x / T) * An[n]

    return y.cpu().numpy()
