import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


def zero_normal(data: np.ndarray) -> np.ndarray:
    # 用于0均值化的处理 x=(x-u)/std
    return (data - data.mean(axis=0)) / data.std(axis=0)


def center_normal(data: np.ndarray) -> np.ndarray:
    return data - data.mean(axis=0)


def pow(x):
    return x ** 2


if __name__ == '__main__':
    pass
