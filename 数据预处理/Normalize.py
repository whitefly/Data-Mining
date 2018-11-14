import numpy as np
from numpy import asanyarray

import matplotlib.pyplot as plt


def zero_normal(data: np.ndarray) -> np.ndarray:
    # 用于0均值化的处理 x=(x-u)/std
    return (data - data.mean(axis=0)) / data.std(axis=0)


def pow(x):
    return x ** 2


if __name__ == '__main__':
    # x坐标组[1,2,3] y坐标组[10,11,12]
    one = list(zip(range(1, 4), range(10, 13)))
    my_data = np.array(one)

    from sklearn.preprocessing import scale

    new_data2 = scale(my_data)


    # new_data = zero_normal(my_data)
    # print(new_data)
    # print(new_data.mean())
    # print(new_data.std())
    #

    # print(new_data2)
    # print(new_data2.mean())
    # print(new_data2.std())
