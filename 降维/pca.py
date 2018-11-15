"""
基于numpy实现的pca
为了实现这个,把考研的线代代数整体复习了一般 /(ㄒoㄒ)/~~
"""
import numpy as np
import re
import matplotlib.pyplot as  plt


def loadtxt(name, my_type='col'):
    '''
    将txt转为ndarray,每个数据为一个列向量
    :param my_type:
    :param name: 文件地址
    :return:
    '''
    list_x, list_y = [], []
    with open(name, 'r') as f:
        pat = re.compile(r'\s+')
        for line in f:
            x, y = pat.split(line.strip())
            list_x.append(float(x))
            list_y.append(float(y))
    if my_type == 'col':
        return np.array([list_x, list_y], dtype='float64')
    elif my_type == 'row':
        return np.array([list_x, list_y], dtype='float64').T
    else:
        raise ValueError("请填入col 或者 row")


def my_pac(data, k=1):
    # 降维,默认降维后为1

    # 0均值化(横向广播)
    xy_mean = np.mean(data, axis=1)
    normal_data = data - xy_mean.reshape(-1, 1)

    # 画出均值化后的数据
    axes1 = plt.subplot(312)
    axes1.scatter(normal_data[0], normal_data[1], alpha=0.2, s=5)

    # 计算协方差矩阵(列为每个数据),然后得到其特征向量,并排序得到k_vectors
    cov_m = np.cov(normal_data, rowvar=1)
    vals, vectors = np.linalg.eig(cov_m)
    k_index = np.argsort(vals)[:-(k + 1):-1]
    k_vectors = vectors[:, k_index]

    # 画出投影线(2点确定)
    direction = np.hstack((k_vectors, np.array([[0], [0]])))
    plt.plot(direction[0], direction[1], 'r')

    # 开始投影
    pac_data = np.dot(k_vectors.T, normal_data)
    return pac_data


if __name__ == '__main__':
    data = loadtxt('实战数据集.txt')
    axes1 = plt.subplot(311)
    axes1.scatter(data[0], data[1], alpha=0.2, s=5)

    # 降维后的数据
    new_data = my_pac(data)
    axes2 = plt.subplot('313')
    axes2.scatter(new_data[0], np.full((1, len(new_data[0])), 1), alpha=0.2, s=5)

    plt.show()
