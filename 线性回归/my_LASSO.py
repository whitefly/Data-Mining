"""
lasso加了L1的正则项,导致loss转为 g(X)+h(X)的凸函数形式(g(X)可微,h(X)不可微)
对于这样的优化,在无法微分的点,采用次微分.普遍采用2种方式,
思入:
1.坐标下降法,每一轮中,固定其他系数,只计算某个系数的解析解.搞定后,按顺序求下一个系数的解析解,直到所有系数遍历完,该轮结束
1.方向试探法:在每一轮中,遍历每个系数,每个系数的变化有2个方向,尝试性的2个方向都走,然后选出最好的方向(loss比变化之前小).然后在此基础下,进行下一轮(迭代次数自己规定)
"""
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class my_lasso:
    def __init__(self, threshold=0.01, t=0.01, iter_n=150):
        self.threshold = threshold  # 迭代停止的阈值
        self.t = t  # L1正则项的λ值
        self.iter_n = iter_n  # 规定的迭代次数

        self.eps = 0.01

        self.coef = None
        self.itercept = None
        self.trained = False

    def set_params(self, threshold=0.01, t=0.01):
        self.threshold = threshold  # 迭代停止的阈值
        self.t = t  # L1正则项的λ值

    @staticmethod
    def rss(_X, _Y, _W):
        temp = _Y - _X * _W  # Y,W均为列向量,防止重复,X变为_X
        return (temp.T * temp)[0, 0]

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        """
        坐标下降法
        """
        row_size, col_size = train_X.shape
        X = np.mat(np.hstack([train_X, np.ones((row_size, 1))]))
        Y = np.mat(train_Y).T
        W = np.mat(np.zeros((col_size + 1, 1)))

        for i in range(self.iter_n):
            total_rss = my_lasso.rss(X, Y, W)
            for k in range(col_size + 1):

                # 遍历每一个属性,计算p1,p2
                fuck = X * W - X[:, k] * W[k, 0]  # 整体-局部
                p1 = ((Y - fuck).T * X[:, k])[0, 0]
                p2 = (X[:, k].T * X[:, k])[0, 0]

                # 根据p1,p2计算修改W
                if p1 > self.t / 2:
                    wk = (p1 - self.t / 2) / p2
                elif p1 < -self.t / 2:
                    wk = (p1 + self.t / 2) / p2
                else:
                    wk = 0
                W[k, 0] = wk

            # 一轮结束,判断rss的减少量是否超过阈值
            temp_rss = my_lasso.rss(X, Y, W)
            delta = total_rss - temp_rss

            if delta <= self.threshold:
                break
            total_rss = temp_rss

        # 赋值参数
        *self.coef, self.itercept = np.asarray(W).flatten()
        self.trained = True

    def fit2(self, train_X: np.ndarray, train_Y: np.ndarray):
        """
        逐步向前回归
        """
        row_size, col_size = train_X.shape
        X = np.mat(np.hstack([train_X, np.ones((row_size, 1))]))
        Y = np.mat(train_Y).T
        W = np.mat(np.zeros((col_size + 1, 1)))
        min_error = np.inf
        W_process = []

        for i in range(self.iter_n):
            for k in range(col_size + 1):
                # 计算每一列的属性
                for sign in [-1, 1]:
                    W_test = W.copy()
                    # 该列的参数,试探性的尝试改变一点(eps)
                    W_test[k, 0] += self.eps * sign
                    test_error = my_lasso.rss(X, Y, W_test)
                    # 计算一下损失函数.若比最小还小,说明方向对了. 对w进行正式修改
                    if test_error < min_error:
                        min_error = test_error
                        W = W_test

            W_process.append(np.asarray(W).flatten())  # 记录W的变化过程

        *self.coef, self.itercept = np.asarray(W).flatten()
        self.trained = True
        return W_process


def normal(X: pd.DataFrame):
    return X - X.mean(axis=0) / X.std(axis=0)


if __name__ == '__main__':
    data = normal(pd.read_csv('../线性回归数据集/广告数据.txt', names=['id', 'tv', 'newspaper', 'radio', 'sales'], index_col='id'))
    # reg = my_lasso()
    # # 画出坐标下降法的轨迹图
    # alphas = np.logspace(-4, 6, 100)
    # coefs = []
    # for alpha in alphas:
    #     reg.set_params(t=alpha)
    #     reg.fit(data[['tv', 'newspaper', 'radio']].values, data['sales'].values)
    #     coefs.append(reg.coef)
    #
    # ax = plt.gca()
    # ax.plot(alphas, coefs)
    # ax.set_xscale('log')
    # plt.title('every alpha shows a LASSO coef ')
    # plt.show()
    # pprint(coefs)

    # 画出 逐步向前的变化图
    reg = my_lasso()
    w_process = reg.fit2(data[['tv', 'newspaper', 'radio']].values, data['sales'].values)
    step = np.arange(reg.iter_n)
    plt.plot(step, w_process)
    plt.show()
