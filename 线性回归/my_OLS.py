'''
最小二乘法:实现比较简单,可以直接求出系数w的解析解.
基于numpy
'''
import numpy as np
import pandas as pd


class Line_Regressor:
    def __init__(self):
        self.W = None
        self.trained = False
        self.coef = None
        self.intercept = None

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        """
        直接公式得到解析解
        :param X:  为n*m矩阵, n为样本个数,m为列个数
        :param Y:  和sklearn一样,为一维行向量
        :return:
        """
        row_size, col_size = train_X.shape
        Y = np.mat(train_Y.reshape(-1, 1))  # 列向量
        X = np.mat(np.hstack([train_X, np.ones((row_size, 1))]))  # 增加一个截距列

        # 训练代码就一行
        self.W = (X.T * X).I * X.T * Y

        temp = np.asarray(self.W).flatten()
        self.coef = temp[:-1]
        self.intercept = temp[-1]

        self.trained = True

    def predict(self):
        if not self.trained:
            raise Exception("未训练,无法预测")


if __name__ == '__main__':
    # data = np.loadtxt('../线性回归数据集/广告数据.txt', delimiter=',')
    # my_X = data[:, 1:-1]
    # my_Y = data[:, -1]
    # reg = Regressor()
    # reg.fit(my_X, my_Y)
    # print(reg.coef)
    # print(reg.intercept)

    # 用二维点来可视化
    # reg.fit(my_X[:, [0]], my_Y)
    # plt.scatter(my_X[:, 0], my_Y, alpha=0.7)
    # demo_x = np.linspace(0, 300, 450)
    # plt.plot(demo_x, demo_x * reg.coef + reg.intercept, 'r')
    # plt.show()

    # 复现奇异阵出现的错误
    data = pd.read_csv('../线性回归数据集/奇异值数据.txt')
    my_X = data.iloc[:, 1:-1].values
    my_Y = data.iloc[:, -1].values
    print(data.head())
    reg = Line_Regressor()
    reg.fit(my_X, my_Y)
    print(reg.coef)
    print(reg.intercept)
