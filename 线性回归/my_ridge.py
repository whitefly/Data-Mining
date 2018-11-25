"""
岭回归: 在线性回归的基础上加一个L2惩罚项,
作用:
1.将奇异值变为非奇异阵.
2.缩减系数的大小
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Ridge_Regressor:
    def __init__(self, alpha=0.1):
        self._trained = False
        self.coef = None
        self.intercept = None
        self.alpha = alpha

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray):
        """
        直接公式得到解析解
        :param train_Y: 和sklearn一样,为一维行向量
        :param train_X: 为n*m矩阵, n为样本个数,m为列个数
        :return:
        """
        row_size, col_size = train_X.shape
        Y = np.mat(train_Y.reshape(-1, 1))  # 列向量
        X = np.mat(np.hstack([train_X, np.ones((row_size, 1))]))  # 增加一个截距列

        eye = np.mat(np.eye(col_size + 1))
        # 训练代码就一行
        W = (X.T * X + self.alpha * eye).I * X.T * Y

        temp = np.asarray(W).flatten()
        self.coef = temp[:-1]
        self.intercept = temp[-1]

        self._trained = True

    def predict(self):
        if not self._trained:
            raise Exception("未训练,无法预测")


if __name__ == '__main__':
    data = pd.read_csv('../线性回归_数据集/广告数据.txt')
    my_X = data.iloc[:, 1:-1].values
    my_Y = data.iloc[:, -1].values

    # 显示岭迹
    alphas = np.logspace(-2, 8, 40)
    coefs = []
    for alpha in alphas:
        reg = Ridge_Regressor(alpha=alpha)
        reg.fit(my_X, my_Y)
        coefs.append(reg.coef)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.title('every alpha shows a Ridge coef ')
    plt.show()
