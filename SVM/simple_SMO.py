import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


df = pd.read_csv('../SVM_数据集/非线性数据.csv', names=['X1', 'X2', 'label'])
fg = sns.FacetGrid(data=df, hue='label')
fg.map(plt.scatter, 'X1', 'X2').add_legend()
plt.show()


class Simple_linerSVM:

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray, iter_n, C):
        """
        思想: 使用简化的smo算法来找出最优的α
        :param train_X: 每行数据为行向量
        :param train_Y: 一维行向量
        :param iter_n: 迭代次数
        :param C: 固定距离
        :return:
        """
        row_size, col_size = train_X.shape
        X = np.mat(train_X)
        Y = np.mat(train_Y).T
        # 初始化α
        alphas, b = np.mat(np.zeros(row_size, 1)), 0

        # 开始迭代
        iter_id = 1
        while iter_id <= iter_n:
            # 寻找最坏的x点,(一个x对应一个alpha)
            for id in row_size:
                # 根据对偶公式,用α来表示w
                W = X.T * np.multiply(alphas, Y)
                pro_Y = X[id] * W + b

                Y_i = Y[id, 0]
                alpha_i = alphas[id]
                error = pro_Y - Y_i  # 单个y的差值

                # 是否符合约束条件,
                # 需要≥c
                # α≥0


if __name__ == '__main__':
    pass
    # import base64
    #
    # print(base64.b64decode('OTkwMDIwMTg2QHFxLmNvbQ=='))
