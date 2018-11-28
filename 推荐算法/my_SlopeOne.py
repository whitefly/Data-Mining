"""
加权的slope_one算法
基于pandas
"""

import pandas as pd
import numpy as np


class my_SlopeOne:
    def __init__(self):
        self.user_item = None  # user_item矩阵
        self.itemGap = None  # gap矩阵
        self.weight = None  # gap的权重矩阵

        self.trained = False

    def fit(self, DF: pd.DataFrame):
        """
        生成 item的gap矩阵 和 gap矩阵的权重矩阵
        :param DF:  传入user_item矩阵
        :return:
        """
        self.user_item = DF
        col_names = DF.columns
        row_size, col_size = DF.shape
        gap_M, weight_M = np.zeros((col_size, col_size)), np.zeros((col_size, col_size))

        for i in range(col_size):
            for j in range(i + 1, col_size):
                # temp为2个属性的无缺失值的DF
                temp = DF.iloc[:, [i, j]].dropna()
                pair_gap = (temp.iloc[:, 0] - temp.iloc[:, 1]).mean()  # 电影1-电影2的均值
                pair_size = temp.shape[0]

                # 填充矩阵
                gap_M[i, j] = pair_gap
                gap_M[j, i] = -pair_gap
                weight_M[i, j] = pair_size
                weight_M[j, i] = pair_size

        weight_M /= row_size
        self.itemGap = pd.DataFrame(gap_M, columns=col_names, index=col_names)
        self.weight = pd.DataFrame(weight_M, columns=col_names, index=col_names)
        self.trained = True

    def predict(self, S: pd.Series, predict_name):
        if not self.trained:
            raise EOFError("未进行训练,无法预测!")

        names = S.dropna().index
        result_rate = 0
        # names为该用户评价过的所有电影名
        for name in names:
            # 加权求和
            temp_score = (S[name] - self.itemGap.at[name, predict_name]) * self.weight.at[name, predict_name]
            result_rate += temp_score
        return result_rate


if __name__ == '__main__':
    demo = pd.read_csv('../推荐算法_数据集/简单电影评分.txt', na_values='?', index_col='用户')

    user = demo.loc['用户3']
    want_name = '东游'
    slope = my_SlopeOne()
    slope.fit(demo)
    print(slope.predict(user, want_name))
