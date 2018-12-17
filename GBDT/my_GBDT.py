"""
GBDT的实现,看起来和单颗回归树差不多.
对于一维回归来说,只不过每次的y会变为上一轮的残差. 然后根据这些y来计算
思入: 每轮使用单个树桩,迭代次数默认为100. 代码大致框架和my_AdaBoost差不多
基于numpy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class my_GBDT:
    def __init__(self):
        self.container = []
        self.X = None
        self.Y = None

    def predict(self, data, col_index, threshold, bin_y) -> np.ndarray:
        """
        由于是回归,且为2分回归,传入阈值和 2个y值[y1,y2],<阈值:y1,≥阈值:为y2
        :param data: X
        :param col_index:
        :param threshold: 某属性的X阈值
        :param bin_y:  二分回归的预测值
        :return: y1,y2组成的行向量
        """
        y1, y2 = bin_y
        result = np.full(data.shape[0], y1)
        result[data[:, col_index] >= threshold] = y2
        return result

    def split(self, data, Y, col_index, threshold):
        bool_index = (data[:, col_index] < threshold)
        return data[bool_index, :], data[~bool_index, :], Y[bool_index], Y[~bool_index]

    def error(self, Y, predict_Y):
        # 用于计算 损失和
        return np.square(Y - predict_Y).sum()

    def create_single_node(self, data: np.ndarray, Y: np.ndarray):
        """
        将选中的属性值分为20分,遍历属性,找到最佳的划分点,返回残差
        :param Y:
        :param data:
        :return: 树桩节点的属性,本轮的残差
        """

        blocks = 20
        min_error = np.inf
        best_feature = {}
        rss = None
        for index in range(data.shape[1]):
            min_v, max_v = data[:, index].min(), data[:, index].max()
            step = (max_v - min_v) / blocks

            for i in range(-1, blocks + 1):
                candidate = i * step + min_v
                X1, X2, Y1, Y2 = self.split(data, Y, index, candidate)  # 分离数据集X ->(x1,y1)    (x2,y2)
                if X1.shape[0] and X1.shape[0]:
                    c1, c2 = Y1.mean(), Y2.mean()

                    error1 = self.error(Y1, np.array([c1] * X1.shape[0]))
                    error2 = self.error(Y2, np.array([c2] * X2.shape[0]))
                    now_error = error1 + error2

                    predict_Y = self.predict(X, index, candidate, (c1, c2))
                    if now_error < min_error:
                        min_error = now_error
                        best_feature['index'] = index
                        best_feature['value'] = candidate
                        best_feature['c'] = (c1, c2)  # 2分的y值
                        rss = Y - predict_Y
        return min_error, best_feature, rss

    def fit(self, data: np.ndarray, Y: np.ndarray, size=6):
        last_rss = Y  # last_rss为为上一轮的残差,核心训练代码比Adaboost还简答~_~
        for i in range(size):
            one_error, best_feature, rss = self.create_single_node(data, last_rss)
            last_rss = rss
            self.container.append(best_feature)

        self.X = data
        self.Y = Y

    def plot(self):
        # 本函数仅仅用来可视化回归线(只限于2维使用)
        T = [(d['c'][0], d['c'][1], d['value']) for d in self.container]
        extra = 4

        # 得到顺序范围数组
        my_list = set([X for _, _, X in T])
        my_list = sorted(list(my_list))
        mapping = dict([(v, i) for i, v in enumerate(my_list)])

        # 填充每个范围对应的固定值
        result = [0] * (len(my_list) + 1)
        for c1, c2, X in T:
            index = mapping[X]
            for i in range(len(result)):
                result[i] += (c1 if i <= index else c2)

        # 生成3元组, x_下限,x_上限,固定y值
        x_down = [my_list[0] - extra] + my_list
        x_up = my_list + [my_list[-1] + extra]
        scope_M = np.array([x_down, x_up, result]).T

        # 画图,散点+回归横线,
        print("最后的回归方程:\nx_下限  x_上限  回归值\n", scope_M)
        plt.scatter(self.X, self.Y)
        for row in scope_M:
            y = row[-1]
            plt.plot(row[:2], [y] * 2, 'r')
        plt.show()


if __name__ == '__main__':
    all = pd.read_csv('../GBDT_数据集/统计学习方法.csv')
    X = all[['x']].values
    Y = all['y'].values
    dt = my_GBDT()
    dt.fit(X, Y)
    dt.plot()
