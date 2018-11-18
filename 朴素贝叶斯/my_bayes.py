"""
算法实现:
由于和决策树一样,频繁使用出现的频率,所以尝试使用pandas来计算贝叶斯的先验分布
"""

import numpy as np
import pandas as pd


class My_Bayes:
    def __init__(self):
        self.pro_Y = None  # Y的频率
        self.pro_X = None  # X的频率
        self.label = None  # X的列名
        self.cols = None  # Y的列名
        self.trained = False

    def train(self, some: pd.DataFrame):
        """
            根据数据集,初始化贝叶斯的先验概率
            :param some:
            :return:
            """
        self.cols, self.label = some.columns[:-1], some.columns[-1]
        # 计算Y的频率
        self.pro_Y = some[self.label].value_counts(normalize=True)

        # 聚合,有几个属性就聚合几次,最后保存先验概率信息为self.pro_X
        my_dict = {}
        for col_name in self.cols:
            k = some.groupby([col_name, self.label]).size().unstack(0)
            k = k.divide(k.sum(axis=1), axis=0)  # 计算频率
            my_dict[k.columns.name] = k
        result = pd.concat(my_dict, axis=1)
        result.columns.names = ['属性', '属性值']

        self.trained = True
        self.pro_X = result

    def predict(self, one: pd.Series):
        if not self.trained:
            raise Exception("训练器未训练,无法预测!")

        result_pro = -1
        result_label = None
        # 遍历可能的Y值,找到最大值概率值输出
        for y_value, y_pro in self.pro_Y.iteritems():
            one_pro = y_pro
            for name, value in one.iteritems():
                one_pro *= self.pro_X[name].at[y_value, value]
            if one_pro > result_pro:
                result_pro = one_pro
                result_label = y_value
        return result_label, result_pro


if __name__ == '__main__':
    clf = My_Byes()
    data = pd.read_csv('../朴素贝叶斯数据集/统计学习方法.txt')
    clf.train(data)

    new_one = pd.Series([2, 'S'], index=['X1', 'X2'])
    my_result = clf.predict(new_one)
    print(my_result)
