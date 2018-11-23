"""
ID3算法无法处理连续数据,也无法做回归.
算法实现:CART中,将所有数据都作为连续值,然后通过二分划分为2个数据集,通过2个数据集的∑方差最小,得到最优的特征和val
这货是随机森林和gbdt的基础,面试高频考点.木有办法
基于pandas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class My_cart:
    def __init__(self):
        pass

    @staticmethod
    def split_Set(data_set: pd.DataFrame, col_name, col_val):
        """
        根据属性的某个值,对数据集进行2分
        :param col_val:  属性的名字
        :param col_name:  属性的特定数值
        :param data_set: DF数据集
        :return:
        """
        bool_index = data_set[col_name] <= col_val
        set1 = data_set[bool_index]
        set2 = data_set[~bool_index]
        return set1, set2

    @staticmethod
    def get_deviation(data_set: pd.DataFrame):
        # 获取label列的离差和 -> 有偏方差*样本数
        return data_set.iloc[:, -1].var(ddof=0) * data_set.shape[0]

    @staticmethod
    def choose_best_feature(data_set: pd.DataFrame, threshold_size=1, threshold_delta=1):
        """
        这个切分是用于回归
        切分的依据: 切分后label,离差和越小(方差*样本数据),表示越紧凑,分的越好
        :param threshold_delta:
        :param threshold_size:
        :param data_set:
        :return:
        """
        # 所有label值相同,不同再分
        result = data.iloc[:, -1].value_counts()
        if result.size == 1:
            return None, result.index[0]

        # 遍历所有列,找到找到的列
        row_size, col_size = data_set.shape
        init_error, best_error = My_cart.get_deviation(data_set), np.inf
        best_feature, best_value = None, None

        for name in data_set.columns[:-1]:
            # 遍历所有列
            for value in data_set[name].unique():
                s1, s2 = My_cart.split_Set(data_set, name, value)
                if len(s1) < threshold_size or len(s2) < threshold_size:
                    continue
                temp_error = My_cart.get_deviation(s1) + My_cart.get_deviation(s2)
                if temp_error < best_error:
                    best_feature = name
                    best_value = value
                    best_error = temp_error

        delta = init_error - best_error
        if delta < threshold_delta:
            #  下降的误差太小, 不再划分数据,将平均值作为划分值
            return None, data_set.iloc[:, -1].mean()

        return best_feature, best_value

    def create_tree(self, train_X):
        name, val = My_cart.choose_best_feature(train_X)

        if not name:
            return val

        root = {'name': name, 'val': val}
        # 划分数据集
        s1, s2 = My_cart.split_Set(train_X, name, val)
        root['left'] = self.create_tree(s1)
        root['right'] = self.create_tree(s2)
        return root


if __name__ == '__main__':
    data = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    data = pd.DataFrame(data, columns=['属性1', '属性2', 'label'])
    cart = My_cart()
    name, val = cart.choose_best_feature(data)
    print(name)
    print(val)
