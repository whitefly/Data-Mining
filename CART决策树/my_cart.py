"""
ID3算法无法处理连续数据,也无法做回归.
算法实现:CART中,将所有数据都作为连续值,然后通过二分划分为2个数据集,通过2个数据集的∑方差最小,得到最优的特征和val
这货是随机森林和gbdt的基础,面试高频考点.木有办法
基于pandas
"""
from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class My_cart:
    def __init__(self, kind='回归树', threshold_size=2, threshold_delta=1):
        self.kind = kind
        self.threshold_size = threshold_size
        self.threshold_delta = threshold_delta

    def split_Set(self, data_set: pd.DataFrame, col_name, col_val):
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

    def get_reg_error(self, data_set: pd.DataFrame):
        # 回归树:非叶节点判定标准
        # 返回label列的离差和 -> 有偏方差*样本数
        return data_set.iloc[:, -1].var(ddof=0) * data_set.shape[0]

    def get_reg_leaf(self, data_set: pd.DataFrame):
        # 回归树:叶节点的值,
        # 就是一个固定的Y(Y的均值)
        return data_set.iloc[:, -1].mean()

    def __linerReg(self, data_set: pd.DataFrame):
        # 做个简单线性回归,是模型树的基础
        m, n = data_set.shape
        X = np.mat(np.hstack([data_set.iloc[:, :-1].values, np.ones((m, 1))]))
        Y = np.mat(data_set.iloc[:, -1].values).T
        fuck = X.T * X

        if np.linalg.det(fuck) == 0.0:
            raise ValueError("数据集构成的矩阵为奇异值,无法使用最小二乘法")
        W = fuck.I * X.T * Y
        return X, Y, W

    def get_model_error(self, data_set: pd.DataFrame):
        # 模型树:非叶节点判定标准
        # 返回残差
        X, Y, W = self.__linerReg(data_set)
        temp = Y - X * W
        return (temp.T * temp)[0, 0]

    def get_model_leaf(self, data_set: pd.DataFrame):
        # 模型树:用来叶节点的值,
        # 返回系数W向量
        X, Y, W = self.__linerReg(data_set)
        return np.asarray(W).flatten()

    def choose_best_feature(self, data_set: pd.DataFrame, leaf_v=get_reg_leaf, error=get_reg_error):
        """
        这个切分是用于回归
        切分的依据: 切分后label, 横线拟合:离差和越小(方差*样本数据) 斜线拟合:残差和
        :param error: 用来评定 切分点的好坏
        :param leaf_v: 返回叶节点的值.用来实现模型树和回归树的代码复用
        :param threshold_delta:
        :param threshold_size:
        :param data_set:
        :return:
        """

        # 所有label值相同,不同再分
        result = data.iloc[:, -1].value_counts()
        if result.size == 1:
            return None, result.index[0]

        init_error, best_error = error(data_set), np.inf
        best_feature, best_value = None, None

        for name in data_set.columns[:-1]:
            # 遍历所有列,找到最好的列和对应的值
            for value in data_set[name].unique():
                s1, s2 = self.split_Set(data_set, name, value)
                if len(s1) < self.threshold_size or len(s2) < self.threshold_size:
                    continue
                try:
                    temp_error = error(s1) + error(s2)
                    if temp_error < best_error:
                        best_feature = name
                        best_value = value
                        best_error = temp_error
                except ValueError:
                    pass
        delta = init_error - best_error
        if delta < self.threshold_delta:
            #  下降的误差太小, 不再划分数据,将平均值作为划分值
            return None, leaf_v(data_set)

        return best_feature, best_value

    def create_tree(self, train_X):
        if self.kind == "回归树":
            name, val = self.choose_best_feature(train_X, leaf_v=self.get_reg_leaf, error=self.get_reg_error)
        else:
            name, val = self.choose_best_feature(train_X, leaf_v=self.get_model_leaf, error=self.get_model_error)

        if not name:
            # name为none,表示不继续切分,直接生成叶节点
            return val

        root = {'name': name, 'val': val}
        # 划分数据集
        s1, s2 = self.split_Set(train_X, name, val)
        root['left'] = self.create_tree(s1)
        root['right'] = self.create_tree(s2)
        return root


if __name__ == '__main__':
    kind = '模型树'
    # data = pd.read_csv('../CART决策树_数据集/ex0.txt', names=['fuck', '属性1', 'label'], delimiter='\t')  # 直线
    # data = pd.read_csv('../CART决策树_数据集/ex00.txt', names=['属性1', 'label'], delimiter='\t')  #直线
    # data = pd.read_csv('../CART决策树_数据集集/斜线1.txt', names=['属性1', 'label'], delimiter='\t')  # 线性
    data = pd.read_csv('../CART决策树_数据集/自行车数据.txt', names=['属性1', 'label'], delimiter='\t')  # 线性

    cart = My_cart(kind)
    node = cart.create_tree(data)
    pprint(node)
    # 可视化
    from Test.Tools import plot

    # 二维散点
    plt.scatter(data['属性1'], data['label'], alpha=0.6, s=0.8)
    # 多条拟合直线
    min_x, max_x = data['属性1'].min(), data['属性1'].max()
    plot(node, cart.kind, min_x, max_x)
