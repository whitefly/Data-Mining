"""
决策树思想: 选出最优属性,每次划分.直到要划分的数据集都为同一label 或者 所有属性已都使用完
算法实现: <机器学习实战>采用list来作为数据集,但是感觉完全可以用pandas来实现,所以尝试用pandas实现
"""

import pandas as pd
import numpy as np


def get_entropy(some):
    """
    :param some:一个DF数据,含有一个label属性,表示最后的分类
    :return: np.float
    """
    label = some['label']  # type:pd.Series
    pro = label.value_counts(normalize=True)  # type:pd.Series
    # 计算熵
    return (-np.log2(pro) * pro).sum()


def _get_entropy_by_col(data, col_name):
    """
    根据属性名来划分,
    :param data:
    :param col_name:
    :return:
    """
    # 找出同列中不同值的数据,分别计算熵
    weights = data[col_name].value_counts(normalize=True)
    result = 0
    for value, weight in weights.iteritems():  # 行号是不同的值,元素值为数据集权重
        # 根据值来划分数据
        temp_set = data[data[col_name] == value]
        temp_ent = get_entropy(temp_set)
        result += temp_ent * weight
    return result


def choose_best_feature(data):
    """
    根据信息增益,选出当前数据集中最优的特征
    :param data:
    :return:
    """
    col_name, max_entgain = '', -1
    base_ent = get_entropy(data)

    # 扫遍除label外的所有,计算最大信息增溢差值
    for name in data.columns[:-1]:
        entgain = base_ent - _get_entropy_by_col(data, name)
        print(name, entgain)
        if entgain > max_entgain:
            col_name = name
            max_entgain = entgain
    return col_name, max_entgain


def majorityCnt(data):
    """

    :param data:
    :return:
    """


if __name__ == '__main__':
    # data = pd.read_csv('book.txt')
    data = pd.read_csv('dataSet1.txt', delimiter='\t')
    print(choose_best_feature(data))
