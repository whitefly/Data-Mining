"""
决策树思想: 选出最优属性,每次划分.直到要划分的数据集都为同一label 或者 所有属性已都使用完
算法实现: <机器学习实战>采用list来作为数据集,但是感觉完全可以用pandas来实现,所以尝试用pandas实现
"""
import pandas as pd
import numpy as np

label_name = '好瓜'


def get_entropy(some):
    """
    :param some:一个DF数据,含有一个label属性,表示最后的分类
    :return: np.float
    """
    label = some[label_name]  # type:pd.Series
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
    ent_sum = 0
    for value, weight in weights.iteritems():  # 行号是不同的值,元素值为数据集权重
        # 根据值来划分数据
        temp_set = data[data[col_name] == value]
        temp_ent = get_entropy(temp_set)
        ent_sum += temp_ent * weight
    return ent_sum


def choose_best_feature(data):
    """
    根据信息增益,选出当前数据集中最优的特征,并分离出数据集
    :param data:
    :return:
    """
    col_name, max_entgain = '', -1
    base_ent = get_entropy(data)

    # 扫遍除label外的所有,计算最大信息增溢差值
    for name in data.columns[:-1]:
        entgain = base_ent - _get_entropy_by_col(data, name)
        if entgain > max_entgain:
            col_name = name
            max_entgain = entgain
    return col_name, max_entgain


def majorityCnt(label: pd.Series):
    """
    传入最后的标签,取出现频率高的值作为返回值
    :param label:
    :param data:传入最后的标签,
    :return:
    """
    return label.value_counts().index[0]  # value_counts()默认是按降序排序,取第0个即可为最大


def create_tree(data: pd.DataFrame) -> dict:
    """
    传入数据集,递归创造树
    :param data:
    :return: 返回字典节点(本质就是字典)
    """
    # 递归终点: 数据子集已经被分开(label为同一个) or 特征已经用完
    counts = data[label_name].value_counts()  # type: pd.Series
    if counts.size == 1:  # 同一个label
        return counts.index[0]
    if data.columns.size == 1:  # feature用尽
        return majorityCnt(data[label_name])

    # 继续递归,创造新dict节点
    best_feature, ent_gain, = choose_best_feature(data)
    dict_node = {best_feature: {}}

    # todo  感觉这里重复2遍(choose_best_feature时就算了一遍),看看能不能优化下,直接把 子集和对应属性值以tuple返回
    best_feature_vals = data[best_feature].value_counts().index
    for value in best_feature_vals:  # 行号是不同的值,元素值为数据集权重
        temp_set = data[data[best_feature] == value]  # type: pd.DataFrame
        dict_node[best_feature][value] = create_tree(temp_set.drop(columns=[best_feature]))
    return dict_node


if __name__ == '__main__':
    basefold = '../决策树数据集/'
    # data = pd.read_csv(basefold+'实战1.txt') #<实战>数据1
    # data = pd.read_csv(basefold+'实战2.txt') #<实战>数据2
    data = pd.read_csv(basefold + '西瓜书2.0.txt', delimiter='\t')  # <西瓜书>数据2.0
    label_name = 'label'
    result = create_tree(data)
    from draw import createPlot

    createPlot(result)  # 字典表示的树可视化
