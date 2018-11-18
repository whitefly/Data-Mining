"""
思想: boost的串行训练思想
分别给每个样本点设置样本权重,
1.用该样本设置训练弱分类器.
2.用该分类器的误差来更改样本点的权值.然后继续训练下一个弱分类器

特点: 在<实战>中,使用的是单点决策树 训练数据是连续的.所以不能用 ID3算法来构造决策树. 需要直接二分法来需要最好的点
基于numpy
"""
from pprint import pprint

import numpy as np


def predict(data, col_index, threshold, compare) -> np.ndarray:
    """
    列属性≥阈值,返回1,列属性＜阈值,返回-1.最后得到预测的结果
    每个数据为行向量
    :param compare: '>'表示比阈值大的为正 '<'表示阈值小的正
    :param col_index:
    :param data:
    :param threshold:
    :return: -1,1元素组成的行向量
    """
    result = np.ones(data.shape[0])
    if compare == '>':
        result[data[:, col_index] <= threshold] = -1.0
    else:
        result[data[:, col_index] > threshold] = -1.0
    return result


def create_single_node(data: np.ndarray, labels: np.ndarray, D: np.ndarray):
    """
    label为连续数据,需要遍历所有特征,划分区间来找到合适的二分值
    初始设置划分区间为10
    :param labels: 数据的标签值,为行向量
    :param data: 每个数据为行向量
    :param D: 样本的权重,为行向量
    :return:
    """
    blocks = 10
    min_error = np.inf
    best_feature = {}
    best_labels = None
    for index in range(data.shape[1]):
        min_v, max_v = data[:, index].min(), data[:, index].max()
        step = (max_v - min_v) / blocks

        for comp in ['>', "<"]:
            for i in range(-1, blocks + 1):
                candidate = i * step + min_v
                predict_labels = predict(data, index, candidate, comp)
                error = D[predict_labels != labels].sum()

                # 得到error最小值
                if error < min_error:
                    min_error = error
                    best_labels = predict_labels
                    best_feature["index"] = index
                    best_feature['value'] = candidate
                    best_feature['comp'] = comp

    return min_error, best_feature, best_labels


def adaBoost_DT(data: np.ndarray, labels: np.ndarray, size=20):
    """
    用单个节点来作为弱分类器
    弱分类个数默认为20
    :param data:
    :param labels:
    :param size:
    :return:  弱分类组成的list,包含弱分类的 特征下标,特征阈值,分类器的重要度
    """
    container = []
    simples_size, col_size = data.shape
    # 初始化权重 和 f(x)
    D = np.ones(simples_size) / simples_size
    f_x = np.zeros(simples_size)
    # 串行化训练弱分类器
    for i in range(size):
        error, best_feature, predict_labels = create_single_node(data, labels, D)

        # 根据公式更新求出alpha,更新样本权重
        alpha = (np.log((1 - error) / max(1e-16, error))) / 2
        D *= np.exp(- alpha * labels * predict_labels)
        D /= D.sum()

        # 增加弱分类器
        best_feature['alpha'] = alpha
        container.append(best_feature)

        # 更新f(x),用已有的分类器投票选举
        f_x += alpha * predict_labels
        error_size = np.sum(np.sign(f_x) != labels)
        error_rate = error_size / simples_size
        print("第{}轮\n综合投票后 误分类点个数:{}\n样本权值:{}\n".format(i + 1, error_size, list(D)))

        if error_rate == 0.0:
            break
    return container


if __name__ == '__main__':
    some = np.arange(10, dtype='float').reshape(-1, 1)  # 李航<统计学习方法数据>的adaboost的数据例子
    some_labels = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
    print()
    pprint(adaBoost_DT(some, some_labels))

    # some = [[1, 2.1], [2, 1.1], [1.3, 1], [1, 1], [2, 1]]
    # some_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    # with open('../Adaboost数据集/机器学习实战.txt', 'w') as f:
    #     for i in range(len(some)):
    #         f.write("{},{},{}\n".format(*some[i], some_labels[i]))
