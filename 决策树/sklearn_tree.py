"""
直接调sklearn中的决策树分类器
"""
from sklearn import preprocessing
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd_data = pd.read_csv('../决策树数据集/西瓜书2.0.txt', delimiter='\t')

np_data = pd_data.values

y = np.copy(np_data[:, -1])


def encode_data(data):
    encoder = preprocessing.LabelEncoder()
    for i in range(len(data[0])):
        data[:, i] = encoder.fit_transform(data[:, i])
    return data


clf = tree.DecisionTreeClassifier(criterion='entropy')
num_data = encode_data(np_data)
x = num_data[:, :-1]
clf.fit(x, y)

# 可视化
feature_name = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
target_name = ['好瓜', '坏瓜']

from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_name,
                     class_names=target_name, filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("WineTree.pdf")
print('Visible tree plot saved as pdf.')
