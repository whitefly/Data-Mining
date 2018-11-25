"""
直接调用sklearn的pca模块
"""
from .pca import loadtxt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = loadtxt('../降维_数据集/实战数据集.txt', 'row')  # 需要单个数据为行向量

pca = PCA(n_components=1)  # 设置k
new_data = pca.fit_transform(data)  # 转化数据
axes1 = plt.subplot(111)

print("降维的方差为:", pca.explained_variance_)
print("降维后的纬度为", pca.n_components_)

axes1.scatter(new_data, np.full((1, len(new_data)), 1), alpha=0.2, s=5)
plt.show()
