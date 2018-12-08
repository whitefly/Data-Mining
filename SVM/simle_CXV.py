"""
这个代码是根据SVM的对偶问题,然后扔进cvxpy求解的过程
数据集来自<统计学习方法>,最后的成果参见 ../SVM_数据集/实战数据集_线性(支持向量).png,
结果和书上一模一样
"""
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from cvxpy import Minimize, quad_form, sum, Variable, Parameter
from mpl_toolkits.mplot3d import Axes3D


class CVX_SVM:
    def __init__(self):
        self.w = None  # 一维行向量
        self.b = None  # 标量
        self.trained = False

    def fit(self, train_X, train_Y):
        X = train_X.values if isinstance(train_X, pd.DataFrame) else train_X
        Y = train_Y.values.reshape(-1, 1) if isinstance(train_Y, pd.Series) else train_Y.reshape(-1, 1)

        row_size, col_size = X.shape
        # 二次型矩阵
        H = np.dot(X, X.T) * np.dot(Y, Y.T)
        # a1,a2...的列向量
        alphas = Variable((row_size, 1))
        # quad_form:用二次型矩阵构造损失函数
        # objective = Minimize((1.0 / 2) * quad_form(alphas, H) + sum(-1 * alphas))
        objective = Minimize((1.0 / 2) * quad_form(alphas, Parameter(shape=H.shape, value=H, PSD=True)) + sum(-1 * alphas))
        # 用矩阵表示的约束条件
        constraints = [alphas >= 0,
                       Y.T * alphas == 0]

        prob = cvx.Problem(objective, constraints)
        prob.solve()

        # 根据alphas,求出w和b,(w为列向量,b为单元素行向量)
        w = np.dot(X.T, alphas.value * Y)
        sv_index = np.nonzero(np.around(alphas.value, 10))[0]  # 所有支持向量的下标
        first = sv_index[0]
        yi, xi = Y[first, 0], X[first]

        b = yi - np.dot(xi, w)  # 一维行向量,在np.dot中,可以作为行向量来做矩阵乘法

        self.w = w.flatten()
        self.b = b[0]
        self.sv_index = sv_index
        self.X = X
        self.Y = Y

    def plot2d(self, df: pd.DataFrame):
        # 画出二维数据散点
        col_names = df.columns
        w1, w2 = self.w
        b = self.b
        fg = sns.FacetGrid(df, hue=col_names[-1])
        fg.map(plt.scatter, col_names[0], col_names[1]).add_legend()

        # 画出特性向量点
        sv_points = self.X[self.sv_index]
        plt.scatter(sv_points[:, 0], sv_points[:, 1], c='b', marker='+')

        # 画出超平面(1维,即直线)
        min_x1, max_x1 = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        min_x2, max_x2 = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        demo_x = np.linspace(min_x1, max_x1, 50)
        get_y = lambda x: -(b + w1 * x) / w2
        plt.plot(demo_x, get_y(demo_x))

        # 根据数据集来设置x,y轴的显示范围
        plt.xlim(min_x1, max_x1)
        plt.ylim(min_x2, max_x2)

        plt.show()

    def plot3d(self, df: pd.DataFrame):
        # 画出三维数据散点
        ax = plt.subplot(111, projection='3d')  # type:Axes3D
        col_names = df.columns
        w1, w2, w3 = self.w
        b = self.b
        fg = sns.FacetGrid(df, hue=col_names[-1], palette={1: 'r', -1: 'b'})
        fg.map(ax.scatter, col_names[0], col_names[1], col_names[2])

        # 画出特性向量点
        sv_points = self.X[self.sv_index]
        ax.scatter(sv_points[:, 0], sv_points[:, 1], sv_points[:, 2], c='y', s=15)

        # 根据数据集来设置x,y轴的显示范围
        min_x1, max_x1 = self.X[:, 0].min() - 10, self.X[:, 0].max() + 10
        min_x2, max_x2 = self.X[:, 1].min() - 10, self.X[:, 1].max() + 10
        min_x3, max_x3 = self.X[:, 2].min() - 10, self.X[:, 2].max() + 10
        ax.set_xlim(min_x1, max_x1)
        ax.set_ylim(min_x2, max_x2)
        ax.set_zlim(min_x3, max_x3)
        # 画出超平面(2维,即平面)
        # 只需要画出支持向量的超平面即可,否则平面的z太大
        sv_min_x1, sv_max_x1 = self.X[:, 0].min(), self.X[:, 0].max()
        sv_min_x2, sv_max_x2 = self.X[:, 1].min(), self.X[:, 1].max()
        demo_x = np.linspace(sv_min_x1, sv_max_x1, 50)
        demo_y = np.linspace(sv_min_x2, sv_max_x2, 50)
        demo_x, demo_y = np.meshgrid(demo_x, demo_y)

        get_z = lambda x, y: -(b + w1 * x + w2 * y) / w3
        demo_z = get_z(demo_x, demo_y)
        ax.plot_surface(demo_x, demo_y, demo_z)

        plt.show()


if __name__ == '__main__':
    # 二维线性可分
    # my_df = pd.read_csv('../SVM_数据集/实战数据集_线性.csv', names=['x1', 'x2', 'label'])
    # svm = CVX_SVM()
    # svm.fit(my_df[['x1', 'x2']], my_df['label'])
    # print(svm.w)
    # svm.plot2d(my_df)

    # 三维线性可分
    my_df = pd.read_csv('../SVM_数据集/非线性数据(升维可分).csv', names=['x1', 'x2', 'x3', 'label'])
    svm = CVX_SVM()
    svm.fit(my_df[['x1', 'x2', 'x3']], my_df['label'])
    print("支持向量点的id为:{}".format(svm.sv_index))
    svm.plot3d(my_df)
