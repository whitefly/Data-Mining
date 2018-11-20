import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('数据挖掘作业/销量预测/Advertising.csv', index_col='id')  # type:pd.DataFrame

# 画图看趋势
cols = data.columns[:-1]
label = data.columns[-1]
sns.pairplot(data, x_vars=cols, y_vars=label, kind='reg', aspect=0.8, height=7)
plt.show()

# 分解X和Y数据 ,data为完整的数据
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
# 数据集分解
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)

# 一个简单的线性回归
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(train_X, train_Y)

# 查看回归结果
print(dict(zip(train_X.columns, reg.coef_)))

# 评价预测效果
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predict_Y = reg.predict(train_X)
r2_score(train_Y, predict_Y)

reg.score(train_X, train_Y)
