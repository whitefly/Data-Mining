import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# YearBuilt ,SalePrice
train_data = pd.read_csv('/Users/zhouang/Desktop/all/train.csv')

plt.figure(figsize=(12, 5))
plt.scatter(train_data['YearBuilt'], train_data['SalePrice'])
plt.show()

# GrLivArea
plt.figure(figsize=(12, 5))
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], alpha=0.6, c='b')
plt.show()

# 删除异常点
train_data = train_data[~((train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000))]

# 融合test数据, 一起处理缺失值
test_data = pd.read_csv('/Users/zhouang/Desktop/all/test.csv')
full = pd.concat([train_data, test_data], axis=0, ignore_index=True)  # type:pd.DataFrame

# 删除毫无用处的id列,最后形状(2917, 80)
full.drop(['Id'], axis=1, inplace=True)

# 查看各列缺失值->有些列的缺失有2000,超过了90%,感觉不用管
full.isnull().sum(axis=0).sort_values(ascending=False).head()

# 从LotFrontage的500个缺失值开始,查看宽度和地块面积的关系
cols = ['LotFrontage', 'LotArea']
relat = full.loc[~full[cols].isnull().any(axis=1), cols]
pd.scatter_matrix(relat)
plt.show()
del cols, relat

# LotArea无缺失值,大致线性(有些异常值,取中位数or众数),去分组后的中位数填充好了
full['地块面积范围'] = pd.qcut(full['LotArea'], q=10)
full.groupby('地块面积范围')['LotFrontage'].agg(['mean', 'median', 'count'])

# 根据LotArea,Neighborhood的分组来填充LotFrontage,但是还是有缺失值,所以只用地块面积范围
full['LotFrontage'] = full.groupby(['地块面积范围'])['LotFrontage'].transform(lambda s: s.fillna(s.median()))
full['LotFrontage'].isnull().sum()

# 找出离散类型 且 有缺失值 列
nan_clo = full.dtypes[(full.dtypes == 'object') & (full.isnull().sum(axis=0) > 0)]

# 对作者暴力填充缺失值的操作不太理解,先放在这里,做特征工程再说
cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    full[col].fillna(0, inplace=True)

cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
         "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1",
         "MasVnrType"]
for col in cols1:
    full[col].fillna("None", inplace=True)

cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType",
         "Exterior1st", "Exterior2nd"]
for col in cols2:
    full[col].fillna(full[col].mode()[0], inplace=True)


