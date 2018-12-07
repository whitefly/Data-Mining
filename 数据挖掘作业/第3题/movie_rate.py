import numpy as np
import pandas as pd

# 分离训练集和测试集,
# 结论:训练集80000*3,(涉及到8372部电影) 测试集20000*3,(涉及到4869部电影),
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.svm import SVR, LinearSVR
from surprise import KNNWithMeans, Reader, Dataset
from sklearn.model_selection import cross_val_score, GridSearchCV

# 读取数据,分离train和test集
movie = pd.read_csv("/Users/zhouang/Desktop/数据挖掘作业/question3_recommend.csv")
test_index = (movie['rating'] == -1)
test_data = movie[test_index]
train_data = movie[~test_index]
train_data.to_csv('/Users/zhouang/Desktop/数据挖掘作业/作业3/train.csv', index=False)
test_data.to_csv('/Users/zhouang/Desktop/数据挖掘作业/作业3/test.csv', index=False)

# 发现测试集中的某些电影在训练集中根本不存在,有694部. 意味着只能靠瞎猜
m1 = set(train_data["movieId"].unique())
m2 = set(test_data["movieId"].unique())
gap = m2 - m1
print(len(gap))


#  计算出电影特征表:平均分,最低,最高,众数,评论人数
#  结论:训练集涉及到8372部电影,但是4040部只有≤2人来评分, 估计统计偏差相当严重
def get_mode(s):
    return s.value_counts(ascending=False).index[0]


movie_grouped = train_data.groupby(by='movieId')
movie_feature = movie_grouped['rating'].agg(
    {'评论人数': 'count', "平均分": 'mean', "众数": get_mode, '中位数': 'median', '最高分': 'max', '最低分': "min"})
movie_feature.to_csv('/Users/zhouang/Desktop/数据挖掘作业/作业3/电影特征.csv')  # 存一下


# 制作用户特征表:  用户乐观度(大于中位数的数量和频率,小于中位数的数量和频率), 疑似水军指标,黑子指标
def positive_rate(DF):
    # 评分总是高于中位数的概率
    return pd.Series([np.sum((DF['rating'] - DF['中位数']) >= 0) / DF.shape[0], DF.shape[0]], index=['容忍度', '评论个数'])


# 分组统计,构造特征
user = pd.merge(train_data, movie_feature, left_on='movieId', right_index=True)
user_grouped = user.groupby(by='userId')
user_feature = user_grouped.apply(positive_rate)

high = 3.577777777777778
low = 2.3111111111111113


def high_score(DF):
    # 喜欢打高分的概率
    row = np.sum((DF['rating'] >= high)) / DF.shape[0]
    return pd.Series(row, index=['高分率'])


user_feature['高分率'] = user_grouped.apply(high_score)


def get_black(DF):
    # 对高分电影打低分的概率, 4分以上为高分电影
    good_movie = (DF['平均分'] >= high)
    good_size = np.sum(good_movie)
    return pd.Series([np.sum(DF[good_movie]['rating'] < low) / good_size, good_size], index=['黑评率', '看过高分电影个数'])


user_feature = pd.concat([user_feature, user_grouped.apply(get_black)], axis=1)


def get_faker(DF):
    # 对大量低分电影打高分
    bad_movie = (DF['平均分'] <= low)
    bad_size = np.sum(bad_movie)
    return pd.Series([np.sum(DF[bad_movie]['rating'] > high) / bad_size, bad_size], index=['水军率', '看过低分电影个数'])


user_feature = pd.concat([user_feature, user_grouped.apply(get_faker)], axis=1)

# 把所有特征都汇总到total中
total = pd.merge(user, user_feature, left_on='userId', right_index=True)


# 选定rmse作为指标
def rmse_cv(model, X, y):
    # rmse作为评分
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse


# 增加协同过滤作为特征
fuker = KNNWithMeans()
r = Reader(rating_scale=(0.5, 5))
data_set = Dataset.load_from_df(train_data, r)
knn_val = []
for index, row in train_data.iterrows():
    user_id, movie_id, rate = row['userId'], row['movieId'], row['rating']
    predict_rate = fuker.predict(user_id, movie_id, rate).est
    knn_val.append(predict_rate)
s = pd.Series(knn_val, index=train_data.index)
total['协同值'] = s

# 有些用户没看过好电影 or  垃圾电影. 所以将NaN填充为0
total.fillna({'水军率': 0, '黑评率': 0}, inplace=True)
X = total[['评论人数', '平均分', '众数', '中位数', '最高分', '最低分', '容忍度', '评论个数', '高分率', '黑评率', '看过高分电影个数', '水军率', '看过低分电影个数', '协同值']]
Y = total['rating']

# 不加协同列,使用LASSO,0.80-0.85之间浮动 , 只用平均数大约0.85-0.9之间, 加了协同列在0.63-0.67之间浮动
# 用其他分类器试试
# 总结:Extra,SGD,RF 训练相当耗时.所以不采用.只采用采用简单的线性回归好了
models = [LinearRegression(),
          Ridge(),
          Lasso(alpha=0.01, max_iter=1000),
          GradientBoostingRegressor(),
          ElasticNet(alpha=0.001, max_iter=1000),
          BayesianRidge()]

names = ["LR", "Ridge", "Lasso", "GBR", "Ela", "Bay"]
for name, model in zip(names, models):
    score = rmse_cv(model, X, Y)
    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))


# 简单超参数搜索
class grid:
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X, y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])


# LASSO alpha=0.0001
grid(Lasso()).grid_get(X, Y, {'alpha': [0.0001, 0.0004, 0.0006, 0.0009], 'max_iter': [1000]})
# Ridge alpha=0.5
grid(Ridge()).grid_get(X, Y, {'alpha': [0.5, 1, 3, 4, 8, 10]})
# 'alpha': 0.0008, 'l1_ratio': 0.3
grid(ElasticNet()).grid_get(X, Y, {'alpha': [0.0008, 0.004, 0.005], 'l1_ratio': [0.08, 0.1, 0.3], 'max_iter': [1000]})


# 模型融合
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mod, weight):
        self.mod = mod
        self.weight = weight

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mod]
        for my_model in self.models_:
            my_model.fit(X, y)
        return self

    def predict(self, X):
        w = []
        pred = np.array([my_model.predict(X) for my_model in self.models_])
        for data in range(pred.shape[1]):
            single = [pred[model, data] * weight for model, weight in zip(range(pred.shape[0]), self.weight)]
            w.append(np.sum(single))
        return w


ols = LinearRegression()
lasso = Lasso(alpha=0.0001)
ridge = Ridge(alpha=0.5)
ela = ElasticNet(alpha=0.0008, l1_ratio=0.3, max_iter=1000)
gbr = GradientBoostingRegressor()
bay = BayesianRidge()

w_ols = 0.02
w_lasso = 0.03
w_ridge = 0.1
w_ela = 0.2
w_gbr = 0.45
w_bay = 0.2

weight_avg = AverageWeight(mod=[ols, lasso, ridge, ela, gbr, bay],
                           weight=[w_ols, w_lasso, w_ridge, w_ela, w_gbr, w_bay])
score = rmse_cv(weight_avg, X, Y)
# rmse=0.8148411905749648,凑合用吧   优化高分为0.8127, 增加协同列后为0.608
print(score.mean())

# 对test数据进行merge,构成完整的特征
total_test = pd.merge(test_data, movie_feature, left_on='movieId', right_index=True, how='left')
total_test = pd.merge(total_test, user_feature, left_on='userId', right_index=True, how='left')

# 增加协同列
knn_val = []
for index, row in total_test.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    rate = row['rating']
    predict = fuker.predict(user_id, movie_id, rate)
    if predict.details['was_impossible']:
        knn_val.append(None)
    else:
        knn_val.append(predict.est)
s1 = pd.Series(knn_val, index=total_test.index)
total_test['协同值'] = s1

# 填充缺失值,由于有些电影缺失,所以直接采用 3作为平均值,调整了一下最低分
na_value = 3
fill_value = {'评论人数': 1, "平均分": na_value, '中位数': na_value, '众数': na_value, '最高分': na_value, '最低分': 1.5, '黑评率': 0.0,
              '水军率': 0.0, '协同值': na_value}
total_test.fillna(fill_value, inplace=True)

# 训练,预测,存储
select_col = ['评论人数', '平均分', '众数', '中位数', '最高分', '最低分', '容忍度', '评论个数', '高分率', '黑评率', '看过高分电影个数', '水军率', '看过低分电影个数',
              '协同值']
weight_avg.fit(X, Y)
result = weight_avg.predict(total_test[select_col])
total_test['最后结果'] = result

fuker = total_test[['userId', 'movieId', '最后结果']]


# 替换 <0.5 和 大于5的值
def sub(x):
    if x >= 5:
        return 5.0
    elif x <= 0.5:
        return 0.5
    return x


fuker['最后结果'] = fuker['最后结果'].map(sub)
fuker.rename(columns={'最后结果': 'rating'}, inplace=True)
# 保存
fuker.to_csv("/Users/zhouang/Desktop/数据挖掘作业/作业3/预测结果.csv", index=False)

# 后期改进,
# 1.哪些高分电影,哪些是低分电影,可以搞一个搜索来确定预测最优值的设定这2个值 (结论:高分搜索[3.5-4.2],低分搜索[2,2.7之间],最优结果 high:3.578 low:2.311)
# 2.测试集中的独有电影,均分随便认定了一个3分, 这个暂时没想好怎么优化
# 3.可以引入关联,看哪些用户口味相似. 加入 相似用户所打的平均分作为一个feature. (结论:增加协同列,最后交叉验证为0.61,可惜提交时间过了,没有测试,哎😔)
