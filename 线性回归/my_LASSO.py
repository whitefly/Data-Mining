import itertools
import numpy as np


def lasso_regression(X, y, lambd=0.2, threshold=0.1):
    ''' 通过坐标下降(coordinate descent)法获取LASSO回归系数
    '''
    # 计算残差平方和 Y-XW的平方
    rss = lambda X, y, w: (y - X * w).T * (y - X * w)
    # 初始化回归系数w
    m, n = X.shape
    w = np.matrix(np.zeros((n, 1)))
    r = rss(X, y, w)
    # 使用坐标下降法优化回归系数w ???

    niter = itertools.count(1)
    for it in niter:
        for k in range(n):
            # 计算常量值z_k和p_k, 选取某一个属性k
            z_k = (X[:, k].T * X[:, k])[0, 0]
            p_k = 0
            # todo 为啥p_k这么算,和j≠k?
            for i in range(m):
                p_k += X[i, k] * (y[i, 0] - sum([X[i, j] * w[j, 0] for j in range(n) if j != k]))

            # 计算每个属性的系数 Wk,通过公式得到解析解,但是这个解是分段函数
            if p_k < -lambd / 2:
                w_k = (p_k + lambd / 2) / z_k
            elif p_k > lambd / 2:
                w_k = (p_k - lambd / 2) / z_k
            else:
                w_k = 0
            # 该列的参数确定,开始确定next列的参数
            w[k, 0] = w_k

        # 假设全部搞一遍是一轮, 搞很多轮是什么意思,而且沿用上轮的其他参数作为固定值.
        r_prime = rss(X, y, w)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime

        print('Iteration: {}, delta = {}'.format(it, delta))
        # 减少的rss小于阈值,就停止迭代
        if delta < threshold:
            break

    return w


def stagewise_regression(X, y, eps=0.01, niter=100):
    ''' 通过向前逐步回归获取回归系数
    '''
    m, n = X.shape
    w = np.matrix(np.zeros((n, 1)))
    min_error = float('inf')
    all_ws = np.matrix(np.zeros((niter, n)))
    # 计算残差平方和
    rss = lambda X, y, w: (y - X * w).T * (y - X * w)
    for i in range(niter):
        print('{}: w = {}'.format(i, w.T[0, :]))
        for j in range(n):
            # 计算每一列的属性
            for sign in [-1, 1]:
                w_test = w.copy()
                # 该列的参数,试探性的尝试改变一点(eps)
                w_test[j, 0] += eps * sign
                test_error = rss(X, y, w_test)
                # 计算一下损失函数.若比最小还小,说明方向对了. 对w进行正式修改
                if test_error < min_error:
                    min_error = test_error
                    w = w_test

        # 对每次记录一下w的变换,即每一轮都是选择的方向都是最优的(沿着最优的方向改变eps一点),然后在此基础上,
        all_ws[i, :] = w.T
    return all_ws
