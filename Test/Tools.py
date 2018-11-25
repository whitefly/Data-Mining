import matplotlib.pyplot as plt


def go_through(root, l, r, result):
    if not isinstance(root, dict):
        result.append((root, [l, r]))
        return

    v = root['val']
    go_through(root['left'], l, v, result)  # 左子
    go_through(root['right'], v, r, result)  # 右子


def func(w, x):
    return x * w[0] + w[1]


def plot(tree, kind='回归树'):
    # 用来画出回归树
    result1 = []
    go_through(tree, None, None, result1)
    limit_l, limit_r = 0, 1
    for leaf in result1:
        # 处理边界的none值
        y, (x1, x2) = leaf
        if not x1 and x2:
            x1 = limit_l
        elif x1 and not x2:
            x2 = limit_r

        # 画出叶节点的直线
        if kind == "回归树":
            # 用于回归树,Y为固定值
            plt.plot([x1, x2], [y, y], 'r', linewidth=2)
        elif kind == '模型树':
            w = y
            # 用于模型树,Y为拟合直线,其中w为二维直线系数
            plt.plot([x1, x2], [func(w, x1), func(w, x2)])
        else:
            raise ValueError("请指定 回归树 or 模型树")
    plt.show()
