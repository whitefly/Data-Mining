import matplotlib.pyplot as plt


def go_through(root, l, r, result):
    if not isinstance(root, dict):
        result.append((root, [l, r]))
        return

    v = root['val']
    go_through(root['left'], l, v, result)  # 左子
    go_through(root['right'], v, r, result)  # 右子


def main(tree):
    result1 = []
    go_through(tree, None, None, result1)
    limit_l, limit_r = 0, 1
    for leaf in result1:
        y, (x1, x2) = leaf
        if not x1 and x2:
            x1 = limit_l
        elif x1 and not x2:
            x2 = limit_r
        plt.plot([x1, x2], [y, y], 'r', linewidth=2)
    plt.show()
