"""
测试信息熵是否可以用来表示不纯度?
"""
from math import log
from collections import Counter


def get_inpurity(data):
    c = Counter(data)
    size = len(data)
    result = 0
    for v in c.values():
        p1 = v / size
        result += -p1 * log(p1, 2)  # 单个数据集的信息熵
    return result


def split(data, index):
    return data[:index], data[index:]


def main(nums):
    print("初始不纯度:{}\n".format(get_inpurity(nums)))
    min_inpurity = 9999
    index = -1
    for i in range(len(nums)):
        d1, d2 = split(nums, i)
        w1, w2 = len(d1) / len(nums), len(d2) / len(nums)

        temp = w1 * get_inpurity(data=d1) + w2 * get_inpurity(data=d2)
        print("index={}时,不纯度为:{}".format(i, temp))
        if temp < min_inpurity:
            index = i
            min_inpurity = temp
    print("\nindex={}时,不纯度最小:{}".format(index, min_inpurity))


if __name__ == '__main__':
    my_nums = [1, 1, 0, 0, 0]
    print(get_inpurity(my_nums))
