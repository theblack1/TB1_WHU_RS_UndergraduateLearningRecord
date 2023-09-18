import cv2 as cv
import numpy as np


def LinearTrans(src, new_scale=[100,200]):
    # 分离为三个灰度图
    (B, G, R) = cv.split(src)

    min = 255
    max = 0
    # 确定原图最大最小灰度
    for channel in [B, G, R]:
        if (min > np.min(channel)):
            min = np.min(channel)

        if (max < np.max(channel)):
            max = np.max(channel)

    # 检验防止出错
    if (min > max):
        print('ERROR!(LinearTrans"min>max")')
        exit()

    # print(min)
    # print(max)
    # 确定参数
    c = new_scale[0]
    d = new_scale[1]
    a = min
    b = max

    # 准备容器
    shape_ = B.shape
    B1 = np.zeros(shape_)
    G1 = np.zeros(shape_)
    R1 = np.zeros(shape_)

    Ones = np.ones(shape_)

    B1 = c * Ones + ((d - c) / (b - a)) * (B - a * Ones)
    G1 = c * Ones + ((d - c) / (b - a)) * (G - a * Ones)
    R1 = c * Ones + ((d - c) / (b - a)) * (R - a * Ones)

    # min = 255
    # max = 0
    # for channel in [B1, G1, R1]:
    #     if (min > np.min(channel)):
    #         min = np.min(channel)
    #
    #     if (max < np.max(channel)):
    #         max = np.max(channel)
    # print(min)
    # print(max)

    # 合并
    merged = cv.merge([B1, G1, R1])
    return merged/256

def garyLinearTrans(src,new_scale=[0,255]):
    min = 255
    max = 0

    # 确定原图最大最小灰度
    min = np.min(src)

    max = np.max(src)

    # 检验防止出错
    if (min > max):
        print('ERROR!(LinearTrans"min>max")')
        exit()


    # 确定参数
    c = new_scale[0]
    d = new_scale[1]
    a = min
    b = max

    # 准备容器
    shape_ = src.shape
    output = np.zeros(shape_)

    Ones = np.ones(shape_)

    output = c * Ones + ((d - c) / (b - a)) * (src - a * Ones)

    return output/256