import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import *
import cv2 as cv

# choice = 0
lr = 0.093

# choice = 1：
lr = 0.015

def isRotationMatrix(R):
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], np.float))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one


def Angel2R(Phi, Ome, Kap):

    sy = math.sin(Phi)
    cy = math.cos(Phi)

    sw = math.sin(Ome)
    cw = math.cos(Ome)

    sk = math.sin(Kap)
    ck = math.cos(Kap)

    R = np.array([
        [cy*ck - sy*sw*sk, -cy*sk - sy*sw*ck, -sy*cw],
        [cw*sk, cw*ck, -sw],
        [sy*ck + cy*sw*sk, -sy*sk + cy*sw*ck, cy*cw]
    ])

    # R = np.array([
    #     [-cy*ck + sy*sw*sk, -cy*sk - sy*sw*ck, -sy*cw],
    #     [-cw*sk, cw*ck, -sw],
    #     [-sy*ck - cy*sw*sk, -sy*sk + cy*sw*ck, cy*cw]
    # ])

    return R


def GetErrorsFunctionParam(inParam, outParam, realP, pictP):
    # 提取坐标
    X = realP[0]
    Y = realP[1]
    Z = realP[2]

    x = pictP[0]
    y = pictP[1]

    # 提取参数
    x0 = inParam['x0']
    y0 = inParam['y0']
    f = inParam['f']

    Xs = outParam['Xs']
    Ys = outParam['Ys']
    Zs = outParam['Zs']

    phi = outParam['Phi']
    ome = outParam['Ome']
    kap = outParam['Kap']

    # 计算旋转矩阵
    R = Angel2R(phi, ome, kap)

    a1 = R[0][0]
    a2 = R[0][1]
    a3 = R[0][2]
    b1 = R[1][0]
    b2 = R[1][1]
    b3 = R[1][2]
    c1 = R[2][0]
    c2 = R[2][1]
    c3 = R[2][2]

    # 引入符号
    # Xbar = a1(X - Xs) + b1(Y - Ys) + c1(Z - Zs)
    # Ybar = a2(X - Xs) + b2(Y - Ys) + c2(Z - Zs)
    Zbar = a3 * (X - Xs) + b3 * (Y - Ys) + c3 * (Z - Zs)

    # 计算误差方程参数
    a11 = (1 / Zbar) * (a1 * f + a3 * (x - x0))
    a12 = (1 / Zbar) * (b1 * f + b3 * (x - x0))
    a13 = (1 / Zbar) * (c1 * f + c3 * (x - x0))

    a21 = (1 / Zbar) * (a2 * f + a3 * (y - y0))
    a22 = (1 / Zbar) * (b2 * f + b3 * (y - y0))
    a23 = (1 / Zbar) * (c2 * f + c3 * (y - y0))

    a14 = (y - y0) * math.sin(ome) - (
            ((x - x0) / f) * ((x - x0) * math.cos(kap) - (y - y0) * math.sin(kap)) + f * math.cos(kap)) * math.cos(
        ome)
    a24 = - (x - x0) * math.sin(ome) - (
            ((y - y0) / f) * ((x - x0) * math.cos(kap) - (y - y0) * math.sin(kap)) - f * math.sin(kap)) * math.cos(
        ome)

    a15 = -f * math.sin(kap) - ((x - x0) / f) * ((x - x0) * math.sin(kap) + (y - y0) * math.cos(kap))
    a25 = -f * math.cos(kap) - ((y - y0) / f) * ((x - x0) * math.sin(kap) + (y - y0) * math.cos(kap))

    a16 = y - y0
    a26 = -(x - x0)

    A = [[a11, a12, a13, a14, a15, a16], [a21, a22, a23, a24, a25, a26]]

    return np.matrix(A)

def ProjectOnePoint(inParam, outParam, realP):
    # 提取坐标
    X = realP[0]
    Y = realP[1]
    Z = realP[2]

    # 提取参数
    x0 = inParam['x0']
    y0 = inParam['y0']
    f = inParam['f']

    Xs = outParam['Xs']
    Ys = outParam['Ys']
    Zs = outParam['Zs']

    phi = outParam['Phi']
    ome = outParam['Ome']
    kap = outParam['Kap']


    # 计算旋转矩阵
    R = Angel2R(phi, ome, kap)
    # print(isRotationMatrix(R))
    # exit()

    a1 = R[0][0]
    a2 = R[0][1]
    a3 = R[0][2]
    b1 = R[1][0]
    b2 = R[1][1]
    b3 = R[1][2]
    c1 = R[2][0]
    c2 = R[2][1]
    c3 = R[2][2]

    # 引入符号
    Xbar = a1 * (X - Xs) + b1 * (Y - Ys) + c1 * (Z - Zs)
    Ybar = a2 * (X - Xs) + b2 * (Y - Ys) + c2 * (Z - Zs)
    Zbar = a3 * (X - Xs) + b3 * (Y - Ys) + c3 * (Z - Zs)

    # x = x0 + f * Xbar / Zbar
    # y = y0 + f * Ybar / Zbar
    x = x0 - f * Xbar / Zbar
    y = y0 - f * Ybar / Zbar

    # print()
    # print('   世界坐标：', X, Y, Z)
    # print('  投影坐标系：', x, y)
    # print()

    outP = [x, y]
    # outP = [int(x), int(y)]

    return outP

def ProjectAll(real_xyz, picture_xy, inParam, outParam):
    # 计算投影后坐标
    cal_xy = []
    for index, p in enumerate(real_xyz):
        refP = picture_xy[index]
        if refP[0] == 0 and refP[1] == 0:
            cal_xy.append([0, 0])
            continue

        cal_xy.append(ProjectOnePoint(inParam, outParam, p))

    return cal_xy

def CalParamFunction(A, L):
    H = np.transpose(A) * A
    B = np.linalg.inv(H)
    x = B * np.transpose(A) * L

    dets = x

    dets_ = []
    for d in dets:
        dets_.append(np.array(d)[0][0])


    return dets_

def BatchError(input, output):
    # 如果批量处理
    errorList = []

    # 遍历得到误差值
    for index, refP in enumerate(input):
        # 跳过不存在的点
        if refP[0] == 0 and refP[1] == 0:
            continue

        errorOne = Error(refP, output[index])

        # if errorOne[0] * errorOne[1] != 0:
        #     print(refP)
        #     print(output[index])
        #     print(errorOne)
        #     print()

        errorList.append((abs(errorOne[0]) + abs(errorOne[1])) / 2)

    error = np.average(errorList)

    # print('Cheack error:', error)
    # exit()

    return error

def Error(refP, outP):
    error = [refP[0] - outP[0], refP[1] - outP[1]]

    # print(refP)
    # print(outP)
    # print('error:', error)

    return error

def UpdateOutParam(outParam, update_dets, lr = lr):
    # 改正数乘以学习率lr
    update_dets_ = np.dot(lr, update_dets)

    Xs = outParam['Xs']
    Ys = outParam['Ys']
    Zs = outParam['Zs']

    Phi = outParam['Phi']
    Omega = outParam['Ome']
    Kappa = outParam['Kap']

    outParam_ = {'Xs': Xs + update_dets_[0], 'Ys': Ys + update_dets_[1], 'Zs': Zs + update_dets_[2], 'Phi': Phi + update_dets_[3], 'Ome': Omega + update_dets_[4], 'Kap': Kappa + update_dets_[5]}
    return outParam_

def IntTool(data_set):
    new_set = []
    for p in data_set:
        new_set.append([int(p[0]), int(p[1])])

    return new_set