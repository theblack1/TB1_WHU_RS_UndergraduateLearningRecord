import numpy as np
from matplotlib import pyplot as plt
import CalBasic


def OnePoint(inParam, outParam, outP, inP, refP):
    # 计算L&A
    A = CalBasic.GetErrorsFunctionParam(inParam, outParam, inP, refP)

    errorOne = CalBasic.Error(refP, outP)
    L = np.matrix([[errorOne[0]], [errorOne[1]]])

    dets = CalBasic.CalParamFunction(A, L)
    # print(dets)

    return dets

def OneEpoch(inParam, outParam, picture_xy, real_xyz, lr = 1e-2):
    cal_xy = CalBasic.ProjectAll(real_xyz, picture_xy, inParam, outParam)
    befor_error = CalBasic.BatchError(picture_xy, cal_xy)

    # 遍历每个点
    for index, refP in enumerate(picture_xy):
        # 去除无值点
        if refP[0] * refP[1] == 0:
            continue

        inP = real_xyz[index]
        outP = CalBasic.ProjectOnePoint(inParam, outParam, inP)
        refP = picture_xy[index]


        dets = OnePoint(inParam, outParam, outP, inP, refP)

        outParam_ = CalBasic.UpdateOutParam(outParam, dets, lr=lr)

        cal_xy = CalBasic.ProjectAll(real_xyz, picture_xy, inParam, outParam_)

        error = CalBasic.BatchError(picture_xy, cal_xy)

        # # 检测过分大的误差点
        # if error > 1.5 * befor_error:
        #     print()
        #     print('     befor:', befor_error)
        #     print('     after:', error)
        #     print('         index :', index)
        #     print('         output:', outP)
        #     print('         refer :', refP)

        # 若迭代后的误差小于原本点的误差，则保留迭代后的结果
        if error < befor_error:
            outParam = outParam_
            befor_error = error
            # print('update:', error)
            # print('dets:', dets)

    return outParam, befor_error

def CalData(inParam, outParam0, real_xyz, picture_xy, choice):
    # 选择学习率
    if choice == 0:
        lr = 0.093
    elif choice == 1:
        lr = 0.015

    # 迭代n次或者误差不再改变
    n = 100000
    outParam = outParam0
    befor_error_ = 1000

    # 绘图设置
    eList = []
    errList = []

    print('开始迭代')
    print()
    for epoch in range(n):

        outParam_, befor_error = OneEpoch(inParam, outParam, picture_xy, real_xyz,lr)

        errList.append(befor_error)
        eList.append(epoch)

        if befor_error_ == befor_error:
            print()
            print()
            print('迭代完毕')
            print('最终误差: ', befor_error)
            break

        befor_error_ = befor_error

        outParam = outParam_

        print('\r-—迭代数：{};误差值：{}--'.format(epoch + 1, befor_error), end='', flush=True)

    # 绘制过程曲线
    plt.figure()
    plt.plot(eList, errList, color='blue')
    plt.show()

    # 迭代后
    cal_xy = CalBasic.ProjectAll(real_xyz, picture_xy, inParam, outParam)

    return CalBasic.IntTool(cal_xy), outParam

def FindBestLR(inParam, outParam0, real_xyz, picture_xy):
    print('开始查找最优lr')

    # 迭代n次或者误差不再改变
    n = 10000000


    LR = 0
    error0 = 1000
    print('开始迭代')
    for lr0 in range(100):
        lr = (lr0 + 1)/1000
        print('\r-—正在查找--{}/{}'.format(lr0, 100), end='', flush=True)

        outParam = outParam0
        befor_error_ = 1000

        for epoch in range(n):
            outParam_, befor_error = OneEpoch(inParam, outParam, picture_xy, real_xyz, lr = lr)

            # 检验是否更小
            if befor_error_ == befor_error:
                if befor_error < error0:
                    LR = lr
                    error0 = befor_error
                    # print(' upDate:', lr)
                    # print(' error :', befor_error)
                    # print()
                break

            befor_error_ = befor_error

            outParam = outParam_

            if epoch == n - 1:
                print('OVER!!!!!!!!!!!!!!!!')
                exit()



    print('查找结束')
    print('Best lr   :', LR)
    print('Best error:', error0)
    exit()