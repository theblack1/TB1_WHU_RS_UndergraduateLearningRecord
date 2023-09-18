import cv2 as cv
from sympy import *

RealList = [28, 16, 4, 52]


def takePosition(elem):
    return elem[0][0]

def Binary(img, showFlag=0):
    # 图像预处理
    img1 = cv.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 图像转灰度

    # 二值化
    binImg = cv.adaptiveThreshold(
        img_1,
        255,  # 大于阈值的改为255  否则改为0
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY,  # 黑白二值化
        blockSize=11,
        C=3)

    binImg2 = cv.medianBlur(binImg, 7)  # 中值滤波去椒盐

    # cv.namedWindow("Binary", cv.WINDOW_NORMAL)  # 二值图显示
    # cv.imshow("Binary", binImg2)

    # cv.imwrite("outputBinImg.jpg", binImg2, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    if showFlag == 1:
        cv.namedWindow("edge", cv.WINDOW_NORMAL)
        cv.imshow("edge", binImg2)
        cv.waitKey(0)

    return binImg2

def distance(r, r0):
    dis = abs(r[0][0] - r0[0][0]) + abs(r[0][1] - r0[0][1])
    return dis

def distance1(r, x, y):
    dis = abs(r[0][0] - x) + abs(r[0][1] - y)
    return dis

def GetEllipse(img, showFlag=0):
    imgC = img.copy()

    # 边缘二值化
    binImg2 = Binary(img)

    # 边缘检测
    edgs, _ = cv.findContours(binImg2, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_NONE)

    SList = []
    rateList = []
    retvalList = []

    # 椭圆检测
    for edg in edgs:
        if len(edg) < 5:  # 剔除散点
            continue
        retval = cv.fitEllipse(edg)  # 取轮廓拟合椭圆
        # imgC_raw = cv.ellipse(imgC, retval, (0, 0, 255), thickness=3)  # 未经处理的结果

        S = (retval[1][0] * retval[1][1]) / 2  # 面积计算
        # SList.append(S)

        if S > 1100 and S < 6000:  # 面积筛选
            rate = retval[1][1] / retval[1][0]
            # rateList.append(rate)

            if rate < 4:  # 比例筛选
                retvalList.append(retval)
                # rateList.append(rate)
                # SList.append(S)

    # 绘制原图
    # cv.namedWindow("mark_ellipse_raw", cv.WINDOW_NORMAL)
    # cv.imshow("mark_ellipse_raw", imgC_raw)

    # 辅助分析，判断椭圆取值

    # SList.sort()
    # for s in SList:
    #     print(s)

    # rateList.sort()
    # for r in rateList:
    #     print(r)

    retvalList.sort(key=takePosition)  # 辅助去除同心圆
    # distList = []

    # cent_x = binImg2.shape[0] / 2  # 辅助剔除佛像处的椭圆
    # cent_y = binImg2.shape[1] / 2
    # centerDistance = []

    # angleList = []  # 辅助剔除倾斜椭圆

    # 绘制到图像
    num = 0
    r0 = retvalList[0]
    retvalList_real = []  # 记录准确的值
    for index, r in enumerate(retvalList):
        # 剔除同心圆
        if index != 0:
            dist2Before = distance(r, r0)
            if dist2Before < 5:
                S0 = (r0[1][0] * r0[1][1]) / 2
                S = (r[1][0] * r[1][1]) / 2
                if S < S0:
                    continue
                else:
                    retvalList_real[num-1] = r
                    continue

            # distList.append(dist2Before)
            r0 = r

        # 剔除佛像处的椭圆
        # dist2Center = distance2center(r, cent_x, cent_y)
        # if dist2Center < 400:
        #     continue
        # centerDistance.append(dist2Center)

        #  剔除倾斜椭圆

        angle = abs(r[2] - 90)
        if angle > 10:
            continue
        # angleList.append(angle)

        retvalList_real.append(r)
        num += 1
        imgC = cv.ellipse(imgC, r, (0, 0, 255), thickness=3)  # 在原图画椭圆

    # 辅助查找，中心差距&同心圆检测&倾斜度

    # distList.sort()
    # for d in distList:
    #     print(d)

    # centerDistance.sort()
    # for cd in centerDistance:
    #     print(cd)

    # angleList.sort()
    # for a in angleList:
    #     print(a)

    # 显示椭圆标注后的图像
    if showFlag == 1:
        cv.namedWindow("mark_ellipse", cv.WINDOW_NORMAL)
        cv.imshow("mark_ellipse", imgC)
        cv.waitKey(0)

    cv.namedWindow("mark_ellipse", cv.WINDOW_NORMAL)
    cv.imshow("mark_ellipse", imgC)
    cv.waitKey(0)

    return retvalList_real, imgC

def GetNumber(imgC, retvalList_real, showFlag=0, putFlag=0):
    if putFlag == 1:
        for i, r in enumerate(retvalList_real):
            text = "(" + str(i) + ")"
            cv.circle(imgC, (int(r[0][0]), int(r[0][1])), radius=1, color=(0, 0, 255), thickness=4)
    else:
        for i, r in enumerate(retvalList_real):
            text = "(" + str(i) + ")"
            cv.putText(imgC, text, (int(r[0][0]), int(r[0][1])), cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3);
            cv.circle(imgC, (int(r[0][0]), int(r[0][1])), radius=1, color=(0, 0, 255), thickness=4)

    if showFlag == 1:
        cv.namedWindow("Show_Number", cv.WINDOW_NORMAL)
        cv.imshow("Show_Number", imgC)
        cv.waitKey(0)

    return imgC

def LoadParam(xy, XY):
    m11, m12, m13, m21, m22, m23, m31, m32 = symbols("m11 m12 m13 m21 m22 m23 m31 m32")
    M = solve(
        [
            -xy[0][0] + (m11 * XY[0][0] + m12 * XY[0][1] + m13) / (m31 * XY[0][0] + m32 * XY[0][1] + 1),
            -xy[0][1] + (m21 * XY[0][0] + m22 * XY[0][1] + m23) / (m31 * XY[0][0] + m32 * XY[0][1] + 1),

            -xy[1][0] + (m11 * XY[1][0] + m12 * XY[1][1] + m13) / (m31 * XY[1][0] + m32 * XY[1][1] + 1),
            -xy[1][1] + (m21 * XY[1][0] + m22 * XY[1][1] + m23) / (m31 * XY[1][0] + m32 * XY[1][1] + 1),

            -xy[2][0] + (m11 * XY[2][0] + m12 * XY[2][1] + m13) / (m31 * XY[2][0] + m32 * XY[2][1] + 1),
            -xy[2][1] + (m21 * XY[2][0] + m22 * XY[2][1] + m23) / (m31 * XY[2][0] + m32 * XY[2][1] + 1),

            -xy[3][0] + (m11 * XY[3][0] + m12 * XY[3][1] + m13) / (m31 * XY[3][0] + m32 * XY[3][1] + 1),
            -xy[3][1] + (m21 * XY[3][0] + m22 * XY[3][1] + m23) / (m31 * XY[3][0] + m32 * XY[3][1] + 1),
        ],
        [m11, m12, m13, m21, m22, m23, m31, m32]
    )
    return M

def Projection2d(M, inputPoints):
    m11, m12, m13, m21, m22, m23, m31, m32 = symbols("m11 m12 m13 m21 m22 m23 m31 m32")
    outputPoints = []
    for p in inputPoints:
        x = (M[m11] * p[0] + M[m12] * p[1] + M[m13]) / (M[m31] * p[0] + M[m32] * p[1] + 1)
        y = (M[m21] * p[0] + M[m22] * p[1] + M[m23]) / (M[m31] * p[0] + M[m32] * p[1] + 1)
        outputPoints.append((x, y))

    return outputPoints

def DisplayResult(imgC, PointInPicture, retvalList_real=[], showFlag = 0, color = (255, 0, 0)):
    picture_xy = []

    if len(retvalList_real) != 0:
        for index, p in enumerate(PointInPicture):
            found = 0
            for r in retvalList_real:
                dis = distance1(r, p[0], p[1])
                if dis < 10:
                    picture_xy.append([int(r[0][0]), int(r[0][1])])
                    # print(dis)
                    text = "(" + str(index) + ")"
                    cv.putText(imgC, text, (picture_xy[index][0], picture_xy[index][1]), cv.FONT_HERSHEY_SIMPLEX, 2.5,
                               color, 3);
                    cv.circle(imgC, (picture_xy[index][0], picture_xy[index][1]), radius=1, color=color,
                              thickness=4)
                    found = 1
                    break
            if found == 0:
                picture_xy.append([0, 0])


    else:
        for i, r in enumerate(PointInPicture):
            text = "(" + str(i) + ")"
            cv.putText(imgC, text, (int(r[0]), int(r[1])), cv.FONT_HERSHEY_SIMPLEX, 2.5, color, 3);
            cv.circle(imgC, (int(r[0]), int(r[1])), radius=1, color=(255, 0, 0), thickness=4)

    if showFlag == 1:
        cv.namedWindow("Result", cv.WINDOW_NORMAL)
        cv.imshow("Result", imgC)
        cv.waitKey(0)

    return imgC, picture_xy


def GetMark(imgC, retvalList_real, RealPoint, Auto=3, showFlag = 0):
    # 选取参考点
    if Auto == 3:
        GetNumber(imgC, retvalList_real, 1)
        RefList = list(map(int, input('Four refer point:').split(" ")))
    elif Auto == 0:
        RefList = [43, 39, 10, 1]
    elif Auto == 1:
        RefList = [45, 26, 2, 6]

    # 射影变换
    RefPoints = []
    RealPoint1 = []
    for i in range(4):
        num = RefList[i]
        RefPoints.append(retvalList_real[num][0])

        num1 = RealList[i]
        tmp = RealPoint[num1]
        RealPoint1.append((tmp[1], tmp[2]))

    M = LoadParam(RefPoints, RealPoint1)

    RealPoint2 = []
    for r in RealPoint:
        RealPoint2.append((r[1], r[2]))

    PointInPicture = Projection2d(M, RealPoint2)

    # 绘制图片
    imgResult, picture_xy = DisplayResult(imgC, PointInPicture, retvalList_real=retvalList_real, showFlag=showFlag)

    return imgResult, picture_xy



