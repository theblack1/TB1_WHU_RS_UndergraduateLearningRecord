import cv2 as cv
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

Per = 2.49
Real_points = [[0, 0], [10 * Per, 0], [0, 7 * Per], [10 * Per, 7 * Per]]
Real_points_3d = [
    [0, 0, 0], [10 * Per, 0, 0], [0, 7 * Per, 0], [10 * Per, 7 * Per, 0],
    [0, 0, Per], [10 * Per, 0, Per]
]

Picture_points = []

Box_points = [
    [0, 0], [Per, 0], [0, Per], [Per, Per],
    [-0.5 * Per, -0.5 * Per], [0.5 * Per, -0.5 * Per], [-0.5 * Per, 0.5 * Per], [0.5 * Per, 0.5 * Per]
]
Box_points_3d = [
    [0, 0, 0], [2 * Per, 0, 0], [0, 2 * Per, 0], [2 * Per, 2 * Per, 0],
    [0, 0, 2 * Per], [2 * Per, 0, 2 * Per], [0, 2 * Per, 2 * Per], [2 * Per, 2 * Per, 2 * Per],
    [3 * Per, 0, 0], [0, 3 * Per, 0], [0, 0, 3 * Per]
]

relation = [[1, 9], [1, 10], [1, 11], [2, 4], [3, 4], [5, 6], [5, 7], [6, 8], [7, 8], [3, 7], [4, 8], [2, 6]]

def Binary(img, showFlag=0):
    # 图像预处理
    img1 = cv.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 图像转灰度

    # 二值化
    ret, binImg = cv.threshold(img_1, 127, 255, cv.THRESH_BINARY)

    binImg2 = cv.medianBlur(binImg, 3)  # 中值滤波去椒盐

    # cv.namedWindow("Binary", cv.WINDOW_NORMAL)  # 二值图显示
    # cv.imshow("Binary", binImg2)

    # cv.imwrite("outputBinImg.jpg", binImg2, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    if showFlag == 1:
        cv.namedWindow("edge", cv.WINDOW_NORMAL)
        cv.imshow("edge", binImg2)
        cv.waitKey(0)

    return binImg2

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


def Projection(M, inputPoints):
    m11, m12, m13, m21, m22, m23, m31, m32 = symbols("m11 m12 m13 m21 m22 m23 m31 m32")
    outputPoints = []
    for p in inputPoints:
        x = (M[m11] * p[0] + M[m12] * p[1] + M[m13]) / (M[m31] * p[0] + M[m32] * p[1] + 1)
        y = (M[m21] * p[0] + M[m22] * p[1] + M[m23]) / (M[m31] * p[0] + M[m32] * p[1] + 1)
        outputPoints.append((x, y))

    return outputPoints


def LoadParam3d(xy, XYZ):
    m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34 = symbols(
        "m11 m12 m13 m14 m21 m22 m23 m24 m31 m32 m33 m34")

    M = solve(
        [
            m34 - 1,

            -xy[0][0] + m11 * XYZ[0][0] + m12 * XYZ[0][1] + m13 * XYZ[0][2] + m14
            - xy[0][0] * XYZ[0][0] * m31 - xy[0][0] * XYZ[0][1] * m32 - xy[0][0] * XYZ[0][2] * m33,
            -xy[0][1] + m21 * XYZ[0][0] + m22 * XYZ[0][1] + m23 * XYZ[0][2] + m24
            - xy[0][1] * XYZ[0][0] * m31 - xy[0][1] * XYZ[0][1] * m32 - xy[0][1] * XYZ[0][2] * m33,

            -xy[1][0] + m11 * XYZ[1][0] + m12 * XYZ[1][1] + m13 * XYZ[1][2] + m14
            - xy[1][0] * XYZ[1][0] * m31 - xy[1][0] * XYZ[1][1] * m32 - xy[1][0] * XYZ[1][2] * m33,
            -xy[1][1] + m21 * XYZ[1][0] + m22 * XYZ[1][1] + m23 * XYZ[1][2] + m24
            - xy[1][1] * XYZ[1][0] * m31 - xy[1][1] * XYZ[1][1] * m32 - xy[1][1] * XYZ[1][2] * m33,

            -xy[2][0] + m11 * XYZ[2][0] + m12 * XYZ[2][1] + m13 * XYZ[2][2] + m14
            - xy[2][0] * XYZ[2][0] * m31 - xy[2][0] * XYZ[2][1] * m32 - xy[2][0] * XYZ[2][2] * m33,
            -xy[2][1] + m21 * XYZ[2][0] + m22 * XYZ[2][1] + m23 * XYZ[2][2] + m24
            - xy[2][1] * XYZ[2][0] * m31 - xy[2][1] * XYZ[2][1] * m32 - xy[2][1] * XYZ[2][2] * m33,

            -xy[3][0] + m11 * XYZ[3][0] + m12 * XYZ[3][1] + m13 * XYZ[3][2] + m14
            - xy[3][0] * XYZ[3][0] * m31 - xy[3][0] * XYZ[3][1] * m32 - xy[3][0] * XYZ[3][2] * m33,
            -xy[3][1] + m21 * XYZ[3][0] + m22 * XYZ[3][1] + m23 * XYZ[3][2] + m24
            - xy[3][1] * XYZ[3][0] * m31 - xy[3][1] * XYZ[3][1] * m32 - xy[3][1] * XYZ[3][2] * m33,

            -xy[4][0] + m11 * XYZ[4][0] + m12 * XYZ[4][1] + m13 * XYZ[4][2] + m14
            - xy[4][0] * XYZ[4][0] * m31 - xy[4][0] * XYZ[4][1] * m32 - xy[4][0] * XYZ[4][2] * m33,
            -xy[4][1] + m21 * XYZ[4][0] + m22 * XYZ[4][1] + m23 * XYZ[4][2] + m24
            - xy[4][1] * XYZ[4][0] * m31 - xy[4][1] * XYZ[4][1] * m32 - xy[4][1] * XYZ[4][2] * m33,

            -xy[5][0] + m11 * XYZ[5][0] + m12 * XYZ[5][1] + m13 * XYZ[5][2] + m14
            - xy[5][0] * XYZ[5][0] * m31 - xy[5][0] * XYZ[5][1] * m32 - xy[5][0] * XYZ[5][2] * m33,

        ],
        [m11, m12, m13, m14,
         m21, m22, m23, m24,
         m31, m32, m33, m34]
    )
    return M


def Projection3d(M, inputPoints):
    # 准备参数
    m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34 = symbols(
        "m11 m12 m13 m14 m21 m22 m23 m24 m31 m32 m33 m34")

    outputPoints = []
    for p in inputPoints:
        x = (M[m11] * p[0] + M[m12] * p[1] + M[m13] * p[2] + M[m14]) / \
            (M[m31] * p[0] + M[m32] * p[1] + M[m33] * p[2] + M[m34])

        y = (M[m21] * p[0] + M[m22] * p[1] + M[m23] * p[2] + M[m24]) / \
            (M[m31] * p[0] + M[m32] * p[1] + M[m33] * p[2] + M[m34])
        outputPoints.append([x, y])

    return outputPoints


def mouse(event, x, y, flags, param):
    global PNumber, Picture_points
    if len(Picture_points) < 6:
        if event == cv.EVENT_LBUTTONDOWN:
            Picture_points.append([x, y])
            xy = "(%d,%d)" % (x, y)
            cv.circle(img_, (x, y), 1, (255, 255, 255), thickness=-1)
            cv.putText(img_, xy, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            cv.imshow("image", img_)
    else:
        cv.destroyAllWindows()


def GetPoint(img_):
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.imshow("image", img_)
    cv.setMouseCallback("image", mouse)

    cv.waitKey(0)
    cv.destroyAllWindows()


def ShowBox(img, M, flag):
    imgC = img
    if flag == 0:
        box = Box_points
        output_points = Projection(M, box)
    elif flag == 1:
        box = Box_points_3d
        output_points = Projection3d(M, box)

    for i, p in enumerate(output_points):
        output_points[i] = [int(p[0]), int(p[1])]

    for p in output_points:
        cv.circle(imgC, (p[0], p[1]), radius=1, color=(0, 255, 0), thickness=4)

    for index, r in enumerate(relation):
        if index == 0:
            cv.arrowedLine(imgC, output_points[r[0] - 1], output_points[r[1] - 1], (0, 0, 255), thickness=4, line_type=8,
                           shift=0, tipLength=0.1)
            cv.putText(imgC, 'x', output_points[r[1] - 1], cv.FONT_HERSHEY_SIMPLEX, 2.5, (128, 128, 128), 3)
        elif index == 1:
            cv.arrowedLine(imgC, output_points[r[0] - 1], output_points[r[1] - 1], (0, 255, 0), thickness=4, line_type=8,
                           shift=0, tipLength=0.1)
            cv.putText(imgC, 'y', output_points[r[1] - 1], cv.FONT_HERSHEY_SIMPLEX, 2.5, (128, 128, 128), 3)
        elif index == 2:
            cv.arrowedLine(imgC, output_points[r[0] - 1], output_points[r[1] - 1], (255, 0, 0), thickness=4, line_type=8,
                           shift=0, tipLength=0.1)
            cv.putText(imgC, 'z', output_points[r[1] - 1], cv.FONT_HERSHEY_SIMPLEX, 2.5, (128, 128, 128), 3)
        else:
            cv.line(imgC, output_points[r[0] - 1], output_points[r[1] - 1], (128, 128, 128), thickness=4, lineType=8)
    return imgC


if __name__ == "__main__":
    imgList = []
    for i in range(1,4):
        filename = '{}.jpg'.format(i)
        imgList.append(cv.imread(filename))

    img = imgList[1]

    # cv.namedWindow("Origin", cv.WINDOW_NORMAL)
    # cv.imshow("Origin", img)
    # cv.waitKey(0)

    bin_img = Binary(img,showFlag=0)

    img_ = img.copy()
    GetPoint(img)

    M = LoadParam3d(Picture_points, Real_points_3d)

    imgC = ShowBox(bin_img, M, 1)

    cv.namedWindow("Result", cv.WINDOW_NORMAL)
    cv.imshow("Result", imgC)
    cv.waitKey(0)
