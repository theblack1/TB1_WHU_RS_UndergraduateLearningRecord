import cv2
import numpy as np
import glob

Per = 2.49

Box_points_3d = [
    [0, 0, 0], [2 * Per, 0, 0], [0, 2 * Per, 0], [2 * Per, 2 * Per, 0],
    [0, 0, -2 * Per], [2 * Per, 0, -2 * Per], [0, 2 * Per, -2 * Per], [2 * Per, 2 * Per, -2 * Per],
    [3 * Per, 0, 0], [0, 3 * Per, 0], [0, 0, -3 * Per]
]

relation = [[1, 9], [1, 10], [1, 11], [2, 4], [3, 4], [5, 6], [5, 7], [6, 8], [7, 8], [3, 7], [4, 8], [2, 6]]

# def Projection(m, r, t, inputPoints):
#     # 旋转向量改旋转矩阵
#
#     R = cv2.Rodrigues((r[0][0], r[1][0], r[2][0]))
#     R = np.matrix(R[0])
#     #print(R)
#     output_points = []
#     for ip in inputPoints:
#         inp = np.transpose(np.matrix(ip))
#         op = m*(R*inp + t)
#         print(m)
#         output_points.append(op)
#
#     # 整数化
#     for i, p in enumerate(output_points):
#         output_points[i] = [int(p[0]), int(p[1])]
#
#     return output_points

def PutBox(imgList, mtx, rvecs, tvecs, dist):
    box = np.array(Box_points_3d)

    for i,img in enumerate(imgList):
        imgC = img.copy()
        m = mtx
        r = rvecs[i]
        t = tvecs[i]

        output_points_, _ = cv2.projectPoints(box, r, t, mtx, dist)
        # print()
        # print(output_points_[0][0])
        output_points = []
        for outp in output_points_:
            output_points.append([int(outp[0][0]), int(outp[0][1])])
            # print(output_points[i])

        for index, r in enumerate(relation):
            if index == 0:
                cv2.arrowedLine(imgC, output_points[r[0] - 1], output_points[r[1] - 1], (0, 0, 255), thickness=4, line_type=8,
                               shift=0, tipLength=0.1)
                cv2.putText(imgC, 'x', output_points[r[1] - 1], cv2.FONT_HERSHEY_SIMPLEX, 2.5, (128, 128, 128), 3)
            elif index == 1:
                cv2.arrowedLine(imgC, output_points[r[0] - 1], output_points[r[1] - 1], (0, 255, 0), thickness=4, line_type=8,
                               shift=0, tipLength=0.1)
                cv2.putText(imgC, 'y', output_points[r[1] - 1], cv2.FONT_HERSHEY_SIMPLEX, 2.5, (128, 128, 128), 3)
            elif index == 2:
                cv2.arrowedLine(imgC, output_points[r[0] - 1], output_points[r[1] - 1], (255, 0, 0), thickness=4, line_type=8,
                               shift=0, tipLength=0.1)
                cv2.putText(imgC, 'z', output_points[r[1] - 1], cv2.FONT_HERSHEY_SIMPLEX, 2.5, (128, 128, 128), 3)
            else:
                cv2.line(imgC, output_points[r[0] - 1], output_points[r[1] - 1], (128, 128, 128), thickness=4, lineType=8)

        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", imgC)
        # cv2.waitKey(0)
        # exit()
        print('111111')
        cv2.imwrite(".\output\{}Result.jpg".format(i), imgC, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def ShowParam(ret, mtx, dist, rvecs, tvecs, newcameramtx):
    print("ret:", ret)
    print("\nmtx:\n", mtx)  # 内参数矩阵
    print("\ndist畸变值:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("\nrvecs旋转（向量）外参:\n", rvecs)  # 旋转向量  # 外参数
    print("\ntvecs平移（向量）外参:\n", tvecs)  # 平移向量  # 外参数

    print('\nnewcameramtx外参', newcameramtx)

if __name__ == "__main__":
    # 找棋盘格角点
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)  # 阈值
    # 棋盘格模板规格
    w = 9  # 10 - 1
    h = 6  # 7  - 1

    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp = objp * Per  # mm

    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = []  # 在世界坐标系中的三维点
    imgpoints = []  # 在图像平面的二维点
    # 加载pic文件夹下所有的jpg图像
    images = glob.glob('./*.jpg')  # 拍摄的十几张棋盘图片所在目录

    i = 0
    imgList = []
    for fname in images:

        img = cv2.imread(fname)
        imgList.append(img.copy())
        # 获取画面中心点
        # 获取图像的长宽
        h1, w1 = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        u, v = img.shape[:2]
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # 如果找到足够点对，将其存储起来
        if ret == True:
            print("i:", i)
            i = i + 1
            # 在原角点的基础上寻找亚像素角点
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 追加进入世界三维点和平面二维点中
            objpoints.append(objp)
            #print(corners)
            imgpoints.append(corners)
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (w, h), corners, ret)
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', 640, 480)
            cv2.imshow('findCorners', img)
            cv2.waitKey(200)
    cv2.destroyAllWindows()

    # %% 标定
    print('正在计算')
    # 标定
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))

    #ShowParam(ret, mtx, dist, rvecs, tvecs,newcameramtx)


    cv2.destroyAllWindows()

    PutBox(imgList, mtx, rvecs, tvecs, dist)
