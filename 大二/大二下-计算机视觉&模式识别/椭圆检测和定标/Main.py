import FindEllipse
import DataCal
import cv2 as cv


def ExchangeXY(data_set, inParam):
    M = inParam['h']
    new_set = []
    for p in data_set:
        # 筛选0点
        if p[0] == 0 and p[1] == 0:
            new_set.append([0, 0])
            continue
        new_set.append([p[0], M - p[1]])

    return new_set

def readControl(referFile):
    data = []
    file = open(referFile, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行

    for index, row in enumerate(file_data):
        if (index == 0):
            continue

        tmpList = row.replace('\n', '').split("  ")
        data.append(tmpList)  # 将每行数据插入data中

    for d in data:
        d[3] = float(d[3])
        d[1] = float(d[1])
        d[2] = float(d[2])
    return data

def readParam(choice):
    # 内部参数
    x0 = 1935.5000000000
    y0 = 1295.5000000000
    f = 7935.786962
    width = 3872
    height = 2592

    inParam = {'x0': x0, 'y0': y0, 'f': f, 'w': width, 'h': height}

    # 外部参数初始值
    if choice == 0:
        Xs = 350.0
        Ys = 520.0
        Zs = 300.0
        Phi = -0.9209585845351669
        Omega = - 0.8780690992255569
        Kappa = 2.1102441253730908
    else:
        Xs = 130.0
        Ys = 610.0
        Zs = 300.0
        Phi = -0.4545695618308865
        Omega = - 1.1219147656947568
        Kappa = 2.6531277188826015

    outParam0 = {'Xs': Xs, 'Ys': Ys, 'Zs': Zs, 'Phi': Phi, 'Ome': Omega, 'Kap': Kappa}

    return inParam, outParam0

if __name__ == "__main__":

    # 数据读取
    imgList = [cv.imread("left.bmp"), cv.imread("right.bmp")]

    RealPoint = readControl("control.txt")
    real_xyz = []
    for p in RealPoint:
        temp = [p[1], p[2], p[3]]
        real_xyz.append(temp)

    for Choice, img in enumerate(imgList):
        # 椭圆检测
        retvalList_real, imgC = FindEllipse.GetEllipse(img, showFlag=0)

        # 标记匹配
        imgResult, picture_xy = FindEllipse.GetMark(imgC, retvalList_real, RealPoint, Auto=Choice, showFlag=0)

        # # 测试
        # testParamCal(real_xyz, picture_xy)

        # 读取内外参数
        inParam, outParam0 = readParam(Choice)

        # 迭代计算外部参数
        cal_xy, outParamFix = DataCal.CalData(inParam, outParam0, real_xyz, ExchangeXY(picture_xy, inParam), Choice)
        cal_xy = ExchangeXY(cal_xy, inParam)

        # 数据输出
        print()
        print('图像{}'.format(Choice + 1))
        print(' 真实值：', picture_xy)
        print(' 计算值：', cal_xy)
        print()
        print(' 修正外部参数', outParamFix)

        # 结果展示
        img2 = img.copy()
        FindEllipse.DisplayResult(imgC, cal_xy, showFlag=1, color = (0, 0, 255))

    #结束
    cv.destroyAllWindows()