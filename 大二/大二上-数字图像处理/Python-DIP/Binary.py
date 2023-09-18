import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plt_hist(img):
    # hist = cv.calcHist([img],
    #                    [0],  # 使用的通道
    #                    None,  # 没有使用mask
    #                    [256],  # HistSize
    #                    [0.0, 255.0])  # 直方图柱的范围
    # cv.imshow('Hist', hist)
    # cv.waitKey()
     plt.hist(img.ravel(), 256, [0, 256])
     plt.show()


def BinaryState(src):

    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    plt_hist(src)
    T = float(input('请输入阈值：'))

    dist = np.zeros_like(src)
    shape_ = src.shape
    print('正在进行二值化（状态法）')
    for i in range(shape_[0]):
        if (i % 2 == 0):
            print('\r--{}/{}--{:.2f}%--'.format(i + 1, shape_[0], (i+1)*100/shape_[0]),
                  end='', flush=True)
        #elif (i % 5 == 0):
        else:
            print('\r||{}/{}||{:.2f}%||'.format(i + 1, shape_[0], (i+1)*100/shape_[0]),
                  end='', flush=True)
        for j in range(shape_[1]):
                if(src[i][j] >= T):
                    dist[i][j] = 255
                    #print(1)
    print()
    return dist

