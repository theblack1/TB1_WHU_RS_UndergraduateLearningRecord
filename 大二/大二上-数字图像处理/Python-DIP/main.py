#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2 as cv

import Binary
import Hist
import LinearTransformation as Linear
import Filter
import hough
import noise
import matplotlib.pyplot as plt


root = './input_data'
output_root = './output_data'

filename = ['/cat.jpg', '/leaf.jpg', '/noodles.jpg', '/noisy_noodles.jpg', '/subset-blue-byte-297-216.bmp']


def ShowImage(img, name, gray = 0):

    cv.namedWindow(name, cv.WINDOW_KEEPRATIO)  # 防止图像过大，自动化窗口大小

    if (gray == 1):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow(name, img)
    cv.waitKey()

def LinearChange():
    src_img = cv.imread(root + filename[4])
    new_img = Linear.garyLinearTrans(src_img, [0, 255])    #灰度线性变换

    plt.subplot(2, 2, 1)
    plt.imshow(src_img)
    plt.title('Origin')

    plt.subplot(2, 2, 2)
    plt.hist(src_img.ravel(), 256, [0, 256])

    plt.subplot(2, 2, 3)
    plt.imshow(new_img)
    plt.title('Output')

    plt.subplot(2, 2, 4)
    plt.hist((new_img*256).ravel(), 256, [0, 256])

    plt.show()

def FilterChange():
    src_img = cv.imread(root + filename[3])
    new_img = Filter.Filter(src_img)   #中值滤波

    ShowImage(src_img, 'Original')
    ShowImage(new_img, 'Output')


def BinaryChange():
    src_img = cv.imread(root + filename[1])
    new_img = Binary.BinaryState(src_img)   #二值化

    ShowImage(src_img, 'Original', gray = 1)
    ShowImage(new_img, 'Output')



def HistChange():
    src_img = cv.imread(root + filename[0])
    ref_img = cv.imread(root + filename[3])

    new_img = Hist.hist_change(src_img, ref_img)  # 直方图规定化

    src_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    plt.subplot(3, 2, 1)
    plt.imshow(src_img, cmap='gray')
    plt.title('Origin')

    ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    plt.subplot(3, 2, 3)
    plt.imshow(ref_img, cmap='gray')
    plt.title('Reference')

    plt.subplot(3, 2, 5)
    plt.imshow(new_img, cmap='gray')
    plt.title('Output')

    plt.subplot(3, 2, 2)
    plt.hist(src_img.ravel(), 256, [0, 256])

    plt.subplot(3, 2, 4)
    plt.hist(ref_img.ravel(), 256, [0, 256])

    plt.subplot(3, 2, 6)
    plt.hist(new_img.ravel(), 256, [0, 256])



    plt.show()


if __name__ == '__main__':
    #LinearChange()
    #FilterChange()
    #BinaryChange()
    #HistChange()

    src_img = cv.imread(root + filename[0])
    new_img=hough.houghline(src_img)

    # 结束
    cv.destroyAllWindows()
