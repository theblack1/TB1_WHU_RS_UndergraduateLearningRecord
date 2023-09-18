# coding = utf8

import io
from PIL import Image, ImageTk
import tkinter as tk

import cv2
import numpy as np

MIN_MATCH_COUNT = 4

def detect_sift(img1, draw = '', showFlag = 0):
    img = img1.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    kp = sift.detect(gray, None)

    img = cv2.drawKeypoints(gray, kp, img)

    if showFlag == 1:
        cv2.imshow('sp', img)
        cv2.waitKey(0)

    if len(draw) != 0:
        cv2.imwrite(draw, img)


def match_shift(img1, img2, draw):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    match = cv2.FlannBasedMatcher(dict(algorithm=2, trees=1), {})
    kp1, de1 = sift.detectAndCompute(g1, None)
    kp2, de2 = sift.detectAndCompute(g2, None)

    m = match.knnMatch(de1, de2, 2)
    m = sorted(m, key=lambda x: x[0].distance)
    ok = [m1 for (m1, m2) in m if m1.distance < 0.7 * m2.distance]

    med = cv2.drawMatches(img1, kp1, img2, kp2, ok, None)

    cv2.imwrite(draw, med)



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # inputImg = cv2.imread('test/input/0.png')
    # detect_sift(inputImg, draw='test/output/0_sp.png', showFlag=1)

    img1 = cv2.imread('test/input/b1.jpg')
    img2 = cv2.imread('test/input/b2.jpg')

    match_shift(img1, img2, draw = 'test/output/b1_b2_match.jpg')
