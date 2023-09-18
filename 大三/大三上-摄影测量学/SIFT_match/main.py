import numpy as np
import cv2
import SIFT
import time

if __name__ == "__main__":
    MIN_MATCH_COUNT = 10 #？？？？

    # 读取图片
    # img1 = cv2.imread('box.png', 0)
    # img2 = cv2.imread('box_in_scene.png', 0)
    img1 = cv2.imread(r'data/GD952328.JPG', 0)
    img2 = cv2.imread(r'data/GD952329.JPG', 0)

    # 计算SIFT
    rescale = 0.4
    img1 = cv2.resize(img1, (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_CUBIC)
    kp1, des1 = SIFT.cal_sift(img1)
    kp2, des2 = SIFT.cal_sift(img2)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        h, w = img1.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # 接合图像
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2 # 总宽度设置
        nHeight = max(h1, h2) # 高度设置
        hdif = int((h2 - h1)/2) # 抬高高度
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8) # 图像尺寸

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        # 画SIFT连线
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

        cv2.imshow('matched', newimg)
        cv2.waitKey()
        more_information = '_rescale0_4'
        cv2.imwrite(r'output/' + str(time.time()) + more_information +'.jpg', newimg)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
