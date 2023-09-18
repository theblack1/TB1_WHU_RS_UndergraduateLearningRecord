import numpy as np
import cv2 as cv

def hist_change(image, ref):
    #灰度化
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)

    #设置新图
    out = np.zeros_like(image)

    #计算直方图
    hist_img, _ = np.histogram(image[:, :], 256)
    hist_ref, _ = np.histogram(ref[:, :], 256)
    #累加运算
    sum_img = np.cumsum(hist_img)
    sum_ref = np.cumsum(hist_ref)

    # 匹配灰度
    for i in range(256):
        # 原图每一个灰度级与参考图各个灰度级比较，选择相差最小的
        cost = abs(sum_img[i] - sum_ref)
        cost = cost.tolist()    #转化为列表方便使用index函数得到索引
        index_ = cost.index(min(cost))  # 找出tmp中最小的数，得到这个数的索引(目标灰度值)

        # 进行对应规定化
        out[:, :][image[:, :] == i] = index_

    return out