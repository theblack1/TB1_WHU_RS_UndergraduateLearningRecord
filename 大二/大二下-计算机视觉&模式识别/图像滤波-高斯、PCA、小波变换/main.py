from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import cv2 as cv
import numpy as np
import os

import pywt
from PIL import Image

import matplotlib.pyplot as plt

# 读取文件夹中的所有图片
def read_all_photo(directory_name):
    imgList = []
    names = []
    for filename in os.listdir(directory_name):
        imgfile = directory_name + "/" + filename
        img = cv.imdecode(np.fromfile(imgfile, dtype=np.uint8), -1)
        imgList.append(img)

        name = filename.split('.')
        names.append(name[0])


    return imgList, names




# 椒盐噪声添加
def noise(img,snr):
    h=img.shape[0]
    w=img.shape[1]
    img1=img.copy()
    sp=h*w   # 计算图像像素点个数
    NP=int(sp*(1-snr))   # 计算图像椒盐噪声点个数
    for i in range (NP):
        randx=np.random.randint(1,h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy=np.random.randint(1,w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random()<=0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx,randy]=0
        else:
            img1[randx,randy]=255
    return img1

# 高斯滤波
def denoising1(image):
    new = cv.GaussianBlur(image, (3, 3), 1)
    # new = cv.medianBlur(new, 3)
    # new = cv.blur(new, (3, 3), borderType=cv.BORDER_REPLICATE)
    return new

# PCA去噪
def image_svd(n, pic):
    a, b, c = np.linalg.svd(pic)
    svd = np.zeros((a.shape[0], c.shape[1]))
    for i in range(0, n):
        svd[i, i] = b[i]
    img = np.matmul(a, svd)
    img = np.matmul(img, c)
    img[img >= 255] = 255
    img[0 >= img] = 0
    img = img.astype(np.uint8)
    return img

def denoising3(image):
    img = image
    h, w = img.shape[:2]

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    plt.figure(figsize=(50, 100))
    i = 10
    j = int(h * 0.005 * i * i)
    r_img = image_svd(j, r)
    g_img = image_svd(j, g)
    b_img = image_svd(j, b)
    pic = np.stack([r_img, g_img, b_img], axis=2)
    new = cv.cvtColor(np.asarray(pic), cv.COLOR_RGB2BGR)

    return new

# 小波变换
def denoising2(image):
    # 读取灰度图
    b, g, r = cv.split(image)
    RGB = [r, g, b]

    w = 'sym4'  # 定义小波基的类型
    l = 3  # 变换层次为3
    for index, img in enumerate(RGB):
        coeffs = pywt.wavedec2(data=img, wavelet=w, level=l)  # 对图像进行小波分解
        threshold = 0.04

        list_coeffs = []
        for i in range(1, len(coeffs)):
            list_coeffs_ = list(coeffs[i])
            list_coeffs.append(list_coeffs_)

        for r1 in range(len(list_coeffs)):
            for r2 in range(len(list_coeffs[r1])):
                # 对噪声滤波(软阈值)
                list_coeffs[r1][r2] = pywt.threshold(list_coeffs[r1][r2], threshold * np.max(list_coeffs[r1][r2]))

        rec_coeffs = []  # 重构系数
        rec_coeffs.append(coeffs[0])  # 将原图像的低尺度系数保留进来

        for j in range(len(list_coeffs)):
            rec_coeffs_ = tuple(list_coeffs[j])
            rec_coeffs.append(rec_coeffs_)

        denoised_img = pywt.waverec2(rec_coeffs, 'sym4')
        denoised_img = Image.fromarray(np.uint8(denoised_img))
        RGB[index] = denoised_img

    pic = Image.merge('RGB', RGB)
    cv2_img = cv.cvtColor(np.asarray(pic), cv.COLOR_RGB2BGR)
    new = cv.resize(cv2_img, (image.shape[1], image.shape[0]))

    return new

# 评估去噪结果
def envalue(origin, noise, new):
    # 峰值信噪比
    # print(origin.shape)
    # print(new.shape)
    psnr1 = peak_signal_noise_ratio(origin, noise)
    psnr2 = peak_signal_noise_ratio(origin, new)

    # 结构相似性
    ssim1 = structural_similarity(origin, noise, multichannel=True)
    ssim2 = structural_similarity(origin, new, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True

    result = {'noise psnr': psnr1, 'new psnr':psnr2, 'noise ssim':ssim1, 'new ssim':ssim2}
    return result

# 主函数
if __name__ == '__main__':
    # 读取所有图片
    imgList, names = read_all_photo('datas/before')
    noiseImgList, _ = read_all_photo('datas/noise')

    # 降噪
    newImgList1 = []
    newImgList2 = []
    newImgList3 = []
    for img in noiseImgList:
        # 单张图片去噪
        img1 = denoising1(img)
        img2 = denoising2(img)
        img3 = denoising3(img)

        # 添加进新数组
        newImgList1.append(img1)
        newImgList2.append(img2)
        newImgList3.append(img3)

    # cv.imshow('new', newImgList3[1])
    # cv.waitKey()

    # 评估降噪效果
        # 储存各个图片的降噪评价结果
    resultSet1 = []
    resultSet2 = []
    resultSet3 = []
    for i in range(len(imgList)):
        # 计算单个图片的降噪效果
        result1 = envalue(imgList[i], noiseImgList[i], newImgList1[i])
        result2 = envalue(imgList[i], noiseImgList[i], newImgList2[i])
        result3 = envalue(imgList[i], noiseImgList[i], newImgList3[i])

        # 记录进入集合中
        resultSet1.append(result1)
        resultSet2.append(result2)
        resultSet3.append(result3)

        # 保存结果
        cv.imwrite('output/GaussianBlur/' + names[i] + '_new.jpg', newImgList1[i])
        cv.imwrite('output/SVD/' + names[i] + '_new.jpg', newImgList1[i])
        cv.imwrite('output/WaveletTransformation/' + names[i] + '_new.jpg', newImgList1[i])


    print(resultSet1)
    print(resultSet2)
    print(resultSet3)

