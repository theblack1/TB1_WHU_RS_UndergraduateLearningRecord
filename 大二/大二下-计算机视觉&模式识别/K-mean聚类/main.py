import cv2
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

from scipy import io as spio
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决windows环境下画图汉字乱码问题

def KMeans(X, K, max_iters):
    '''二维数据聚类过程演示'''
    # initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])  # 初始化类中心
    initial_centroids = kMeansInitCentroids(X,K)
    centroids, idx = runKMeans(X, initial_centroids, max_iters, True)  # 执行K-Means聚类算法
    print(centroids)
    print(idx)


# 找到每条数据距离哪个类中心最近
def findClosestCentroids(X, initial_centroids):
    m = X.shape[0]  # 数据条数
    K = initial_centroids.shape[0]  # 类的总数
    dis = np.zeros((m, K))  # 存储计算每个点分别到K个类的距离
    idx = np.zeros((m, 1))  # 要返回的每条数据属于哪个类


    '''计算每个点到每个类中心的距离'''
    for i in range(m):
        for j in range(K):
            dis[i, j] = np.dot((X[i, :] - initial_centroids[j, :]).reshape(1, -1),
                               (X[i, :] - initial_centroids[j, :]).reshape(-1, 1))

    '''返回dis每一行的最小值对应的列号，即为对应的类别
    - np.min(dis, axis=1)返回每一行的最小值
    - np.where(dis == np.min(dis, axis=1).reshape(-1,1)) 返回对应最小值的坐标
     - 注意：可能最小值对应的坐标有多个，where都会找出来，所以返回时返回前m个需要的即可（因为对于多个最小值，属于哪个类别都可以）
    '''
    dummy, idx = np.where(dis == np.min(dis, axis=1).reshape(-1, 1))
    return idx[0:dis.shape[0]]  # 注意截取一下


# 计算类中心
def computerCentroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros((K, n))
    for i in range(K):
        centroids[i, :] = np.mean(X[np.ravel(idx == i), :], axis=0).reshape(1,
                                                                            -1)  # 索引要是一维的,axis=0为每一列，idx==i一次找出属于哪一类的，然后计算均值


# 聚类算法
def runKMeans(X, initial_centroids, max_iters, plot_process):
    m, n = X.shape  # 数据条数和维度
    K = initial_centroids.shape[0]  # 类数
    centroids = initial_centroids  # 记录当前类中心
    previous_centroids = centroids  # 记录上一次类中心
    idx = np.zeros((m, 1))  # 每条数据属于哪个类
    previous_centroids=

    for i in range(max_iters):  # 迭代次数
        print(u'迭代计算次数：%d' % (i + 1))
        idx = findClosestCentroids(X, centroids)
        if plot_process:  # 如果绘制图像
            plt = plotProcessKMeans(X, centroids, previous_centroids)  # 画聚类中心的移动过程
            previous_centroids = centroids  # 重置
        centroids = computerCentroids(X, idx, K)  # 重新计算类中心
    if plot_process:  # 显示最终的绘制结果
        plt.show()
    return centroids, idx  # 返回聚类中心和数据属于哪个类


# 画图，聚类中心的移动过程
def plotProcessKMeans(X, centroids, previous_centroids):
    plt.scatter(X[:, 0], X[:, 1])  # 原数据的散点图
    plt.plot(previous_centroids[:, 0], previous_centroids[:, 1], 'rx', markersize=10, linewidth=5.0)  # 上一次聚类中心
    plt.plot(centroids[:, 0], centroids[:, 1], 'rx', markersize=10, linewidth=5.0)  # 当前聚类中心
    for j in range(centroids.shape[0]):  # 遍历每个类，画类中心的移动直线
        p1 = centroids[j, :]
        p2 = previous_centroids[j, :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "->", linewidth=2.0)
    return plt


# 初始化类中心--随机取K个点作为聚类中心
def kMeansInitCentroids(X, K):
    m = X.shape[0]
    m_arr = np.arange(0, m)  # 生成0-m-1
    centroids = np.zeros((K, X.shape[1]))
    np.random.shuffle(m_arr)  # 打乱m_arr顺序
    rand_indices = m_arr[:K]  # 取前K个
    centroids = X[rand_indices, :]
    return centroids


def K_meaning_txt():
    ##########加载数据############
    def load_data_set():
        """
        加载数据集
        :return:返回两个数组，普通数组
            data_arr -- 原始数据的特征
            label_arr -- 原始数据的标签，也就是每条样本对应的类别
        """
        data_arr = []
        label_arr = []
        # 如果想下载参照https://github.com/varyshare/AiLearning/blob/master/data/6.SVM/testSet.txt
        # 欢迎follow的我github
        f = open('myspace/svm_data.txt', 'r')
        for line in f.readlines():
            line_arr = line.strip().split()
            data_arr.append([np.float(line_arr[0]), np.float(line_arr[1])])
            label_arr.append(int(line_arr[2]))
        return np.array(data_arr), np.array(label_arr)

    x, label = load_data_set()
    # 绘制出数据点分析看有几个聚类
    plt.scatter(x[:,0],x[:,1])

    ##############k-Means算法#################
    # 创建k个聚类数组，用于存放属于该聚类的点
    clusters = []
    p1 = [6, 4]
    p2 = [1, 3]
    cluster_center = np.array([p1, p2])
    k = 2
    for i in range(k):
        clusters.append([])

    epoch = 3
    for _ in range(epoch):
        for i in range(k):
            clusters[i] = []
        # 计算所有点到这k个聚类中心的距离
        for i in range(x.shape[0]):
            xi = x[i]
            distances = np.sum((cluster_center - xi) ** 2, axis=1)
            # 离哪个聚类中心近，就把这个点序号加到哪个聚类中
            c = np.argmin(distances)
            clusters[c].append(i)

        # 重新计算k个聚类的聚类中心（每个聚类所有点加起来取平均）
        for i in range(k):
            cluster_center[i] = np.sum(x[clusters[i]], axis=0) / len(clusters[i])

    plt.scatter(x[clusters[0], 0], x[clusters[0], 1])
    plt.scatter(x[clusters[1], 0], x[clusters[1], 1])

def K_meaning_pic(img, K,draw ='', showFlag = 0):
    if showFlag == 1:
        cv2.namedWindow('origin', cv2.WINDOW_FREERATIO)
        cv2.imshow('origin', img)
        cv2.waitKey()

    #change img(2D) to 1D
    img1 = img.reshape((img.shape[0]*img.shape[1],1))
    img1 = np.float32(img1)

    #define criteria = (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

    #set flags: hou to choose the initial center
    #---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
    flags = cv2.KMEANS_RANDOM_CENTERS
    # apply kmenas
    compactness, labels, centers = cv2.kmeans(img1,K,None,criteria,10,flags)

    # 中心绘制
    labels_c = []
    for i in tqdm(labels):
        labels_c.append(centers[i-1])

    labels_c = np.array(labels_c)

    # 转换成图像
    img2 = labels_c.reshape((img.shape[0], img.shape[1]))
    img2 = np.array(img2, dtype=np.uint8)

    if showFlag == 1:
        cv2.namedWindow('k-meaning', cv2.WINDOW_FREERATIO)
        cv2.imshow('k-meaning', img2)
        cv2.waitKey()

    if len(draw) != 0:
        string = draw.split('.')
        cv2.imwrite(string[0]+'_'+ str(K) +'.'+ string[1], img2)

if __name__ == '__main__':
    data = spio.loadmat("test/input/data.mat")
    X = data['X']
    KMeans(X, 3, 10)

    # img = cv2.imread('test/input/test.jpg', cv2.IMREAD_GRAYSCALE)
    # K_meaning_pic(img, 3, 'test/output/test.jpg', showFlag = 1)
