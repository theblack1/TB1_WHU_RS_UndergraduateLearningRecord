import numpy as np
import cv2
from tqdm import trange
import math
import numba as nb
import time
# 自定义包
import imshow_cv
from read_radar import ReadRadar

# 中值滤波
@nb.jit(nopython=True, nogil=True)
def fast_median_blur(input_img, win_size):
    # 读取基础数据
    h=int(input_img.shape[0])
    w=int(input_img.shape[1])

    # 准备输出图像
    blured = input_img.copy()
    
    # 计算边界值
    barry_size = int((win_size-1)/2)
    
    # 只遍历中间的值
    for i in range(1,h-barry_size):
        for j in range(1,w-barry_size):
            # 每个窗口所有数值记录在这里
            win_array = []
            # 从左上到右下读取，第一个是(i-barry_size, j-barry_size),最后一个是(i+barry_size, j+barry_size)
            for w1 in range(-barry_size, barry_size+1):
                for w2 in range(-barry_size, barry_size+1):
                    win_array.append(input_img[i+w1,j+w2])
            
            # 数值转换
            win_array = np.array(win_array)
            
            # 执行运算
            result = np.median(win_array)
            
            # 对应位置数值替换
            blured[i,j] = result
    
    return blured

# 均值滤波
@nb.jit(nopython=True, nogil=True)
def fast_avg_blur(input_img, win_size):
    # 读取基础数据
    h=int(input_img.shape[0])
    w=int(input_img.shape[1])

    # 准备输出图像
    blured = input_img.copy()
    
    # 计算边界值
    barry_size = int((win_size-1)/2)
    
    # 只遍历中间的值
    for i in range(1,h-barry_size):
        for j in range(1,w-barry_size):
            # 每个窗口所有数值记录在这里
            win_array = []
            # 从左上到右下读取，第一个是(i-barry_size, j-barry_size),最后一个是(i+barry_size, j+barry_size)
            for w1 in range(-barry_size, barry_size+1):
                for w2 in range(-barry_size, barry_size+1):
                    win_array.append(input_img[i+w1,j+w2])
            
            # 数值转换
            win_array = np.array(win_array)
            
            # 执行运算
            result = np.average(win_array)
            
            # 对应位置数值替换
            blured[i,j] = result
    
    return blured

# Lee滤波
@nb.jit(nopython=True, nogil=True)
def fast_lee_blur(input_img, win_size, sigma):
    # 读取基础数据
    h=int(input_img.shape[0])
    w=int(input_img.shape[1])
    sigma2_v = sigma*sigma

    # 准备输出图像
    blured = input_img.copy()
    
    # 计算边界值
    barry_size = int((win_size-1)/2)
    
    # 只遍历中间的值
    for i in range(1,h-barry_size):
        for j in range(1,w-barry_size):
            # 每个窗口所有数值记录在这里
            win_array = []
            # 从左上到右下读取，第一个是(i-barry_size, j-barry_size),最后一个是(i+barry_size, j+barry_size)
            for w1 in range(-barry_size, barry_size+1):
                for w2 in range(-barry_size, barry_size+1):
                    win_array.append(input_img[i+w1,j+w2])
            
            # 数值转换
            win_array = np.array(win_array)
            # 反推数值
            avg_y = np.average(win_array)
            var_y = np.var(win_array)
            var_x = (var_y - sigma2_v*avg_y*avg_y)/(1+sigma2_v)
            b = var_x/var_y
            
            # 中心像元值
            y_c = win_array[int(len(win_array)/2-0.5)]
            result = avg_y + b*(y_c - avg_y)
            
            # 对应位置数值替换
            blured[i,j] = result
    
    return blured

# Forst滤波
@nb.jit(nopython=True, nogil=True)
def fast_forst_blur(input_img, K,win_size):
    # 读取基础数据
    h=int(input_img.shape[0])
    w=int(input_img.shape[1])
    
    K2 = 1

    # 准备输出图像
    forst_blured = np.zeros_like(input_img)
    
    # 计算边界值
    barry_size = int((win_size-1)/2)
    
    # 只遍历中间的值
    for i in range(1,h-barry_size):
        for j in range(1,w-barry_size):
            # 每个窗口所有数值记录在这里
            win_array = []
            # 从左上到右下读取，第一个是(i-barry_size, j-barry_size),最后一个是(i+barry_size, j+barry_size)
            for w1 in range(-barry_size, barry_size+1):
                for w2 in range(-barry_size, barry_size+1):
                    win_array.append(input_img[i+w1,j+w2])
            
            # 进行数值提取
            # 一维转二维
            win_array = np.array(win_array)
            win_size = int(math.sqrt(len(win_array)))
            win_mat = win_array.reshape(win_size, win_size)
            # 中心坐标（相对）
            center_i = int(win_size/2-0.5)
            center_j = int(win_size/2-0.5)
            # 均值标准差
            avg_I = np.average(win_array)
            std_I = np.std(win_array)
            
            # 计算结果
            result = 0
            for w1 in range(win_size):
                for w2 in range(win_size):
                    # 系数计算
                    CI = std_I/avg_I
                    alpha = math.sqrt(abs(K*CI*CI))
                    
                    #归一化常量
                    i_distance = w1-center_i
                    j_distance = w2-center_j
                    t = math.sqrt(i_distance*i_distance+j_distance*j_distance)
                    # 权重生成
                    m = K2 * alpha * math.exp(-alpha * abs(t))
                    # 累加
                    result += (m*win_mat[w1,w2])/len(win_array)
            
            # 对应位置数值替换
            forst_blured[i,j] = result
    
    return forst_blured


class ImgBlurTool():
    # 初始化
    def __init__(self, input_img):
        self.input_img = input_img
        # 读取基础数据
        
        # 储存结果图像
        self.median_blured = []
        self.average_blured = []
        self.lee_blured = []
        self.forst_blured = []
    
    # 中值滤波
    def median_blur(self, win_size = 3):
        # 检验窗口大小是否合法
        if win_size%2 != 1:
            print('Warnning: Size of the Window Should be Odd Number!')
            return []
        
        # 开始计时
        begin_time = time.time()
        
        # 打印启动信息
        print()
        print(f"正在运行'中值滤波';预计运行时间：{ 0.788*win_size*win_size + 22.44:.3f}秒")
        
        # 执行滤波
        self.median_blured = fast_median_blur(self.input_img, win_size)
        
        # 程序结束时间
        end_time = time.time()
        # 运行时间run_time。round()函数取整
        run_time = round(end_time-begin_time)
        # 计算时分秒
        hour = run_time//3600
        minute = (run_time-3600*hour)//60
        second = run_time-3600*hour-60*minute
        
        # 输出
        print (f'运行中值滤波(window size:{win_size})耗时:\n{hour}小时{minute}分钟{second}秒')
        
        
        return self.median_blured
    
    # 均值滤波
    def avg_blur(self, win_size = 3):
        # 检验窗口大小是否合法
        if win_size%2 != 1:
            print('Warnning: Size of the Window Should be Odd Number!')
            return []
        
        # 开始计时
        begin_time = time.time()
        
        # 打印启动信息
        print()
        print(f"正在运行'均值滤波';预计运行时间：{0.3955*win_size*win_size + 14.172:.3f}秒")
        
        # 执行滤波
        self.avg_blured = fast_avg_blur(self.input_img, win_size)
        
        # 程序结束时间
        end_time = time.time()
        # 运行时间run_time。round()函数取整
        run_time = round(end_time-begin_time)
        # 计算时分秒
        hour = run_time//3600
        minute = (run_time-3600*hour)//60
        second = run_time-3600*hour-60*minute
        
        # 输出
        print (f'运行均值滤波(window size:{win_size})耗时:\n{hour}小时{minute}分钟{second}秒')
        
        return self.avg_blured
    
    # Lee滤波
    def lee_blur(self, win_size = 3, sigma = 0.9):
        # 检验窗口大小是否合法
        if win_size%2 != 1:
            print('Warnning: Size of the Window Should be Odd Number!')
            return []
        
        # 开始计时
        begin_time = time.time()
        
        # 打印启动信息
        print()
        print(f"正在运行'Lee滤波';预计运行时间：{0.486*win_size*win_size + 14.036:.3f}秒")
        
        # 执行滤波
        self.lee_blured = fast_lee_blur(self.input_img, win_size, sigma)
        
        # 程序结束时间
        end_time = time.time()
        # 运行时间run_time。round()函数取整
        run_time = round(end_time-begin_time)
        # 计算时分秒
        hour = run_time//3600
        minute = (run_time-3600*hour)//60
        second = run_time-3600*hour-60*minute
        # 输出
        print (f'运行Lee滤波(window size:{win_size}, sigma:{sigma})耗时:\n{hour}小时{minute}分钟{second}秒')
        
        
        return self.lee_blured

    # Forst滤波
    def forst_blur(self, K = 1, win_size = 3):
        # 检验窗口大小是否合法
        if win_size%2 != 1:
            print('Warnning: Size of the Window Should be Odd Number!')
            return []
        
        # 开始计时
        begin_time = time.time()
        
        # 打印启动信息
        print()
        print(f"正在运行'Forst滤波';预计运行时间：{1.2509*win_size*win_size + 15.048:.3f}秒")
        
        # 执行滤波
        self.forst_blured = fast_forst_blur(self.input_img, K,win_size)
        
        # 程序结束时间
        end_time = time.time()
        # 运行时间run_time。round()函数取整
        run_time = round(end_time-begin_time)
        # 计算时分秒
        hour = run_time//3600
        minute = (run_time-3600*hour)//60
        second = run_time-3600*hour-60*minute
        # 输出
        print (f'运行Forst滤波(window size:{win_size}, K:{K})耗时:\n{hour}小时{minute}分钟{second}秒')
        
        return self.forst_blured

def show_in_cv(img_array, _enhance = False, win_name = "img", save_name = ''):
    imshow_cv.CvImshow(img_array, win_name=win_name, _stretch = True, _enhance = _enhance, _fake_color=cv2.COLORMAP_PLASMA, save_name=save_name)

if __name__ == '__main__':
    
    # 打开文件
    file_name_str = r"data/input_radar/GF3_HH_str.tif"
    file_name_raw = r"data/input_radar/GF3_KRN_QPSI_005782_W122.4_N37.6_20170915_L1A_HH_L10002599253.tiff"
    
    radar = ReadRadar(file_name_str)
    
    # 展示文件信息
    radar.show_info()
    
    # 展示第一波段信息
    radar.show_band_info(band_num = 1)

    # 展示图像
    show_in_cv(img_array=radar.str_array, _enhance = True, win_name="origin") #, save_name='GF3_HH_stretched'
    
    # 滤波处理
    process_blur = ImgBlurTool(input_img = radar.str_array)
    
    #1.中值滤波
    print("################# 1.中值滤波 ##################")
    media_blured_img_3x3 = process_blur.median_blur(win_size = 3)
    media_blured_img_5x5 = process_blur.median_blur(win_size = 5)
    media_blured_img_9x9 = process_blur.median_blur(win_size = 9)

    show_in_cv(media_blured_img_3x3, _enhance = True, win_name= "3x3_media_blured_img",
            save_name='GF3_HH_stretched_mediaBlurWin3')
    show_in_cv(media_blured_img_5x5, _enhance = True, win_name= "5x5_media_blured_img",
            save_name='GF3_HH_stretched_mediaBlurWin5')    
    show_in_cv(media_blured_img_9x9, _enhance = True, win_name= "9x9_media_blured_img",
            save_name='GF3_HH_stretched_mediaBlurWin9')
    
    #2.均值滤波
    print("################# 2.均值滤波 ##################")
    average_blured_img_3x3 = process_blur.avg_blur(win_size = 3)
    average_blured_img_5x5 = process_blur.avg_blur(win_size = 5)
    average_blured_img_9x9 = process_blur.avg_blur(win_size = 9)
    
    show_in_cv(average_blured_img_3x3, _enhance = True, win_name= "3x3_average_blured_img",
            save_name='GF3_HH_stretched_averageBlurWin3')
    show_in_cv(average_blured_img_5x5, _enhance = True, win_name= "5x5_average_blured_img",
            save_name='GF3_HH_stretched_averageBlurWin5')
    show_in_cv(average_blured_img_9x9, _enhance = True, win_name= "9x9_average_blured_img",
            save_name='GF3_HH_stretched_averageBlurWin9')

    #3.lee滤波
    print("################# 3.Lee滤波 ##################")
    lee_blured_img_3x3 = process_blur.lee_blur(win_size=3)
    lee_blured_img_5x5 = process_blur.lee_blur(win_size=5)
    lee_blured_img_9x9 = process_blur.lee_blur(win_size=9)
    
    show_in_cv(lee_blured_img_3x3, _enhance = True, win_name= "3x3_lee_blured_img",
            save_name='GF3_HH_stretched_leeBlurWin3Sigma.9')    
    show_in_cv(lee_blured_img_5x5, _enhance = True, win_name= "5x5_lee_blured_img",
            save_name='GF3_HH_stretched_leeBlurWin5Sigma.9')
    show_in_cv(lee_blured_img_9x9, _enhance = True, win_name= "9x9_lee_blured_img",
            save_name='GF3_HH_stretched_leeBlurWin9Sigma.9')

    # 4.forst滤波
    print("################# 4.Forst滤波 ##################")
    forst_blured_img_3x3 = process_blur.forst_blur(win_size = 3, K=1)
    forst_blured_img_5x5 = process_blur.forst_blur(win_size = 5, K=1)
    forst_blured_img_9x9 = process_blur.forst_blur(win_size = 9, K=1)
    
    show_in_cv(forst_blured_img_3x3, _enhance = True, win_name= "3x3_forst_blured_img",
            save_name='GF3_HH_stretched_forstBlurWin3K1')
    show_in_cv(forst_blured_img_5x5, _enhance = True, win_name= "5x5_forst_blured_img",
            save_name='GF3_HH_stretched_forstBlurWin5K1')
    show_in_cv(forst_blured_img_9x9, _enhance = True, win_name= "9x9_forst_blured_img",
            save_name='GF3_HH_stretched_forstBlurWin9K1')