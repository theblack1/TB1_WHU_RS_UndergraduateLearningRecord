import cv2
import numpy as np
import os
import time,datetime

class CvImshow():
    # 创建类，自动处理并显示
    def __init__(self, img_array, win_name = 'img', _stretch = False, _enhance = False, _fake_color = None, save_name = ''):
        # 全局变量
        self.g_window_name = win_name  # 窗口名
        self.g_window_wh = [800, 600]  # 窗口宽高

        self.g_location_win = [0, 0] # 相对于大图，窗口在图片中的位置
        self.location_win = [0, 0]  # 鼠标左键点击时，暂存self.g_location_win
        self.g_location_click, self.g_location_release = [0, 0], [0, 0]  # 相对于窗口，鼠标左键点击和释放的位置

        self.g_zoom, self.g_step = 1, 0.1  # 图片缩放比例和缩放系数
        
        self.g_image_original = np.zeros([100,100], dtype=np.uint8) # 原始图像
        
        img_array_fix = np.copy(img_array)
        
        # 数据拉伸
        if _stretch:
            cv2.normalize(np.copy(img_array),img_array_fix,0,255,cv2.NORM_MINMAX)
            
        
        # 直方图强化+伪彩色合成
        if _enhance:
            img_array_fix_enhance = self.enhance_grey(src=img_array_fix, _fake_color=_fake_color)
            self.g_image_original = img_array_fix_enhance
            # 如果准备保存
            if len(save_name):
                img_array_fix_toSave = np.copy(img_array)
                cv2.normalize(np.copy(img_array.astype(np.uint8)),img_array_fix_toSave,0,255,cv2.NORM_MINMAX)
                self.save_result(save_name + "_origin", img_array_fix_toSave)
                self.save_result(save_name + "_enhance", img_array_fix_enhance)
        else:
            self.g_image_original = img_array_fix
            if len(save_name):
                img_array_fix_toSave = np.copy(img_array)
                cv2.normalize(np.copy(img_array.astype(np.uint8)),img_array_fix_toSave,0,255,cv2.NORM_MINMAX)
                self.save_result(save_name, img_array_fix_toSave)
        
        
        
        self.g_image_zoom = self.g_image_original.copy()  # 缩放后的图片
        self.g_image_show = self.g_image_original[self.g_location_win[1]:self.g_location_win[1] + self.g_window_wh[1], self.g_location_win[0]:self.g_location_win[0] + self.g_window_wh[0]]  # 实际显示的图片

        # 设置窗口
        cv2.namedWindow(self.g_window_name, cv2.WINDOW_NORMAL)
        # 设置窗口大小，只有当图片大于窗口时才能移动图片
        cv2.resizeWindow(self.g_window_name, self.g_window_wh[0], self.g_window_wh[1])
        cv2.moveWindow(self.g_window_name, 700, 100)  # 设置窗口在电脑屏幕中的位置
        # 鼠标事件的回调函数
        cv2.setMouseCallback(self.g_window_name, self.mouse)
        cv2.waitKey()  # 不可缺少，用于刷新图片，等待鼠标操作
        cv2.destroyAllWindows()

    # 数据对比度强化
    def enhance_grey(self, src,  _fake_color = None):
        dst = cv2.equalizeHist(src.astype(np.uint8))       #自动调整图像对比度，把图像变得更清晰
        
        # 假彩色
        if _fake_color:
            dst = cv2.applyColorMap(cv2.convertScaleAbs(dst), _fake_color)
        
        return dst
        
    # 保存内容
    def save_result(self, file_name, img, driver = 'jpg'):
        path = "./data/imshow_save"
        # 递归创建文件夹
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_string = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss",time.localtime(time.time()))
        file_save_name = path + '/' + time_string + '_' + file_name + '.' + driver
        # file_save_name = path + '/' + file_name + '.' + driver
        
        cv2.imwrite(file_save_name, img)
    
    # 矫正窗口在图片中的位置
    # img_wh:图片的宽高, win_wh:窗口的宽高, win_xy:窗口在图片的位置
    def check_location(self, img_wh, win_wh, win_xy):
        for i in range(2):
            if win_xy[i] < 0:
                win_xy[i] = 0
            elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
                win_xy[i] = img_wh[i] - win_wh[i]
            elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
                win_xy[i] = 0
        # print(img_wh, win_wh, win_xy)

    # 计算缩放倍数
    # flag：鼠标滚轮上移或下移的标识, step：缩放系数，滚轮每步缩放0.1, zoom：缩放倍数
    def count_zoom(self,flag, step, zoom):
        if flag > 0:  # 滚轮上移
            zoom += step
            if zoom > 1 + step * 20:  # 最多只能放大到3倍
                zoom = 1 + step * 20
        else:  # 滚轮下移
            zoom -= step
            if zoom < step:  # 最多只能缩小到0.1倍
                zoom = step
        zoom = round(zoom, 2)  # 取2位有效数字
        return zoom

    # OpenCV鼠标事件
    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            self.g_location_click = [x, y]  # 左键点击时，鼠标相对于窗口的坐标
            self.location_win = [self.g_location_win[0], self.g_location_win[1]]  # 窗口相对于图片的坐标，不能写成location_win = self.g_location_win
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            self.g_location_release = [x, y]  # 左键拖曳时，鼠标相对于窗口的坐标
            h1, w1 = self.g_image_zoom.shape[0:2]  # 缩放图片的宽高
            w2, h2 = self.g_window_wh  # 窗口的宽高
            show_wh = [0, 0]  # 实际显示图片的宽高
            if w1 < w2 and h1 < h2:  # 图片的宽高小于窗口宽高，无法移动
                show_wh = [w1, h1]
                self.g_location_win = [0, 0]
            elif w1 >= w2 and h1 < h2:  # 图片的宽度大于窗口的宽度，可左右移动
                show_wh = [w2, h1]
                self.g_location_win[0] = self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]
            elif w1 < w2 and h1 >= h2:  # 图片的高度大于窗口的高度，可上下移动
                show_wh = [w1, h2]
                self.g_location_win[1] = self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]
            else:  # 图片的宽高大于窗口宽高，可左右上下移动
                show_wh = [w2, h2]
                self.g_location_win[0] = self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]
                self.g_location_win[1] = self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]
            self.check_location([w1, h1], [w2, h2], self.g_location_win)  # 矫正窗口在图片中的位置
            self.g_image_show = self.g_image_zoom[self.g_location_win[1]:self.g_location_win[1] + show_wh[1], self.g_location_win[0]:self.g_location_win[0] + show_wh[0]]  # 实际显示的图片
        elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
            z = self.g_zoom  # 缩放前的缩放倍数，用于计算缩放后窗口在图片中的位置
            self.g_zoom = self.count_zoom(flags, self.g_step, self.g_zoom)  # 计算缩放倍数
            w1, h1 = [int(self.g_image_original.shape[1] * self.g_zoom), int(self.g_image_original.shape[0] * self.g_zoom)]  # 缩放图片的宽高
            w2, h2 = self.g_window_wh  # 窗口的宽高
            self.g_image_zoom = cv2.resize(self.g_image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
            show_wh = [0, 0]  # 实际显示图片的宽高
            if w1 < w2 and h1 < h2:  # 缩放后，图片宽高小于窗口宽高
                show_wh = [w1, h1]
                cv2.resizeWindow(self.g_window_name, w1, h1)
            elif w1 >= w2 and h1 < h2:  # 缩放后，图片高度小于窗口高度
                show_wh = [w2, h1]
                cv2.resizeWindow(self.g_window_name, w2, h1)
            elif w1 < w2 and h1 >= h2:  # 缩放后，图片宽度小于窗口宽度
                show_wh = [w1, h2]
                cv2.resizeWindow(self.g_window_name, w1, h2)
            else:  # 缩放后，图片宽高大于窗口宽高
                show_wh = [w2, h2]
                cv2.resizeWindow(self.g_window_name, w2, h2)
            self.g_location_win = [int((self.g_location_win[0] + x) * self.g_zoom / z - x), int((self.g_location_win[1] + y) * self.g_zoom / z - y)]  # 缩放后，窗口在图片的位置
            self.check_location([w1, h1], [w2, h2], self.g_location_win)  # 矫正窗口在图片中的位置
            # print(self.g_location_win, show_wh)
            self.g_image_show = self.g_image_zoom[self.g_location_win[1]:self.g_location_win[1] + show_wh[1], self.g_location_win[0]:self.g_location_win[0] + show_wh[0]]  # 实际的显示图片
        cv2.imshow(self.g_window_name, self.g_image_show)