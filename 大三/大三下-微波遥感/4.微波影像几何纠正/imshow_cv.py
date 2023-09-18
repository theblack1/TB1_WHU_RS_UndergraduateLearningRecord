# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import time
from threading import Thread

DEFAULT_FAKE_COLOR = cv2.COLORMAP_PLASMA

class CvImshow():
    # 创建类，自动处理并显示
    def __init__(self, img_array, win_name = 'img', _stretch = True, _hist_enhance = False, _fake_color = DEFAULT_FAKE_COLOR,
                save_name = '', step = 0.9, max_zoom=3, _show = True, _each_stretch = False):
        # 准备输出的数据
        self.output_img_dict = {}
        
        # 读取数据
        self._hist_enhance = _hist_enhance
        self._fake_color = _fake_color
        self.save_name = save_name
        self._each_stretch = _each_stretch
        
        if _each_stretch:
            _stretch = True
        
        # 自适应窗口宽高
        self.origin_img_wh = np.shape(img_array)
        self.origin_h = self.origin_img_wh[0]
        self.origin_w = self.origin_img_wh[1]
        
        if self.origin_w > self.origin_h:
            # 宽形
            self.g_window_wh = [1920, int(1080*self.origin_h/self.origin_w)]  # 窗口宽高
        else:
            # 高型
            self.g_window_wh = [int(1080*self.origin_w/self.origin_h), 1080]  # 窗口宽高
        
        # 全局变量
        self.g_window_name = win_name  # 窗口名

        self.g_location_win = [0, 0] # 相对于大图，窗口在图片中的位置
        self.location_win = [0, 0]  # 鼠标左键点击时，暂存self.g_location_win
        self.g_location_click, self.g_location_release = [0, 0], [0, 0]  # 相对于窗口，鼠标左键点击和释放的位置

        self.g_zoom, self.g_step = 1, step  # 图片缩放比例和缩放系数
        self.max_zoom = max_zoom
        
        self.g_image_original = np.zeros_like(img_array, dtype=np.uint8) # 原始图像
        
        img_array_fix = np.copy(img_array)
        
        # 数据拉伸
        if _stretch:
            if img_array_fix.ndim == 3 and self._each_stretch:
                # 如果多通道图像要各自拉伸
                # 通道拆分
                b_img, g_img, r_img = cv2.split(img_array_fix)
                
                # 各自处理
                cv2.normalize(b_img,b_img,0,255,cv2.NORM_MINMAX)
                cv2.normalize(g_img,g_img,0,255,cv2.NORM_MINMAX)
                cv2.normalize(r_img,r_img,0,255,cv2.NORM_MINMAX)
                
                # 通道合并
                img_array_fix = cv2.merge([b_img, g_img, r_img])
            else:    
                cv2.normalize(img_array,img_array_fix,0,255,cv2.NORM_MINMAX)
            
            # 保存数据
            self.output_img_dict['stretch'] = img_array_fix
            if img_array_fix.ndim == 3:
                # 通道拆分
                b_img, g_img, r_img = cv2.split(img_array_fix)
                self.output_img_dict['stretch_b'] = b_img
                self.output_img_dict['stretch_g'] = g_img
                self.output_img_dict['stretch_r'] = r_img
        
        # 直方图强化或伪彩色合成
        if self._hist_enhance or (self._fake_color and (not img_array.ndim == 3)):
            img_array_fix_enhance = self.enhance(src=img_array_fix)
            self.g_image_original = img_array_fix_enhance
            # 如果准备保存
            if len(self.save_name):
                img_array_fix_toSave = np.copy(img_array)
                cv2.normalize(np.copy(img_array.astype(np.uint8)),img_array_fix_toSave,0,255,cv2.NORM_MINMAX)
                # self.save_result(self.save_name + "_origin", img_array_fix_toSave)
                self.save_result(self.save_name + "(enhanced)", img_array_fix_enhance)
        else:
            self.g_image_original = img_array_fix
            if len(self.save_name):
                img_array_fix_toSave = np.copy(img_array)
                cv2.normalize(np.copy(img_array.astype(np.uint8)),img_array_fix_toSave,0,255,cv2.NORM_MINMAX)
                self.save_result(self.save_name, img_array_fix_toSave)
        
        self.g_image_zoom = self.g_image_original.copy()  # 缩放后的图片
        self.g_image_show = self.g_image_original[self.g_location_win[1]:self.g_location_win[1] + self.g_window_wh[1], self.g_location_win[0]:self.g_location_win[0] + self.g_window_wh[0]]  # 实际显示的图片

        # 显示图像
        if _show:
            self.thread(self.show_img)

    # 数据对比度强化
    def enhance(self, src):
        if self._hist_enhance:
            # 如果图像多通道，则所有通道合并处理
            if src.ndim == 3:
                # 通道拆分
                b_img, g_img, r_img = cv2.split(src)
                
                # 临时组合运算
                hstack_img = np.hstack((b_img, g_img, r_img))
                dst_hstack_img = cv2.equalizeHist(hstack_img.astype(np.uint8))
                
                # 横向拆分
                dst_hsplit = np.hsplit(dst_hstack_img, 3)
                
                dst = cv2.merge([dst_hsplit[0], dst_hsplit[1], dst_hsplit[2]])
                
                # 储存数据
                self.output_img_dict["hist_enhance_b"] = dst_hsplit[0]
                self.output_img_dict["hist_enhance_g"] = dst_hsplit[1]
                self.output_img_dict["hist_enhance_r"] = dst_hsplit[2]
                self.output_img_dict["hist_enhance"] = dst
            else:
                dst = cv2.equalizeHist(src.astype(np.uint8))       #自动调整图像对比度，把图像变得更清晰
        else:
            dst = src.astype(np.uint8)
        
        # 假彩色
        # 如果图像有多个通道，则自动放弃伪彩色
        if src.ndim == 3:
            return dst
        elif self._fake_color:
            dst = cv2.applyColorMap(cv2.convertScaleAbs(dst), self._fake_color)
            
            # 储存数据
            self.output_img_dict["fake_color"] = dst
        
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
    
    # 线程
    def thread(self,func,*args):#fun是一个函数  args是一组参数对象
        '''将函数打包进线程'''
        t=Thread(target=func,args=args)#target接受函数对象  arg接受参数  线程会把这个参数传递给func这个函数
        t.setDaemon(True)#守护
        t.start()#启动线程
        t.join()
    
    # 显示影像
    def show_img(self):
        # 设置窗口
        cv2.namedWindow(self.g_window_name, cv2.WINDOW_NORMAL)
        # 设置窗口大小，只有当图片大于窗口时才能移动图片
        cv2.resizeWindow(self.g_window_name, self.g_window_wh[0], self.g_window_wh[1])
        cv2.moveWindow(self.g_window_name, 0, 0)  # 设置窗口在电脑屏幕中的位置
        # 鼠标事件的回调函数
        cv2.setMouseCallback(self.g_window_name, self.mouse)
        cv2.waitKey()  # 不可缺少，用于刷新图片，等待鼠标操作
        
        cv2.destroyAllWindows()
    
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
            zoom /= step
            if zoom > self.max_zoom:  # 最多只能放大到1倍
                zoom = self.max_zoom
        else:  # 滚轮下移
            zoom *= step
            if zoom < 0.01:  # 最多只能缩小到0.01倍
                zoom = 0.01
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
                self.g_location_win[0] =self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]
            elif w1 < w2 and h1 >= h2:  # 图片的高度大于窗口的高度，可上下移动
                show_wh = [w1, h2]
                self.g_location_win[1] =self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]
            else:  # 图片的宽高大于窗口宽高，可左右上下移动
                show_wh = [w2, h2]
                self.g_location_win[0] =self.location_win[0] + self.g_location_click[0] - self.g_location_release[0]
                self.g_location_win[1] =self.location_win[1] + self.g_location_click[1] - self.g_location_release[1]
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