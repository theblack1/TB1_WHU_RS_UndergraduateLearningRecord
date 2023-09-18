# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import xml.dom.minidom
import cv2
import scipy.constants
import time
from tqdm import trange, tqdm
import math

from concurrent.futures import ThreadPoolExecutor

import scipy.optimize

from imshow_cv import CvImshow
from read_radar import ReadRadar


class SARGeoCorr():
    # 初始化
    def __init__(self, input_dir, src_img_name, head_file_name,
                dem_file_name, dem_pixel_size, dem_max_lat_deg, dem_min_lon_deg,
                Nepoch1, Nepoch2):
        # 读取传入数据
        self.input_dir = input_dir
        self.src_img_name = src_img_name
        self.head_file_name = head_file_name
        self.dem_file_name = dem_file_name
        self.dem_pixel_size = dem_pixel_size
        self.dem_max_lat_deg = dem_max_lat_deg
        self.dem_min_lon_deg = dem_min_lon_deg
        self.Nepoch1 = Nepoch1
        self.Nepoch2 = Nepoch2
        
        # 打开影像
        self.open_img_file()
        # 读头文件
        self.read_head_file()
        
    
    # 打开输入影像和dem参考影像
    def open_img_file(self):
        src_file_path = self.input_dir + "/" + self.src_img_name
        
        if not os.path.exists(src_file_path):
            print(f"ERROR:文件'{src_file_path}'不存在！！！")
            sys.exit(0)
            
        # 使用gdal读取影像数据
        self.radar = ReadRadar(file_name=src_file_path)
        # self.src_img = np.fliplr(self.radar.str_array)
        self.src_img = self.radar.str_array
        
        dem_file_path = self.input_dir + "/" + self.dem_file_name
        
        if not os.path.exists(dem_file_path):
            print(f"ERROR:文件'{dem_file_path}'不存在！！！")
            sys.exit(0)
            
        # 使用gdal读取影像数据
        dem_temp = ReadRadar(file_name=dem_file_path)
        self.dem_img = dem_temp.str_array
    
    # 读取头文件
    def read_head_file(self):
        xml_file_path = self.input_dir + "/" + self.head_file_name + ".xml"
        par_file_path = self.input_dir + "/" + self.head_file_name + ".par"
        tra_file_path = self.input_dir + "/" + self.head_file_name + ".tra"

        # 检验存在性
        if not os.path.exists(xml_file_path):
            print(f"ERROR:文件'{xml_file_path}'不存在！！！")
            sys.exit(0)
        if not os.path.exists(par_file_path):
            print(f"ERROR:文件'{xml_file_path}'不存在！！！")
            sys.exit(0)
        if not os.path.exists(tra_file_path):
            print(f"ERROR:文件'{xml_file_path}'不存在！！！")
            sys.exit(0)
        
        # 逐个文件读取原始内容
        '''
        xml: xml_raw_dom
        tra: tra_raw_arr
        par: par_raw_dict
        '''
        # xml: xml_raw_dom
        self.xml_raw_dom = xml.dom.minidom.parse(xml_file_path)
        
        # tra: tra_raw_arr 轨道数据
        # (时间要和影像时间统一,才能知道每一个时刻卫星位置,然后影像点坐标通过时间来确定,距离向采样率决定每一个像元对应持续时间,PRF决定了方位向,每个像元持续时间是多少)
        with open(tra_file_path, "r") as tra_file:
            '''
            tra_raw_arr:
            第一行：
                0:数据行数
                1:年份
                2:初始天
                3:初始秒
                4:?
                5:?
            后续行：
                0:年份
                1:天
                2:秒
                
                3:X(地心坐标)
                4:Y
                5:Z
                
                6:速度X
                7:速度Y
                8:速度Z
            '''
            tra_lines = tra_file.readlines()
            
            tra_raw_data = []
            # 逐行遍历
            for line in tra_lines:
                line_list = str.split(line)
                # 类型转换
                line_list=list(map(float,line_list))
                # 储存
                tra_raw_data.append(line_list)
                
            # 首行补齐
            num1 = len(tra_raw_data[0])
            num2 = len(tra_raw_data[1])
            tra_raw_data[0] += [0 for index in range(num2 - num1)]
            
            # 储存
            self.tra_raw_arr = np.array(tra_raw_data)
        
        # par: par_raw_dict 参数文件
        with open(par_file_path, "r") as par_file:
            '''
            num_valid_az   	= 27748                   有效行数
            nrows   		= 27748                   一共多少行?
            first_line      = 1                       第一行行号
            az_res   		= 3.300000                方位向分辨率(az方位向,xml里面也有)
            nlooks   		= 1                       视数(单视)
            first_sample   	= 1                       第一列列号
            rng_samp_rate   = 164829192.000000        距离向采样率(rng距离向,是频率)
                可以知道每一个像元对应时间多少,根据这个时间定位列坐标,这个分之一就是每个列持续时间
                每一列持续时间 1/rng_samp_rate 
                
            num_rng_bins		= 18880               一共多少列?         
            bytes_per_line		= 75520               每一行多少字节(列数*4,复数,每个像元4字节,实部2个,虚部2个)
            good_bytes_per_line	= 75520               每一行有效字节数
            PRF			        = 3468.627778         PRF脉冲重复周期,对应方位向像元时间(每一行的时间)
            pulse_dur		    = 5.340000e-07        脉冲持续时间
            near_range		    = 579188.307735       离天线最近的点的距离(单位:m, 计算延迟)
            num_lines		    = 27748               一共多少行??
            SC_clock_start		= 2012166.6812201967  卫星开始时间
            SC_clock_stop		= 2012166.6813127859  卫星结束时间
            clock_start		    = 166.681220196759    时钟开始时间(166.68122.0196759天,第一行,第一列)
            clock_stop			= 166.681312785941    时钟结束时间(166.68131.2785941天,最后一行,最后一列)
                *24*3600 ->秒
                
            orbdir	= D                               卫星环绕：降轨
            lookdir	= R                               视角方向(right)
            radar_wavelength	= 0.0310666           雷达波长(单位:m, 是S波段的)
            chirp_slope	        = -2.24719e+08        线性调频的斜率
            rng_samp_rate		= 164829192.000000    距离向采样率(重复)
            equatorial_radius	= 6378137.000000      赤道半径
            polar_radius		= 6356752.310000      极地半径
            SC_vel              = 7388.655256         卫星的瞬时速度,单位(m/s)
            earth_radius        = 6375909.168687      地球半径
            SC_height           = 510583.743670       卫星轨道高度
            SC_height_start     = 510539.559950       卫星初始轨道高度
            SC_height_end       = 510627.885474       卫星结束轨道高度
            fd1                 = 0.000000            多普勒频率1
            fdd1                = 0                   多普勒频率2
            fddd1               = 0                   多普勒频率3
            '''
            par_lines = par_file.readlines()
            
            # par_raw_data = []
            # 逐行读取
            self.par_raw_dict = {}
            for line in par_lines:
                line_split = line.split()
                # 读键值对
                key = line_split[0]
                val = line_split[2]
                # 类型纠正
                if val.isnumeric():
                    val = int(val)
                else:
                    def _IsFloat(str):
                        # 科学计数法
                        if str.find("+") or str.find("-"):
                            str = str.replace("+","")
                            str = str.replace("-","")
                            
                        se=str.split('e')
                        if len(se)>2:
                            return False
                        else:
                            for sei in se:
                                # 非科学计数法
                                s=sei.split('.')
                                if len(s)>2:
                                    return False
                                else:
                                    for si in s:
                                        if not si.isdigit():
                                            return False
                                    return True
                    
                    if _IsFloat(val):
                        val = float(val)
                        
                self.par_raw_dict[key] = val
            
        # 具体读取数据
        def _read_xml(tag_name_list, idx_list =[0], dom = self.xml_raw_dom):
            cur_tag = dom
            for [tag_name, idx] in zip(tag_name_list, idx_list):
                cur_tag = cur_tag.getElementsByTagName(tag_name)[idx]
                
            return cur_tag.firstChild.data
        
        # 1.1.轨道控制点坐标
        # 读取控制点数目
        self.Nctrl_points = int(self.tra_raw_arr[0,0])
        # 在对应行列上裁剪
        self.ctrl_points_arr = self.tra_raw_arr[1:self.Nctrl_points+1, 3:6]
        
        # 1.2.1.轨道参考第一点成像时间，单位为秒
        self.first_ctrl_point_image_t = self.tra_raw_arr[1,2]
        
        # 1.2.2.轨道点间时间间隔，单位为秒；
        self.ctrl_point_gap_time = self.tra_raw_arr[2,2] - self.tra_raw_arr[1,2]
        
        # 中心点坐标
        self.scence_center_coord = np.array([
            float(_read_xml(["sceneCenterCoord","lat"], [0,0])),
            float(_read_xml(["sceneCenterCoord","lon"], [0,0]))])
        
        # 1.3.SAR影像四角点经纬度，单位为度；可根据此范围提取所需DEM块；
        scene_corner_coord_list = []
        for i in range(4):
            scene_corner_coord = [
                float(_read_xml(["sceneCornerCoord","lat"], [i,0])),
                float(_read_xml(["sceneCornerCoord","lon"], [i,0]))
            ]
            scene_corner_coord_list.append(scene_corner_coord)
        self.scene_corner_coord_arr = np.array(scene_corner_coord_list)

        # 1.4.1.第一行成像时间，单位为秒
        first_row_image_t_by_day = self.par_raw_dict["clock_start"]
        self.first_row_image_t = 24*3600*(first_row_image_t_by_day % 1)
        
        # 1.4.2.方位位向（行）采样重复频率；单位HZ；
        self.row_samp_rate = self.par_raw_dict["PRF"]
        
        # 1.5.距离向（列）采样频率，单位HZ；
        self.col_samp_rate = self.par_raw_dict["rng_samp_rate"]
        
        # 1.6.1投影坐标系椭球体长半轴长
        self.proj_a_axis = self.par_raw_dict["equatorial_radius"]
        # 1.6.2投影坐标系短半轴长；
        self.proj_b_axis = self.par_raw_dict["polar_radius"]
        
        # 其他数据
        # 1.7 控制点成像时间
        self.ctrl_points_t_arr = self.tra_raw_arr[1:self.Nctrl_points+1, 2]
        # print(self.ctrl_points_t_arr)
        
        # 1.8 最近点距离
        self.near_range = self.par_raw_dict["near_range"]
        
    # 展示影像
    def show_img(self, img_name, _save = False):
        # 默认参数
        _fake_color = cv2.COLORMAP_PLASMA
        _hist_enhance = True
        
        # 选择打开文件
        if img_name == "src":
            win_name = "src strength img"
            input_img = self.src_img
            if _save:
                save_name = img_name + "(" + self.src_img_name + ")"
        elif img_name == "dem":
            win_name = "dem img"
            input_img = self.dem_img
            if _save:
                save_name = img_name + "(" + self.dem_file_name + ")"
            
            _fake_color = cv2.COLORMAP_JET
            _hist_enhance = False
        elif img_name == "result":
            win_name = "geometric correction img"
            input_img = self.geocorr_img
            if _save:
                save_name = "geometric_correction_img"
        elif img_name == "pick_up":
            win_name = "pick_up img"
            input_img = self.pick_up_img
            if _save:
                save_name = "pick_up_img"
            
            _fake_color = None
            _hist_enhance = False
        else:
            win_name = ""
            print(f"Warning: 无法打开影像'{img_name}',请选择('src'/'dem'/'result')")
        
        if len(win_name):
            print(f"正在显示影像{win_name}")
            CvImshow(img_array=input_img, win_name=win_name + " (Zooming by mouse wheel)", _stretch = True, save_name = save_name,
                    _fake_color = _fake_color, _hist_enhance = _hist_enhance, step = 0.5, max_zoom=1)# , save_name="src_img"
    
    # 通过轨道数据进行坐标计算(地心坐标系)
    def track_cal_arr(self, t_arrvec):
        # 时间归零
        t_arrvec = t_arrvec - self.first_ctrl_point_image_t
        
        # 所有点数
        N = len(t_arrvec)
        
        pos_t_mat=np.mat(np.c_[ np.ones(N),    t_arrvec,  t_arrvec**2,     t_arrvec**3])
        vel_t_mat=np.mat(np.c_[np.zeros(N),  np.ones(N),   2*t_arrvec, 3*(t_arrvec**2)])
        acc_t_mat=np.mat(np.c_[np.zeros(N), np.zeros(N), 2*np.ones(N),      6*t_arrvec])
        # 卫星位置，速度，加速度
        pos_arr = np.array(pos_t_mat * self.tra_pra_mat)
        vel_arr = np.array(vel_t_mat * self.tra_pra_mat)
        acc_arr = np.array(acc_t_mat * self.tra_pra_mat)
        
        return pos_arr, vel_arr, acc_arr
    
    # 最小二乘法拟合轨道参数
    def track_pra_fit(self):
        # 读取所需数据
        N = self.Nctrl_points
        t_arrvec=self.ctrl_points_t_arr - self.first_ctrl_point_image_t
        X0_vec=np.mat(self.ctrl_points_arr[:,0]).T
        Y0_vec=np.mat(self.ctrl_points_arr[:,1]).T
        Z0_vec=np.mat(self.ctrl_points_arr[:,2]).T
        
        # 拼接形成时间矩阵
        t_mat = np.mat(np.c_[np.ones(N), t_arrvec, t_arrvec**2, t_arrvec**3])
        
        # 计算拟合轨道参数
        process_mat=(t_mat.T*t_mat).I*t_mat.T
        Ax=process_mat*X0_vec
        Ay=process_mat*Y0_vec
        Az=process_mat*Z0_vec
        
        self.tra_pra_mat = np.concatenate((Ax, Ay, Az), axis=1)
        
        # 计算误差函数
        def _error_func():
            cal_coord_arr, _, _= self.track_cal_arr(self.ctrl_points_t_arr)
            
            error = np.average(np.sqrt(np.sum((cal_coord_arr - self.ctrl_points_arr)**2, axis = 1)))
            
            return error
        
        # 结果误差
        print()
        print(f"最小二乘化处理后拟合误差:\n{_error_func()}m")
    
    # 几何校正核心步骤
    def _geo_correction_core(self):
        # 打印启动信息
        print()
        print(f"INFO:1.正在运行'几何校正'")
        
        # 设定超参数
        K_TH = 1.0e-15 # 迭代改正数阈值
        THETA_TH = 40  # 迭代次数上限
        
        # 设置初始值
        init_t = self.first_row_image_t
        
        # 初始化影像
        geocorr_img = np.zeros_like(self.dem_img)
        
        # 读取长宽
        h = int(self.dem_img.shape[0])
        w = int(self.dem_img.shape[1])
        
        # 输出
        info_str = "2.读取参数与获取坐标向量"
        print (f'\nINFO:{info_str}')
        # 读取全dem影像B,L,H向量        
        B_list = []
        L_list = []
        H_list = []
        dem_row_list = []
        dem_col_list = []
        for lat_idx in trange(len(range(h))):
            for lon_idx in range(len(range(w))):
                # 记录行列号
                dem_row_list.append(lat_idx)
                dem_col_list.append(lon_idx)
                
                # 计算经纬度和记录高度
                B_list.append(self.dem_max_lat_deg - (lat_idx+0.5)*self.dem_pixel_size)
                L_list.append(self.dem_min_lon_deg + (lon_idx+0.5)*self.dem_pixel_size)
                H_list.append(self.dem_img[lat_idx, lon_idx])
                
        B_arr = np.array(B_list)
        L_arr = np.array(L_list)
        H_arr = np.array(H_list)
        dem_row_arr = np.array(dem_row_list)
        dem_col_arr = np.array(dem_col_list)
        
        # 经纬度转地心坐标系
        # 读取长短半轴
        a_axis = self.proj_a_axis
        b_axis = self.proj_b_axis
        
        # 角度转弧度
        B_rad_arr = np.deg2rad(B_arr)
        L_rad_arr = np.deg2rad(L_arr)
        
        # 计算地球相关参数
        e2=(a_axis**2-b_axis**2)/(a_axis**2)
        W = np.sqrt(1-e2*np.sin(B_rad_arr)*np.sin(B_rad_arr))
        N = a_axis / W
        
        # 计算地心坐标系 
        X_arr = (N + H_arr) * np.cos(B_rad_arr) * np.cos(L_rad_arr)
        Y_arr = (N + H_arr) * np.cos(B_rad_arr) * np.sin(L_rad_arr)
        Z_arr = (N * (1 - e2) + H_arr) * np.sin(B_rad_arr)
        
        # 拼接数据
        XYZ_mat = np.mat(np.c_[X_arr,Y_arr,Z_arr])
        
        
        # 输出
        info_str = "3.迭代获取方位向时间"
        print (f'\nINFO:{info_str}')
        def _list_split(items, n):
            return [items[i:i+n] for i in range(0, len(items), n)]
        
        # 迭代真实成像时间
        az_t_arr = np.ones_like(B_arr) * init_t
        det_t_arr = np.zeros_like(B_arr)
        # 振荡检测
        max_last_last_det_t = -1
        max_last_det_t = -1
        for iter in range(THETA_TH):
            if iter == 0:
                print(f"第[{iter+1}/{THETA_TH}]次迭代")
            else:
                print(f"第[{iter+1}/{THETA_TH}]次迭代，当前时间变化值均值{np.average(abs(det_t_arr))};最大变化值{np.max(abs(det_t_arr))}; 最小变化值{np.min(abs(det_t_arr))}")
            
            # 计算时间残差
            splited_range1 = _list_split(items = range(len(B_arr)), n = int(len(B_arr)/Nepoch1) + 1)
            for batch1 in tqdm(splited_range1):
                batch_begin = batch1[0]
                batch_end = batch1[-1] + 1
                # 读取轨道数据
                pos_arr, vel_arr, acc_arr = self.track_cal_arr(az_t_arr[batch_begin:batch_end])
                # 矩阵化
                pos_mat = np.mat(pos_arr)
                vel_mat = np.mat(vel_arr)
                acc_mat = np.mat(acc_arr)
                det_t_arr[batch_begin:batch_end] = np.diagonal(((pos_mat - XYZ_mat[batch_begin:batch_end,:])*(vel_mat.T)))/(np.diagonal((pos_mat - XYZ_mat[batch_begin:batch_end,:])*(acc_mat.T)) + np.diagonal(vel_mat*(vel_mat.T)))
                # 更新时间
                az_t_arr[batch_begin:batch_end] -= det_t_arr[batch_begin:batch_end]

            # 迭代满足条件时退出
            if iter > 1:
                max_last_last_det_t = max_last_det_t
            if iter > 0:
                max_last_det_t = max_det_t
            max_det_t = np.max(abs(det_t_arr))
            if max_det_t < K_TH:
                print(f"INFO:达到收敛阈值,当前最大时间改变量{max_det_t} < 阈值{K_TH}\n退出迭代")
            # 达到阈值
                break
            if iter > 1:
                if abs(max_last_last_det_t - max_det_t) < K_TH * 1e-2:
                    print(f"INFO:改变值发生振荡,当前最大时间改变量{max_det_t}\n退出迭代")
                # 发生振荡
                    break
        
        # 阶段输出
        info_str = "4.获取行列号"
        print (f'\nINFO:{info_str}')
        
        # 通过最近邻法确定雷达影像对应行列号(radar_row_idx, radar_col_idx)
        geocorr_row_arr = (0.5 + (az_t_arr - self.first_row_image_t)*self.row_samp_rate).astype(int)
        
        # 斜距计算
        slope_dist_arr = np.zeros_like(B_arr)
        splited_range2 = _list_split(items = range(len(B_arr)), n = int(len(B_arr)/Nepoch2) + 1)
        
        for batch in tqdm(splited_range2):
            # 读取索引
            batch_begin = batch[0]
            batch_end = batch[-1] + 1
            # 读取轨道数据
            pos_arr, vel_arr, acc_arr = self.track_cal_arr(az_t_arr[batch_begin:batch_end])
            # 矩阵化
            pos_mat = np.mat(pos_arr)
            vel_mat = np.mat(vel_arr)
            acc_mat = np.mat(acc_arr)
            # 计算斜距
            slope_dist_arr[batch_begin:batch_end] = np.sqrt(np.diagonal((pos_mat - XYZ_mat[batch_begin:batch_end])*(pos_mat - XYZ_mat[batch_begin:batch_end]).T))
            
        # 距离向时间
        rng_t_arr = 2*(slope_dist_arr - self.near_range)/scipy.constants.speed_of_light
        # 对应列号
        geocorr_col_arr = (0.5 + rng_t_arr*self.col_samp_rate).astype(int)
        
        # 阶段输出
        info_str = "5.几何纠正影像采样"
        print (f'\nINFO:{info_str}')
        # 筛选合适的行列号
        rule_1 = ((geocorr_row_arr >= 0) & (geocorr_row_arr < self.src_img.shape[0]))
        rule_2 = ((geocorr_col_arr >= 0) & (geocorr_col_arr < self.src_img.shape[1]))
        inside_idx = (rule_1) & (rule_2) 
        
        dem_i_list = dem_row_arr[inside_idx]
        dem_j_list = dem_col_arr[inside_idx]
        src_i_list = geocorr_row_arr[inside_idx]
        src_j_list = geocorr_col_arr[inside_idx]

        pick_up_img = np.zeros_like(self.src_img)
        # 影像采样
        for idx in trange(len(dem_i_list)):
            dem_i = dem_i_list[idx]
            dem_j = dem_j_list[idx]
            src_i = src_i_list[idx]
            src_j = src_j_list[idx]
            
            geocorr_img[dem_i, dem_j] = self.src_img[src_i, src_j]
            pick_up_img[src_i, src_j] = 255
        
        self.geocorr_img = geocorr_img
        self.pick_up_img = pick_up_img
    
    # 几何校正
    def geo_correction(self):
        # 开始计时
        begin_time = time.time()
        
        # 创建包含2个线程的线程池
        pool = ThreadPoolExecutor(max_workers=2)
        
        # 向线程池提交任务
        task = pool.submit(self._geo_correction_core)
        
        # 传出数据
        # self.geocorr_img = task.result()
        
        # 关闭线程池
        pool.shutdown()
        
        # 程序结束时间
        end_time = time.time()
        # 运行时间run_time。round()函数取整
        run_time = round(end_time-begin_time)
        # 计算时分秒
        hour = run_time//3600
        minute = (run_time-3600*hour)//60
        second = run_time-3600*hour-60*minute
        # 输出
        print (f'运行几何校正总共耗时:\n{hour}小时{minute}分钟{second}秒')
        
# 主函数
if __name__ == '__main__':
    # 文件路径 
    src_dir = r"data/input_data" # 输入路径
    src_img_name = "image.tif"  # 强度影像名称
    head_file_name = "TSX20120615" # 影像头文件公共名称
    
    dem_file_name = "dem_clip_resample_img1E-4.tif" # dem名称
    dem_pixel_size = 1e-4 # dem重采样像元大小(单位：度)
    # 获取dem最小经度和最大纬度
    dem_max_lat_deg = 19.6620833333
    dem_min_lon_deg = -155.424027778
    Nepoch1 = 1000000
    Nepoch2 = 500000
    
    # 打开影像
    SAR_geocorr_process = SARGeoCorr(src_dir, src_img_name, head_file_name,
                                    dem_file_name, dem_pixel_size, dem_max_lat_deg, dem_min_lon_deg,
                                    Nepoch1, Nepoch2)
    # SAR_geocorr_process.radar.show_info()
    
    # 展示影像
    # SAR_geocorr_process.show_img("src", _save = True)
    
    # 一.读取头文件
    print("##################### 一.处理参数提取 #####################")
    
    # print("tra文件数据读取")
    # print(SAR_geocorr_process.tra_raw_arr)
    print()
    print("1.轨道控制点坐标X,Y,Z")
    print(SAR_geocorr_process.ctrl_points_arr)
    
    print()
    print(f"2.轨道参考第一点成像时间:{SAR_geocorr_process.first_ctrl_point_image_t}s;轨道点间时间间隔:{SAR_geocorr_process.ctrl_point_gap_time}s")
    
    print()
    print(f"3.四个角点经纬度\n{SAR_geocorr_process.scene_corner_coord_arr}")
    
    print()
    print(f"4.第一行成像时间:{SAR_geocorr_process.first_row_image_t}s; 方位向采样重复频率：{SAR_geocorr_process.row_samp_rate}Hz")
    
    print()
    print(f"5.距离向采样频率:{SAR_geocorr_process.col_samp_rate}Hz")
    
    print()
    print(f"6.投影坐标系椭球体长半轴长:{SAR_geocorr_process.proj_a_axis}m; 短半轴长：{SAR_geocorr_process.proj_b_axis}m")
    print()
    
    # 二.轨道信息提取
    print("##################### 二.轨道信息提取 #####################")
    
    # 执行轨道拟合
    SAR_geocorr_process.track_pra_fit()
    
    print(f"\n最小二乘得到轨道参数:\n{SAR_geocorr_process.tra_pra_mat}")
    print()
    
    # 三.图像数据准备
    print("##################### 三.图像数据准备 #####################")
    # # 展示雷达影像
    SAR_geocorr_process.show_img("src", _save = True)
    print()
    
    # 四.DEM准备
    print("##################### 四.DEM准备 #####################")
    # 展示DEM影像
    SAR_geocorr_process.show_img("dem", _save = True)
    print()
    
    # 五.影像坐标解求
    print("##################### 五.影像坐标解求 #####################")
    # 执行几何校正
    SAR_geocorr_process.geo_correction()
    # 展示影像
    SAR_geocorr_process.show_img("result", _save = True)