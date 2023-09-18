import numpy as np
import numpy.matlib as matlib
import numba as nb
import os
import struct
import cv2
from tqdm import trange
import sys
from scipy.ndimage import convolve

from imshow_cv import CvImshow

C3_NAME_LIST = [
        "C11", "C12_real", "C12_imag",
        "C13_real", "C13_imag", "C22",
        "C23_real", "C23_imag", "C33"]

class PolSAR():
    # 初始化函数
    def __init__(self, c3_read_dir):
        # 常量
        self.C3_FILE_NAME_LIST = [
        "C11.bin", "C12_real.bin", "C12_imag.bin",
        "C13_real.bin", "C13_imag.bin", "C22.bin",
        "C23_real.bin", "C23_imag.bin", "C33.bin"]
        
        # 基础信息读取
        self.read_dir = c3_read_dir
        
        config_dict = self.read_config()
        self.Nrow = int(config_dict['Nrow'])
        self.Ncol = int(config_dict['Ncol'])
        self.PolarCase = config_dict['PolarCase']
        self.PolarType = config_dict['PolarType']
        
        # 读取影像
        self.C3_array = self.open_C3_file()
        
        # 待滤波影像
        self.flitered_C3_array = np.zeros_like(self.C3_array)
    
    # 读取参数
    def read_config(self, config_name_list = ['Nrow', 'Ncol', 'PolarCase', 'PolarType']):
        config_file_name = self.read_dir + '/' + 'config.txt'
        if not os.path.exists(config_file_name):
            print(f'WARNING: config文件: "{config_file_name}" 不存在！！！')
        
        # 打开文件
        with open(config_file_name, 'r') as config_file:
            config_txt_list=[]
            for line in config_file:
                config_txt_list.append(line.strip())
            
            # 读取配置数据
            config_dict = {}
            for config_name in config_name_list:
                # 配置数据位置在名称的下一行
                config_idx = 1 + config_txt_list.index(config_name)
                config = config_txt_list[config_idx]
            
                # 打包
                config_dict[config_name] = config
        
        return config_dict

    # 打开C3文件为np.array
    def open_C3_file(self):
        # 检查读取路径是否存在
        if not os.path.exists(self.read_dir):
            print(f"ERROR:路径：'{self.read_dir}'不存在！！！")
            exit()

        Npolar = 9
        Ntotal_pixels = self.Ncol*self.Nrow

        self.C3_array = np.array([None] * Npolar)

        for Np in range(Npolar):
            # 拼接输入文件路径
            in_file_name = self.read_dir + '/' + self.C3_FILE_NAME_LIST[Np]
            # 读取二进制数据
            with open(in_file_name, 'rb') as raw_data_file:
                # 检验是否正常打开
                if not raw_data_file:
                    print(f"Could not open input file : {in_file_name}")
                    continue
                
                # 读取二进制，转换为浮点数，浮点数大小为4
                raw_data = struct.unpack('f' * Ntotal_pixels, raw_data_file.read(4*Ntotal_pixels))
                data_array =  np.asarray(raw_data).reshape(-1,self.Ncol)
                self.C3_array[Np] = data_array
        
        return self.C3_array

    # 保存到本地为bin文件
    def write_C3_file(self, input_mat, save_name, file_dir = "data/PolSAR_write_default", description = "Gesha write PolSAR img to ENVI"):
        # 读取尺寸
        if input_mat.ndim == 3:
            rows,cols, bands=input_mat.shape
        else:
            rows,cols=input_mat.shape
            bands = 1
        
        input_mat = np.array(input_mat)
    
        file_path = file_dir + '/' + save_name + ".bin"
        # 递归创建文件夹
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        
        with open(file_path, 'wb') as f:
            for row in input_mat:
                for value in row:
                    f.write(struct.pack('f', value))
                    
        # 写头文件
        with open(file_path + ".hdr", "w") as f:
            file_name = save_name + ".bin"
            hdr_passage = f"ENVI\n"\
                f"description = {{\n"\
                f"{description}}}\n"\
                f"samples = {cols}\n"\
                f"lines   = {rows}\n"\
                f"bands   = {bands}\n"\
                f"header offset = 0\n"\
                f"file type = ENVI Standard\n"\
                f"data type = 4\n"\
                f"interleave = bsq\n"\
                f"sensor type = Unknown\n"\
                f"byte order = 0\n"\
                f"band names = {{\n"\
                f"{file_name}}}"
                
            f.write(hdr_passage)
        
        # 另一个头文件
        if not os.path.exists(file_dir + '/' + "config.txt"):
            with open(file_dir + '/' + "config.txt", "w") as f:
                txt_passage = "Nrow\
                            900\
                            ---------\
                            Ncol\
                            1024\
                            ---------\
                            PolarCase\
                            monostatic\
                            ---------\
                            PolarType\
                            full"
                
                f.write(txt_passage)
    
    # 显示某个Cxx的影像
    def show_single_band(self, band_name, _filtered, save_name = "", _fake_color = None, _hist_enhance = None):
        # 获取序号
        band_idx = self.C3_FILE_NAME_LIST.index(band_name + '.bin')
        # 获取影像
        if _filtered:
            selected_single_band_img = self.flitered_C3_array[band_idx]
        else:
            selected_single_band_img = self.C3_array[band_idx]
            
        # 显示影像
        CvImshow(img_array=selected_single_band_img , win_name=band_name, _stretch = True,
                _hist_enhance = _hist_enhance, save_name= save_name, _fake_color = _fake_color)

    # 假彩色合成
    def merge_img(self, _type, save_name = "", _hist_enhance = True, _each_stretch = False):
        # 选择滤波后或者前影像展示
        if _type == 2:
            # 显示滤波影像
            b_img = self.flitered_C3_array[0]
            g_img = self.flitered_C3_array[5]
            r_img = self.flitered_C3_array[8]
            win_name = "boxcar filtered merged img"
            # _each_stretch = True
        elif _type == 1:
            # 原始图像
            b_img = self.C3_array[0]
            g_img = self.C3_array[5]
            r_img = self.C3_array[8]
            win_name = "original merged img"
            # _each_stretch = True
        elif _type == 3:
            # 分解图像
            b_img = self.ODD_mat
            r_img = self.DBL_mat
            g_img = self.VOL_mat
            win_name = "Freeman merged img"
            # _hist_enhance = False

        merged_img = cv2.merge([b_img, g_img, r_img])
        
        # 显示影像
        cv_imshow = CvImshow(img_array=merged_img , win_name= win_name, _stretch = True,
                _hist_enhance = _hist_enhance, save_name= save_name, _each_stretch = _each_stretch)
        
        return cv_imshow.output_img_dict
    
    # boxcar滤波
    def filter(self, win_size = 3, _save = False):
        # 检验窗口大小是否合法
        if win_size%2 != 1:
            print('Warnning: Size of the Window Should be Odd Number!')
            return []
        
        # 核心函数
        def _boxfilter_core(input_img,win_size):
            # 自定义滤波器大小
            kernel_size = (win_size, win_size)  
            kernel = np.ones(kernel_size) / (win_size**2)  # 计算滤波器权重
            # 进行均值滤波
            filtered_data = convolve(input_img, kernel, mode='constant')
            
            return filtered_data
        
        # 执行滤波
        for img_idx in range(len(self.C3_array)):
            # self.flitered_C3_array[img_idx] = cv2.boxFilter(self.C3_array[img_idx], -1, (win_size,win_size))
            self.flitered_C3_array[img_idx] = _boxfilter_core(self.C3_array[img_idx], win_size)
            # 如果打算保存，则同步写到本地
            if _save:
                file_dir = r"data/filtered_C3"
                save_name = "filtered_" + C3_NAME_LIST[img_idx]
                self.write_C3_file(self.flitered_C3_array[img_idx], save_name, file_dir)
        
        return self.flitered_C3_array
    
    # Freeman三分量分解(C矩阵)
    def freeman(self, _save = False):
        C11 = self.flitered_C3_array[0]
        C22 = self.flitered_C3_array[5]
        C33 = self.flitered_C3_array[8]
        C13_re = self.flitered_C3_array[3]
        C13_im = self.flitered_C3_array[4]
        
        self.fv_mat=3 * C22/2
        C11=C11 - self.fv_mat
        C33=C33 - self.fv_mat
        C13_re = C13_re - self.fv_mat/3
        self.fd_mat=np.zeros_like(self.fv_mat)
        self.fs_mat=np.zeros_like(self.fv_mat)
        self.alpha_mat=np.zeros_like(self.fv_mat)
        self.btea_mat=np.zeros_like(self.fv_mat)
        
        eps = sys.float_info.epsilon
        
        m_1=((C11 < eps) | (C33 < eps))
        m_2=~m_1
        m_21=((C13_re**2 + C13_im**2) > C11 * C33)
        m_22=(C13_re>=0)
        m_23=(C13_re<0)
        
        self.fv_mat[m_1]=3*(C11[m_1]+C22[m_1]+C33[m_1]+2*self.fv_mat[m_1])/8
        self.fd_mat[m_1]=0
        self.fs_mat[m_1]=0
        
        rtemp=C13_re**2 + C13_im**2
        C13_re[m_2 & m_21]=C13_re[m_2 & m_21]*np.sqrt(C11[m_2 & m_21]*C33[m_2 & m_21]/rtemp[m_2 & m_21])
        C13_im[m_2 & m_21]=C13_im[m_2 & m_21]*np.sqrt(C11[m_2 & m_21]*C33[m_2 & m_21]/rtemp[m_2 & m_21])
        
        self.alpha_mat[m_2 & m_22]=-1
        self.fd_mat[m_2 & m_22]=(C11[m_2 & m_22]*C33[m_2 & m_22]-C13_re[m_2 & m_22]**2-C13_im[m_2 & m_22]**2)/(C11[m_2 & m_22]+C33[m_2 & m_22]+2*C13_re[m_2 & m_22])
        self.fs_mat[m_2 & m_22]=C33[m_2 & m_22]-self.fd_mat[m_2 & m_22]
        self.btea_mat[m_2 & m_22]=np.sqrt((self.fd_mat[m_2 & m_22]+C13_re[m_2 & m_22])**2+C13_im[m_2 & m_22]**2)/self.fs_mat[m_2 & m_22]
        
        self.btea_mat[m_2 & m_23]=1
        self.fs_mat[m_2 & m_23]=(C11[m_2 & m_23]*C33[m_2 & m_23]-C13_re[m_2 & m_23]**2-C13_im[m_2 & m_23]**2)/(C11[m_2 & m_23]+C33[m_2 & m_23]-2*C13_re[m_2 & m_23])
        self.fd_mat[m_2 & m_23]=C33[m_2 & m_23]-self.fs_mat[m_2 & m_23]
        self.alpha_mat[m_2 & m_23]=np.sqrt((self.fs_mat[m_2 & m_23]-C13_re[m_2 & m_23])**2+C13_im[m_2 & m_23]**2)/self.fd_mat[m_2 & m_23]
        
        ODD=self.fs_mat*(1+self.btea_mat*self.btea_mat)
        DBL=self.fd_mat*(1+self.alpha_mat*self.alpha_mat)
        VOL=8*self.fv_mat/3
        
        self.ODD_mat = ODD
        self.DBL_mat = DBL
        self.VOL_mat = VOL
        
        # 如果准备保存
        if _save:
                file_dir = r"data/FD3"
                self.write_C3_file(self.ODD_mat, save_name = "ODD", file_dir = file_dir)
                self.write_C3_file(self.DBL_mat, save_name = "DBL", file_dir = file_dir)
                self.write_C3_file(self.VOL_mat, save_name = "VOL", file_dir = file_dir)

        
        return self.ODD_mat, self.DBL_mat, self.VOL_mat


    
if __name__ == '__main__':
    # 读取C3文件
    src_C3_dir = r'data/C3'
    pol_SAR = PolSAR(src_C3_dir)
    
    # 展示C11
    band_name = C3_NAME_LIST[0]
    color_map = cv2.COLORMAP_CIVIDIS
    _each_stretch = False
    _hist_enhance = True
    pol_SAR.show_single_band(band_name, _filtered = False , _fake_color = color_map, _hist_enhance = True, save_name=f"{band_name}_origin_fakecolor")
    
    # boxcar滤波并展示C11
    pol_SAR.filter(win_size=5, _save = True)
    pol_SAR.show_single_band(band_name, _filtered = True ,_fake_color = color_map, _hist_enhance = True, save_name=f"{band_name}_boxcarflitered_fakecolor")
    
    # 假彩色合成
    merged_origin_img = pol_SAR.merge_img(_type = 1, save_name = "origin_merged_img")
    merged_filtered_img = pol_SAR.merge_img(_type = 2, save_name = "filtered_merged_img", _each_stretch = _each_stretch, _hist_enhance = _hist_enhance)
    
    # Freeman三极化分解
    pol_SAR.freeman(_save = True)
    merge_img_dict = merged_freeman_img = pol_SAR.merge_img(_type = 3, save_name = "freeman_merged_img", _each_stretch = _each_stretch, _hist_enhance = _hist_enhance)
    
    # 展示三极化分解的伪彩色图像
    ODD_mat = merge_img_dict['hist_enhance_b']
    DBL_mat = merge_img_dict['hist_enhance_r']
    VOL_mat = merge_img_dict['hist_enhance_g']
    CvImshow(img_array=ODD_mat , win_name="ODD img", _stretch = False, _hist_enhance = False, save_name= "ODD img", _fake_color = color_map)
    CvImshow(img_array=DBL_mat , win_name="DBL img", _stretch = False, _hist_enhance = False, save_name= "DBL img", _fake_color = color_map)
    CvImshow(img_array=VOL_mat , win_name="VOL img", _stretch = False, _hist_enhance = False, save_name= "VOL img", _fake_color = color_map)