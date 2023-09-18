
from tqdm import tqdm, trange
import numpy as np

class GeshaProcessTemplate():
    def __init__(self):
        return
    
    # 二维数组化一维数组处理并还原(部分与二维无关的数据处理（如缠绕），可以这样处理来提高运算速度)
    def flatten_process(self, func, input_data, **param):
        # 2D化1D
        flatten_data = input_data.flatten()
        
        # 处理过程
        flatten_data_result = func(flatten_data, **param)
        
        # 数据恢复2D
        output_data = flatten_data_result.reshape(input_data.shape[0],input_data.shape[1])

        return output_data
    
    # 使用窗口计算的过程都可以归一到这个函数里面
    def win_process(self, core_func, input_mat, win_mask, slide=[1,1], **params):
        # todo 升级slide
        # todo 添加padding功能
        # todo 添加多线程
        # todo 添加多通道
        
        # 初始化数据(统一转化为array，防止报错)
        win_mask = np.array(win_mask)
        input_mat = np.array(input_mat)
        
        # 获取窗口中心坐标
        x_center,y_center = np.where(win_mask == 1)
        x_center = x_center[0]
        y_center = y_center[0]
        
        # 读取窗口长宽,和矩阵长宽
        h_win,w_win = win_mask.shape

        h_mat = input_mat.shape[0]
        w_mat = input_mat.shape[1]
        
        # 计算窗口上下左右边界
        barrier_up = x_center
        barrier_down = h_win - (x_center + 1)
        barrier_left = y_center
        barrier_right = w_win - (y_center + 1)
        
        # 读取步长
        [slide_x, slide_y] = slide
        
        # 计算横向和纵向可能取值范围
        y_range = range(barrier_left, w_mat - barrier_right, slide_y)
        x_range = range(barrier_up, h_mat - barrier_down, slide_x)
        
        # >遍历所有可能窗口
        res_list = []
        center_idxs_list = []
        # 遍历中心点位置
        for i_center in x_range:
            for j_center in y_range:
                # >获取窗口所有数据
                # 初始化
                win_mat = np.zeros_like(win_mask, dtype = np.dtype(np.dtype(input_mat.flatten()[0])))

                # 窗口第一个:[i_center - barrier_up, j_center - barrier_left]
                x_win_first = i_center - barrier_up
                y_win_first = j_center - barrier_left
                # 窗口最后一个:[i_center + barrier_down, j_center + barrier_right]
                x_win_last = i_center + barrier_down
                y_win_last = j_center + barrier_right

                for win_idx_x, i_win in enumerate(range(x_win_first, x_win_last + 1)):
                    for win_idx_y, j_win in enumerate(range(y_win_first, y_win_last + 1)):
                        win_mat[win_idx_x, win_idx_y] = input_mat[i_win, j_win]
                        # print(input_mat[i_win, j_win])
                        
                # 核函数处理
                res_list.append(core_func(win_mat, **params))

                # 记录中心点坐标集
                center_idxs_list.append([i_center, j_center])

        
        return res_list, center_idxs_list

