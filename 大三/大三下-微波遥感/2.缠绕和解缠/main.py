import numpy as np
import math
from scipy import stats
# from tqdm import tqdm, trange

from gesha_img_tool import GeshaImgTool
from gesha_process_template import GeshaProcessTemplate

G_IMG = GeshaImgTool()

from matplotlib import cm

# 输入的真实相位
SRC = np.array([
    [0.0,0.0,0.3,0.0,0.0,0.3,0.0,0.0],
    [0.0,0.3,0.6,0.3,0.3,0.6,0.3,0.0],
    [0.0,0.0,0.9,0.6,0.6,0.9,0.0,0.0],
    [0.0,0.0,1.2,0.9,0.9,1.2,0.0,0.0],
    [0.0,0.0,1.2,0.9,0.9,1.2,0.0,0.0],
    [0.0,0.0,0.9,0.6,0.6,0.9,0.0,0.0],
    [0.0,0.3,0.6,0.3,0.3,0.6,0.3,0.0],
    [0.0,0.0,0.3,0.0,0.0,0.3,0.0,0.0]])

# 用于实现实习二主要步骤的类
class WrapData():
    # 初始化
    def __init__(self, origin_data = []):
        # 初始化处理工具
        self.g_process = GeshaProcessTemplate()
        
        # 存放真实相位数据（本次实习已经提供）
        self.origin_data = origin_data
        
        # 初始化缠绕数据
        self.wrap_data = np.zeros_like(origin_data)
        
        # 存放解缠数据
        self.unwrap_data = np.zeros_like(origin_data)
        
        # 计算缠绕数据
        self.wrap()
        
        # # 计算解缠数据
        # self.unwrap()

    # 缠绕数据
    def wrap(self):
        # 定义处理函数，准备代入flatten_process中处理
        def _warp_core(input_data):
            output_data = input_data.copy()
            # 获得[-0.5, 0.5)范围外的数据和其索引,然后按照索引将其改正
            # 为了提高速度，可以先批量处理>=0.5的，后批量处理<-0.5的
            
            # 1.获得>=0.5的元素数目
            over_num = len(np.where(output_data >= 0.5)[0])
            # 循环，反复减少>0.5的，直到没有超过0.5的，如果一开始就没有，则循环不会执行
            while over_num:
                # 执行-2pi（-1）操作
                output_data = np.where(output_data >= 0.5, output_data - 1, output_data)  
                # 重新计算，观测是否有残留
                over_num = len(np.where(output_data >= 0.5)[0])
            
            # 2.获得>=0.5的元素数目
            over_num = len(np.where(output_data < -0.5)[0])
            # 循环，反复增加<-0.5的，直到没有小于-0.5的，如果一开始就没有，则循环不会执行
            while over_num:
                # 执行-2pi（-1）操作
                output_data = np.where(output_data < -0.5, output_data + 1, output_data)
                # 重新计算，观测是否有残留
                over_num = len(np.where(output_data < -0.5)[0])
            
            return output_data
        
        self.wrap_data = self.g_process.flatten_process(func=_warp_core, input_data=self.origin_data)

        return self.wrap_data
        
    # 解缠数据
    def unwrap(self, _radom_choose = False):
        # 初始化
        self.unwrap_data = self.wrap_data.copy()
        # 设置辅助变量
        _status_mat = np.zeros_like(self.origin_data)
        # _status,和源数据一样大小的数组
        '''
            status = 0: 未解缠数据
            status = 1: 已解缠数据
            status = 2: 邻接表上的数据
            status = 3: 支切线上的点(charge = 0)
            status = 4: 正残差点
            status = 5: 负残差点
        '''
        
        # Itoh基础单元
        def _Itoh(f_gap):
            # 判断差值，提供修正
            if f_gap >= 0.5:
                Itoh_fix = - 1
            elif f_gap < -0.5:
                Itoh_fix = 1
            else:
                Itoh_fix = 0
                
            return Itoh_fix
        
        # 一.残差点检测 status 0->4/5
        def _RMR_detect_core(win_mat):
            # 按照顺时针读取窗口中四个元素
            p1 = win_mat[0, 0]
            p2 = win_mat[0, 1]
            p3 = win_mat[1, 1]
            p4 = win_mat[1, 0]
            
            # 环路计算
            det1 = _Itoh(p2 - p1)
            det2 = _Itoh(p3 - p2)
            det3 = _Itoh(p4 - p3)
            det4 = _Itoh(p1 - p4)
            
            # 环路积分
            RMR = det1 + det2 + det3 + det4
            
            # print(win_mat)
            
            if RMR == 0:
                return 0
            elif RMR == 1:
                return 4
            else:
                return 5
        
        # 设置滑动窗口计算
        RMR_detect_mask = np.array([
            [1, 0],
            [0, 0]
        ])
        res_list, res_idxs_list = self.g_process.win_process(core_func=_RMR_detect_core, input_mat=self.wrap_data, win_mask=RMR_detect_mask)
        
        RMR_idxs_list = []
        # 生成残差图
        for idx in range(len(res_list)):
            # 解析数据
            status = res_list[idx]
            [x_RMR, y_RMR] = res_idxs_list[idx]
            
            # 写入数据
            _status_mat[x_RMR, y_RMR] = status
            
            # 记录RMR位置
            if status > 3:
                RMR_idxs_list.append([x_RMR, y_RMR])
        
        print('残差计算结果')
        G_IMG.mat_to_heatmap(_status_mat, file_name="RMR_mat")
        
        #  二.残差点连接
        #  2.1.选取某个残差点（status == 4/5）
        xc_list, yc_list = np.where(_status_mat > 3)
        # 获得抽取的点
        xc0 = xc_list[-1]
        yc0 = yc_list[-1]
        
        # 记录charge = +/- 1
        def _status_to_charge(status):
            # 如果是4,正残差 +1,如果是5,负残差-1,所以公式：charge = -2*_status[xc0, yc0]+9
            charge = -2 * status + 9
            return charge
        
        # 用不断扩大的窗口找其他残差点（status == 4/5）
        def _RMR_search_win(center_xy):
            # 解压坐标数据
            [xc,yc] = center_xy
            # 计算待解缠图像的长宽
            h_mat = input_mat.shape[0]
            w_mat = input_mat.shape[1]
            # 如果尚未检测到其他残差点并且上一代窗口大小还比全图小
            win_size = 3
            searched_RMR_list = []
            searched_RMR_idxs_list = []
            while (not len(searched_RMR_list)) and (win_size - 2 <= h_mat or win_size - 2 <= w_mat):
                # 计算窗口边界
                barrier = int((win_size - 1)/2)
                # 计算横向和纵向窗口可能的取值范围
                win_y_min = max(0, yc - barrier)
                win_y_max = min(w_mat-1, yc + barrier)
                win_x_min = max(0, xc - barrier)
                win_x_max = min(h_mat-1, xc + barrier)
                y_range = range(win_y_min, win_y_max+1)
                x_range = range(win_x_min, win_x_max+1)
                # 遍历窗口,查找RMR值,储存
                for i in x_range:
                    for j in y_range:
                        if not input_mat[i][j] == 0:
                            # 排除自身
                            if i == xc and j == yc:
                                continue
                            searched_RMR_list.append(input_mat[i][j])
                            searched_RMR_idxs_list.append([i,j])
                if len(searched_RMR_list):
                    break
                #  2.8.如果在搜索窗口中无残差点被检测到(else)
                #  2.9.搜索窗口增大，变成5x5窗口
                win_size += 2 
            return searched_RMR_list, searched_RMR_idxs_list
        
        # 路径算法
        def _new_brunch_idxs(start_idxs, end_idxs):
            # 开头和结尾对调没有区别
            # 解压数据
            [x1, y1] = start_idxs
            [x2, y2] = end_idxs
            
            # 储存结果
            new_brunch = []
            
            # 处理特殊情况
            if x1 == x2:
                # 确定大小（防止失控）
                y_max = max(y1, y2)
                y_min = min(y1, y2)
                for y in range(y_min+1, y_max):
                    new_brunch.append([x1, y])
                return new_brunch
            elif y1 == y2:
                # 确定大小（防止失控）
                x_max = max(x1, x2)
                x_min = min(x1, x2)
                for x in range(x_min+1, x_max+1):
                    new_brunch.append([x, y1])
                return new_brunch
            else:
                # 帽子戏法(对换xy，减少代码重复字段)
                xy_switch = False
                if abs(y1-y2) > abs(x1-x2):
                    temp1 = x1
                    temp2 = x2
                    x1 = y1
                    x2 = y2
                    y1 = temp1
                    y2 = temp2
                    
                    xy_switch = True
                # 计算直线数值
                k = float(y1-y2)/float(x1-x2)
                b = float(y1) - k*float(x1)
                
                # 确定大小（防止失控）
                x_max = max(x1, x2)
                x_min = min(x1, x2)
                # 逐个计算
                for x in range(x_min+1, x_max):
                    # 计算y的浮点数值
                    y_about = k*float(x) + b
                    # 计算精确的y的整数值
                    def __get_int_y(y_about):
                        # 分离正负号
                        if y_about < 0:
                            sign = -1
                            y_about = -y_about
                        else:
                            sign = 1

                        y_rate = y_about-math.floor(y_about)
                        # 分析进一还是不进一
                        if y_rate == 0:
                            return [sign * int(y_about)]
                        elif y_rate < 0.5:
                            return [sign * int(y_about)]
                        elif y_rate == 0.5:
                            return [sign * int(y_about), sign * (int(y_about) + 1)]
                        else:
                            return [sign * (int(y_about) + 1)]
                    
                    y_int_list = __get_int_y(y_about)
                    # 遍历输出
                    for y in y_int_list:
                        if xy_switch:
                            new_brunch.append([y,x])
                        else:      
                            new_brunch.append([x,y])
            return new_brunch
        
        # 获取开始和结束的点
        def _reset_point(start_RMR, end_RMR):
                    # 解压
                    [x_start, y_start] = start_RMR
                    [x_end, y_end] = end_RMR
                    
                    # 判断x方向是否需要偏移
                    if x_start < x_end:
                        x_reset = +1
                    else:
                        x_reset = 0
                        
                    # 判断y方向是否需要偏移
                    if y_start < y_end:
                        y_reset = +1
                    else:
                        y_reset = 0
                        
                    reset_point = [x_start + x_reset, y_start + y_reset]
                    return reset_point
        
        # 曼哈顿距离
        def _manhattan(x, y):
                # 曼哈顿距离
                x = np.array(x)
                y = np.array(y)
                return np.sum(np.abs(x - y))
        #  2.2.如果找到了,(if 4/5 in window)用brunch连接,(brunch_list.append)status:0->?
        # 初始化电荷和支切线,还有残差点连接属性
        charge_stack = _status_to_charge(_status_mat[xc0][yc0])
        brunch_list = []
        brunched_RMR_list = []
        searched_RMR_list, searched_RMR_idxs_list = _RMR_search_win([xc0,yc0])
        while not len(RMR_idxs_list) == len(brunched_RMR_list):
            # 虽然可能有多个结果，但是我们只使用其中一个(最短的)
            # 寻找最短的(使用曼哈顿距离)
            chosen_num = 0
            current_min_dist = 100000
            for idx,searched_RMR_indexs in enumerate(searched_RMR_idxs_list):
                new_dist = _manhattan([xc0, yc0], searched_RMR_indexs)
                if (new_dist < current_min_dist) and (not searched_RMR_indexs in brunched_RMR_list):
                    current_min_dist = new_dist
                    chosen_num = idx
            # 获得抽取的点
            searched_RMR = searched_RMR_list[chosen_num]
            [xc_next, yc_next] = searched_RMR_idxs_list[chosen_num]

            next_charge = _status_to_charge(searched_RMR)

            start_idxs = _reset_point([xc0,yc0], [xc_next, yc_next])
            end_idxs = _reset_point([xc_next, yc_next], [xc0,yc0])
            # 计算新支切线中间值
            new_brunch = _new_brunch_idxs(start_idxs, end_idxs)
            new_brunch += [start_idxs, end_idxs]
            # 支切线成员列表更新(charge)尚未计算
            for new_brunch_point in new_brunch:
                if not new_brunch_point in brunch_list:
                    brunch_list.append(new_brunch_point)
            # 更新残差点连接状态,累加电荷
            # 查看是否已经累加过电荷
            if not [xc_next,yc_next] in brunched_RMR_list:
                brunched_RMR_list.append([xc_next,yc_next])
                charge_stack += next_charge
            
            if not [xc0,yc0] in brunched_RMR_list:
                brunched_RMR_list.append([xc0,yc0])
            #  2.3.果另一个残差点的极性与当前残差点相反
            current_charge = _status_to_charge(_status_mat[xc0, yc0])
            if current_charge == -next_charge:
            # 2.4.就认为这个branch是平衡的(balanced)；
                # 判断是否可以结束
                if len(RMR_idxs_list) == len(brunched_RMR_list):
                    break
                # 搜索窗口放在其他RMR点上
                unbrunched_RMR_list = []
                for RMR_idxs in RMR_idxs_list:
                    if not RMR_idxs in brunched_RMR_list:
                        unbrunched_RMR_list.append(RMR_idxs)
                chosen_num = 0
                # 获得抽取的点
                [xc0, yc0] = unbrunched_RMR_list[chosen_num]
            #  2.5.如果另一个残差点的极性和当前残差点相同(else)
            else:
                #  2.6.那搜索窗口就放在新检测到的这个残差点位置
                [xc0,yc0] = [xc_next,yc_next]
            #  2.7.继续寻找残点(status == 4/5)，直到找到极性相反的残点并将它们用branch连接起来(brunch_list.append)且他们极性总和为0(if postive_num == negative_num)
            #  2.10.继续从开始的那个残差点来检测别的残差点 
            _searched_RMR_list = []
            _searched_RMR_idxs_list = []
            for idx in range(len(searched_RMR_list)):
                if not searched_RMR_idxs_list[idx] in brunched_RMR_list:
                    _searched_RMR_list.append(searched_RMR_list[idx])
                    _searched_RMR_idxs_list.append(searched_RMR_idxs_list[idx])
            searched_RMR_list = _searched_RMR_list
            searched_RMR_idxs_list = _searched_RMR_idxs_list
            # 开始新的一轮
            searched_RMR_list, searched_RMR_idxs_list = _RMR_search_win([xc0,yc0])
        
        #  2.11.原先残差点标记为0
        for RMR_indxs in brunched_RMR_list:
            # 解压
            [x_RMR, y_RMR] = RMR_indxs
            # 重新标记
            _status_mat[x_RMR, y_RMR] = 0
        
        #  2.12.对内容进行标记
        for brunch_point in brunch_list:
            # 解压
            [x_brunch_point, y_brunch_point] = brunch_point
            # 全部标记为3
            _status_mat[x_brunch_point, y_brunch_point] = 3
                
        print('生成支接结果')
        G_IMG.mat_to_heatmap(_status_mat, file_name="brunch_mat")
        
        # 遍历四邻域,查看是否满足条件,满足就Itoh并且改status
        def _4neighbor_process(x_former, y_former, _status_mat):
            # 通往四邻域的四个路径
            to_4neighbor = [[-1,0],[0,-1],[1,0],[0,1]]
            # 可行的邻域坐标
            # 遍历四邻域
            for x_to_4,y_to_4 in to_4neighbor:
                # 生成四邻域坐标
                x_4 = x_former + x_to_4
                y_4 = y_former + y_to_4
                # 检查：1.x是否越界 2.y是否越界 3.status == 0?
                if x_4 in range(self.origin_data.shape[0]):
                    if y_4 in range(self.origin_data.shape[1]):
                        if _status_mat[x_4, y_4] == 0:
                            # Itoh解缠
                            f_former = self.unwrap_data[x_former, y_former]
                            f_4 = self.unwrap_data[x_4, y_4]
                            # 判断差值，提供修正
                            f_4_fix = f_4 + _Itoh(f_4 - f_former)
                            # 执行解缠
                            self.unwrap_data[x_4, y_4] = f_4_fix
                            _status_mat[x_4, y_4] = 2
            return _status_mat
        process_name = 1
        # 三.标记完成，开始解缠
        while 0 in _status_mat:
            # 3.1.选择一个非支切线上的点作为起始点(status == 0),标记为已解缠 status 0->1
            x0_list, y0_list = np.where(_status_mat == 0) # 读取status == 0的坐标
            # 是否随机抽取
            if _radom_choose:
                chosen_num = np.random.choice(len(x0_list)) # 抽取随机号码
            else:
                chosen_num = 0
            # 获得抽取的起始点(x0,y0)
            # [x0, y0] = _4neighbor_status0_list[chosen_num]
            x0 = x0_list[chosen_num]
            y0 = y0_list[chosen_num]
            print(f"起始解缠点[{x0},{y0}]")
            # 标记为已解缠
            _status_mat[x0,y0] = 1
            # 3.2.将它的四邻域的非支切线上的点(status == 0),用Itoh方法解缠,然后将该四邻域点加入“邻接表” status 0->2
            # 执行如上函数，更新状态矩阵
            _status_mat = _4neighbor_process(x0, y0, _status_mat)
            # 3.3.只要“邻接表”不为空，从中取出一个点(status == 2)，并标记为已解缠 status 2->1
            # while“邻接表”不为空(exist 2)
            while 2 in _status_mat:
                x2_list, y2_list = np.where(_status_mat == 2) # 读取status == 2的坐标
                # 获得被选中的邻接表点(x2,y2)
                x2 = x2_list[0]
                y2 = y2_list[0]
                # 标记为已解缠
                _status_mat[x2,y2] = 1 
                # 3.4.将该点的四邻域（条件）{1.未在“邻接表” 2.非支切线上 3.未解缠的点 status == 0}，用Itoh方法解缠,并且将他们加入“邻接表” status 0->2
                _status_mat = _4neighbor_process(x2, y2, _status_mat)
                # 打印结果
                # G_IMG.mat_to_heatmap(_status_mat, file_name=f"unwarp_process_{process_name}")
                process_name += 1
        # 3.5.对于在支切线上的点(status == 3)，根据它们的邻域中已解缠点的相位，对它们逐一解缠
        # 抽取支切线上的点
        x3_list, y3_list = np.where(_status_mat == 3) # 读取status == 3的坐标
        # 按照邻域进行相位解缠
        def _brunch_unwarp_by_neighbor(center_xy):
            # 解压坐标数据
            [x3,y3] = center_xy
            # 计算待解缠图像的长宽
            h_mat = input_mat.shape[0]
            w_mat = input_mat.shape[1]
            # 计算4邻域内可能的取值范围
            four_neighbour_list = []
            if y3 - 1 >= 0:
                four_neighbour_list.append([0,-1])
            if y3 + 1 <= w_mat-1:
                four_neighbour_list.append([0, 1])
            if x3 - 1 >= 0:
                four_neighbour_list.append([-1, 0])
            if x3 + 1 <= h_mat-1:
                four_neighbour_list.append([1, 0])
            for [offset_x, offset_y] in four_neighbour_list:
                x_neighbour = x3 + offset_x
                y_neighbour = y3 + offset_y
                status_neighbour = _status_mat[x_neighbour, y_neighbour]
                if status_neighbour == 1:
                    # 符合条件，开始解缠
                    p_neighbour = self.unwrap_data[x_neighbour, y_neighbour]
                    p3 = self.unwrap_data[x3, y3]
                    p3_fix = p3 + _Itoh(p3 - p_neighbour)
                    return p3_fix
            # 暂时无法解缠，返回空值
            return -99999
        while 3 in _status_mat:
            # 反复操作，保证被包围在内部的点也被解缠
            for idx in range(len(x3_list)):
                center_xy = [x3_list[idx], y3_list[idx]]
                p3_fix = _brunch_unwarp_by_neighbor(center_xy)
                
                # 如果不是空值
                if not p3_fix==-99999:
                    self.unwrap_data[center_xy[0],center_xy[1]] = p3_fix
                    _status_mat[center_xy[0],center_xy[1]] = 1
            
            x3_list, y3_list = np.where(_status_mat == 3) # 读取status == 3的坐标

        # print(_status_mat)
        return self.unwrap_data
    
if __name__ == "__main__":
    # 设置输入数据
    input_mat = SRC
    # 打印原始数据
    print('真实相位数据：')
    print(input_mat)
    # 可视化原始数据
    cmap0 = cm.plasma
    G_IMG.mat_to_heatmap(input_mat, cmap=cmap0, file_name='src_heatmap')#, file_name='src_heatmap'
    G_IMG.mat_to_3D(input_mat, img_type = "tri-surface", cmap=cmap0, file_name="src_3D")#, file_name="src_3D"
    print()
    
    # 初始化类，初始化的过程自动进行缠绕
    process_wrap = WrapData(input_mat)
    wrap_data = process_wrap.wrap_data
    
    # 打印缠绕数据
    print('缠绕相位数据：')
    print(wrap_data)
    # 可视化缠绕数据
    G_IMG.mat_to_heatmap(wrap_data, cmap=cmap0, file_name="wrap_heatmap")#, file_name="wrap_heatmap"
    G_IMG.mat_to_3D(wrap_data, cmap=cmap0, img_type = "tri-surface",file_name="wrap_3D")#,file_name="wrap_3D"
    print()
    
    # 打印解缠绕数据
    unwrap_data = process_wrap.unwrap(True)
    print('解缠相位数据：')
    print(unwrap_data)
    G_IMG.mat_to_heatmap(unwrap_data, cmap=cmap0, file_name='unwrap_heatmap')#, file_name='unwrap_heatmap'
    G_IMG.mat_to_3D(unwrap_data, img_type = "tri-surface", cmap=cmap0, file_name="unwrap_3D")#, file_name="unwrap_3D"
    print()
    
    # 测试结果准确性：
    print('解缠相位数据与缠绕相位数据差值')
    diff_data0 = unwrap_data - wrap_data
    diff_data0 -= np.ones_like(diff_data0)*stats.mode(diff_data0)[0][0]
    print(diff_data0)
    
    print('解缠相位数据与真实相位数据差值')
    diff_data = unwrap_data - input_mat
    diff_data -= np.ones_like(diff_data)*stats.mode(diff_data)[0][0]
    print(diff_data)