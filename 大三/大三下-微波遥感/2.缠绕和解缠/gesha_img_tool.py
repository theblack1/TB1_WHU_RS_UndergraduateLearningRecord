from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
# color map信息请见：https://zhuanlan.zhihu.com/p/114420786
DEFAULT_COLORMAP = cm.magma


class GeshaImgTool():
    # 初始化函数
    def __init__(self):
        return
    
    # 生成保存路径
    def get_save_dir(self, file_name, path = "./data/GeshaImgTool_default_save", driver = 'png', _time_labled = True):
        import os
        import time
        
        # 递归创建文件夹
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 是否使用时间标签来保证每一次保存都有不同名称
        time_string = ''
        if _time_labled:
            time_string = time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss",time.localtime(time.time()))
        
        # 生成文件名
        file_save_name = path + '/' + time_string + '_' + file_name + '.' + driver
        
        return file_save_name
    
    # 矩阵转热力图
    def mat_to_heatmap(self, input_mat, file_name = "", cmap=DEFAULT_COLORMAP):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        #　生成标签
        label = ["{}".format(i) for i in range(1, input_mat.shape[0]+1)]
        df = pd.DataFrame(input_mat, index=label, columns=label)

        # 绘制热力图
        plt.figure(figsize=(7.5, 6.3))
        ax = sns.heatmap(df, xticklabels=df.corr().columns, 
                        yticklabels=df.corr().columns, cmap=cmap,
                        linewidths=6, annot=True)

        # 设置坐标系
        plt.xticks(fontsize=16,family='Times New Roman')
        plt.yticks(fontsize=16,family='Times New Roman')

        # 保存文件
        if len(file_name):
            save_dir = self.get_save_dir(file_name = file_name)
            plt.savefig(save_dir)
        
        # 展示热力图
        plt.tight_layout()
        plt.show()
    
    # 矩阵转3D图
    # 参考代码https://blog.csdn.net/qq_40811682/article/details/117027899
    def mat_to_3D(self, input_mat, img_type, file_name = "", cmap=DEFAULT_COLORMAP):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np
        
        # 初始化3D图像
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        # 准备数据
        X = range(input_mat.shape[0])
        Y = range(input_mat.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = input_mat
        
        # 绘制表面
        if img_type == "surface":
            # 表面图Surface plots
            surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                                linewidth=0, antialiased=False)
        elif img_type == "tri-surface":
            surf = ax.plot_trisurf(np.array(X).flatten(), np.array(Y).flatten(), Z.flatten(), cmap = cmap)
            
        
        # 定制化坐标系
        ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # 添加颜色条
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # 保存文件
        if len(file_name):
            save_dir = self.get_save_dir(file_name = file_name)
            plt.savefig(save_dir)
        
        # 显示图像
        plt.show()
        
