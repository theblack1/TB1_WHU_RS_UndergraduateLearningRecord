from osgeo import gdal
import numpy as np

class ReadRadar():
    # 初始化数据读取内容
    def __init__(self, file_name):
        # 读取数据
        self.file_name = file_name
        self.raw_dataset = self.read_image()
        
        # 提取数组
        self.raw_array = self.band_to_array(band_num = 0)
        
        # 初始化数值
        self.str_img = None # 强度图像
        self.num_bands = self.raw_dataset.RasterCount # 波段数目
        self.rows = self.raw_dataset.RasterYSize # 行列
        self.cols = self.raw_dataset.RasterXSize
        self.desc = self.raw_dataset.GetDescription() # 描述
        self.metadata = self.raw_dataset.GetMetadata()  # 元数据
        self.driver = self.raw_dataset.GetDriver() # 驱动
        self.proj =  self.raw_dataset.GetProjection() # 投影信息
        self.gt = self.raw_dataset.GetGeoTransform() # 地理变换
        
        # 提取强度图
        self.str_array = self.get_str_img()

    # 从路径读取雷达图片
    def read_image(self):
        dataset = gdal.Open(self.file_name, gdal.GA_ReadOnly)
        if not dataset:
            # 如果没有正确打开
            print("ERROR:File " + self.file_name + " Can't Open!")
            exit()
        
        return dataset
    
    # 展示图像信息
    def show_info(self):      
        #查看波段数
        print("波段数\nnumber of bands：{}\n".format(self.num_bands))
        print('\n')

        #查看行列数
        print('影像尺寸\nImage size is: {r} rows x {c} columns\n'.format(r=self.rows, c=self.cols))
        print('\n')

        #查看描述和元数据
        print('影像名称\nRaster description: {desc}'.format(desc=self.desc))
        print('\n')
        print('影像元数据\nRaster metadata:\n{meta}'.format(meta = self.metadata))
        print('\n')

        #查看打开这个影像的驱动
        print("影像格式\nRaster driver: {}".format(self.driver))
        print('\n')
        
        #查看投影信息
        print('影像投影格式\nImage projection:{}'.format(self.proj))
        print('\n')
        
        #查看地理变换
        print('影像地理变换\nImage geo-transform:{gt}\n'.format(gt=self.gt))
        print('\n')

    # 展示波段信息
    def show_band_info(self, band_num):
        # 提取并且展示该波段
        band = self.raw_dataset.GetRasterBand(band_num)
        print("这个波段\nThis band:")
        print(band)
        print()
        
        #查看该波段的数据类型
        datatype = band.DataType
        print('波段影像的类型是{}'.format(datatype))
        print()


        #查看数据类型的名称
        datatype_name = gdal.GetDataTypeName(band.DataType)
        print('波段类型名称是{}'.format(datatype_name))
        print()

        #产看该数据结构占用的存储空间
        bytes = gdal.GetDataTypeSize(band.DataType)
        print('波段的数据大小:{}'.format(bytes))
        print()

        #查看一些该波段的统计量
        print('该波段的统计数据：')
        band_max, band_min, band_mean, band_stddev = band.GetStatistics(0,1)
        print('Mean:{m}\nStddev:{s}'.format(m = band_mean, s=band_stddev))

        # 查看波段最大最小值
        min = band_min
        max = band_max
        
        # 如果波段最大最小值过小，则保留多位小数处理
        if not min or not max:
            (min,max) = band.ComputeRasterMinMax(True)
        print("Min={:.8f}\nMax={:.8f}".format(min,max))
        print()
        
        # 判断波段数据是否有异常
        if band.GetOverviewCount() > 0:
            print("Band has {} overviews".format(band.GetOverviewCount()))

        # 读取波段颜色表信息
        if band.GetRasterColorTable():
            print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

    # 波段信息转换为数组
    def band_to_array(self, band_num):
        if band_num == 0:
            # 如果读取为0，将整个数据转换为数组
            dataset_array = self.raw_dataset.ReadAsArray()
            return dataset_array
            
        # 读取波段
        band = self.raw_dataset.GetRasterBand(band_num)  # 获取波段数据
        
        # 转化为数组
        band_array = band.ReadAsArray()
        print("提取得到数组尺寸：")
        print(band_array.shape)
    
        return band_array

    # 提取强度图array
    def get_str_img(self):
        if self.num_bands == 1:
            return self.raw_array
        elif self.num_bands == 2:
            band1 = self.band_to_array(1)
            band2 = self.band_to_array(2)
            
            # 转换为浮点数提高精度
            band1 = band1.astype(np.float32)
            band2 = band2.astype(np.float32)
            
            # 公式计算强度
            str_array = np.sqrt(np.add(np.square(band1),np.square(band2)))
            
            return str_array
        else:
            return None
