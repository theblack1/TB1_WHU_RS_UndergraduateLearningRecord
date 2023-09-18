import open3d as o3d
import numpy as np

#读取电脑中的点云文件
target = o3d.io.read_point_cloud(r'./bunny txt/bun000 - Cloud.txt', format="xyz")  #source 为需要配准的点云
source1 = o3d.io.read_point_cloud(r'./bunny txt/bun045 - Cloud.txt', format="xyz")
# source2 = o3d.io.read_point_cloud(r'./bunny txt/bun270 - Cloud.txt', format="xyz")
source3 = o3d.io.read_point_cloud(r'./bunny txt/bun315 - Cloud.txt', format="xyz")



#为两个点云上上不同的颜色
target.paint_uniform_color([1, 0.706, 0])
source1.paint_uniform_color([0, 0.651, 0])
# source2.paint_uniform_color([0, 0.751, 0])
source3.paint_uniform_color([0, 0.951, 0])

# 显示配准前情况
# vis_before = o3d.visualization.Visualizer()
# vis_before.create_window()
# vis_before.add_geometry(target)
# vis_before.add_geometry(source1)
# vis_before.add_geometry(source2)
# vis_before.add_geometry(source3)
#
# vis_before.poll_events()
# vis_before.update_renderer()
#
# vis_before.run()

threshold = 1.0  #移动范围的阀值
trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                         [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                         [0,0,1,0],   # 这个矩阵为初始变换
                         [0,0,0,1]])

#运行icp
reg_p2p1 = o3d.pipelines.registration.registration_icp(
        source1, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
# reg_p2p2 = o3d.pipelines.registration.registration_icp(
#         source2, target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint())
reg_p2p3 = o3d.pipelines.registration.registration_icp(
        source3, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())


#将我们的矩阵依照输出的变换矩阵进行变换
print(reg_p2p3.transformation)
source1.transform(reg_p2p1.transformation)
# source2.transform(reg_p2p2.transformation)
source3.transform(reg_p2p3.transformation)

#创建一个 o3d.visualizer class
vis = o3d.visualization.Visualizer()
vis.create_window()

#将两个点云放入visualizer
vis.add_geometry(source1)
# vis.add_geometry(source2)
vis.add_geometry(source3)
vis.add_geometry(target)

#让visualizer渲染点云
# vis.update_geometry()
vis.poll_events()
vis.update_renderer()

vis.run()