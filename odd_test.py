
import numpy as np
import open3d as o3d
import os
import time
import matplotlib.pyplot as plt

#from line_profiler import LineProfiler

def speed2cm(speed):
    pass

def calVolume(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # #display_inlier_outlier(pcd, ind)
    inlier_cloud = pcd.select_by_index(ind)
    pcd = inlier_cloud
    hull,  idx = pcd.compute_convex_hull()
    o3d.visualization.draw_geometries([hull])
    volume = hull.get_volume()
    print("volume is :" + str(volume))
    return volume


#@profile
def getCenter():
    path = "C:\\Users\\hasee\\Desktop\\Data\\3dPointCloud"
    pcd = o3d.io.read_point_cloud(os.path.join(path, '8.ply'))
    #pcd=o3d.io.read_triangle_mesh(os.path.join(path,'8.ply'))
    #pcd.paint_uniform_color([1, 0.706, 0])
    #o3d.visualization.draw_geometries([pcd])

    alpha = 0.1
    print(f"alpha={alpha:.3f}")
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    # # 可视化重建结果
    #
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
    #                                                           voxel_size=0.05)
    # voxels = voxel_grid.get_voxels()  # returns list of voxels

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
    voxels = voxel_grid.get_voxels()
    indices = np.stack(list(vx.grid_index for vx in voxels))
    center = np.mean(indices, axis=0)
    print(center)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Removal Outlier',
                                      width=1920,
                                      height=1080, left=50, top=50, point_show_normal=False,
                                      mesh_show_wireframe=False,
                                      mesh_show_back_face=False)

if __name__ == "__main__":
    """
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print('xyz')
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud('/home/ryan/Documents/Data/3D/cache/test_l.ply', pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud('/home/ryan/Documents/Data/3D/cache/test_l.ply')
    pcd=pcd_load

    pcd.paint_uniform_color([0.424, 0.781, 0.929])
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd],point_show_normal=True)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh], zoom=0.664,
                                    front=[-0.4761, -0.4698, -0.7434],
                                    lookat=[1.8900, 3.2596, 0.9284],
                                    up=[0.2304, -0.8825, 0.4101],point_show_normal=True)

    """
    
    # Poisson surface reconstruction
    #radius = 0.001  # 搜索半径
    #max_nn = 10  # 邻域内用于估算法线的最大点数
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 法线估计

    


    #path='/home/ryan/Documents/Data/3D/cache/nbv/'
    #getCenter()
    path = r"C:\Users\hasee\Desktop\Data\3dPointCloud"
    pcd = o3d.io.read_point_cloud(os.path.join(path, '8.ply'))
    alpha = 0.1
    print(f"alpha={alpha:.3f}")

    start = time.time()
    center = pcd.get_center()
    end = time.time()
    print(end-start)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)

    # create a point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # create a numpy array with a single point
    point = center
    points = [point]

    # add the point to the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector([[255,0,0]])

    # visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #o3d.visualization.draw_geometries([point_cloud])

    vis.add_geometry(pcd)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()

    radius = 0.1  # 搜索半径
    max_nn = 30
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    # o3d.visualization.draw_geometries([pcd], window_name="法线估计",
    #                                   point_show_normal=True,
    #                                   width=800,  # 窗口宽度
    #                                   height=600)  # 窗口高度
    print(len(pcd.normals))

    z_coords = np.asarray(pcd.normals)[:, 2]
    indices = np.where(z_coords > 0.9)[0]
    selected_points = pcd.select_by_index(indices)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(selected_points)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    #o3d.visualization.draw_geometries([selected_points])

    # calculate distance
    # dists = selected_points.compute_point_cloud_distance(point_cloud)
    # print(len(dists))

    # distances = np.asarray(dists)
    # indices = np.where(distances < 20)[0]
    # selected_points = selected_points.select_by_index(indices)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            selected_points.cluster_dbscan(eps=1, min_points=100, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    selected_points.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(selected_points)
    vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    vis.run()

    volume = calVolume(pcd)
    density = 1
    weight = volume * density
    single_hand = 200000
    hand_num = weight // single_hand + 1

    if hand_num == 1:
        pass
        #detect_single_hand()

    elif hand_num == 2:
        pass
        #detect_double_hand()


    # pcd = pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # # pcd = pcd.orient_normals_towards_camera_location(camera_location=[0,0,0])
    # pcd = pcd.poisson_mesh()
    # pcd = pcd.sample_points_possion_disk(10000)


    # voxels = voxel_grid.get_voxels()
    # indices = np.stack(list(vx.grid_index for vx in voxels))
    # center = np.mean(indices, axis=0)
    # print(center)

    #cen = o3d.geometry.VoxelGrid.create_from_point_cloud(center, voxel_size=1)
    #o3d.visualization.draw_geometries([voxel_grid])



    ###o3d.visualization.draw_geometries([voxel_grid])

    #
    # o3d.visualization.draw_geometries(xyz)
    #
    # # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    # pcd_c = o3d.geometry.PointCloud()
    # pcd_c.points = o3d.utility.Vector3dVector(xyz)
    # o3d.visualization.draw_geometries([pcd,pcd_c])
    #
    #
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
    #                                                         voxel_size=0.01)
    # #pcd.paint_uniform_color([0.424, 0.781, 0.929])
    # o3d.visualization.draw_geometries([voxel_grid])
    # i=9
    # """
    # for item in os.listdir(path):
    #     pcd = o3d.io.read_point_cloud(os.path.join(path,item))
    #     if not pcd.is_empty():
    #         print(item)
    #         print(np.asarray(pcd.points).shape)
    #         i=9
    #         #o3d.visualization.draw_geometries([pcd])
    # """
    # """
    # #ok
    # alpha = 0.1
    # print(f"alpha={alpha:.3f}")
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    # # 可视化重建结果
    # o3d.visualization.draw_geometries([mesh], window_name="点云重建",
    #                                 width=800,
    #                                 height=600,
    #                                 mesh_show_back_face=True)
    # i=0
    # """
    #
    #
    #
    #
    # #ok
    # pcd.paint_uniform_color([0.424, 0.781, 0.929])
    # # Poisson surface reconstruction
    # radius = 0.001  # 搜索半径
    # max_nn = 5  # 邻域内用于估算法线的最大点数
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 法线估计
    # # BPA重建
    # radii = [0.005, 0.01, 0.02, 0.04]
    # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # # 可视化重建结果
    # o3d.visualization.draw_geometries([pcd,bpa_mesh], window_name="点云重建",
    #                                 width=800,
    #                                 height=600,
    #                                 mesh_show_back_face=True,point_show_normal=True)
    #
    #
    #
    # pcd.paint_uniform_color([0.424, 0.781, 0.929])
    # # Poisson surface reconstruction
    # radius = 0.01  # 搜索半径
    # max_nn = 10  # 邻域内用于估算法线的最大点数
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 法线估计
    #
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    # print(mesh)
    # # 可视化重建结果
    # o3d.visualization.draw_geometries([mesh], window_name="点云重建",
    #                                 width=800,
    #                                 height=600,
    #                                 mesh_show_back_face=True)
    #
    #
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # print(mesh)
    # o3d.visualization.draw_geometries([mesh], zoom=0.664,
    #                                 front=[-0.4761, -0.4698, -0.7434],
    #                                 lookat=[1.8900, 3.2596, 0.9284],
    #                                 up=[0.2304, -0.8825, 0.4101])
    #
    #
    # print("Downsample the point cloud with a voxel of 0.05")
    # downpcd = pcd.voxel_down_sample(voxel_size=5)
    # o3d.visualization.draw_geometries([downpcd])
    #
    # """
    # print("Recompute the normal of the downsampled point cloud")
    # downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))
    # o3d.visualization.draw_geometries([downpcd])
    #
    # print("Print a normal vector of the 0th point")
    # print(downpcd.normals[0])
    # print("Print the normal vectors of the first 10 points")
    # print(np.asarray(downpcd.normals)[:10, :])
    # print("")
    #
    # print("Load a polygon volume and use it to crop the original point cloud")
    # vol = o3d.visualization.read_selection_polygon_volume(
    #     "../../TestData/Crop/cropped.json")
    # chair = vol.crop_point_cloud(pcd)
    # o3d.visualization.draw_geometries([chair])
    # print("")
    #
    # print("Paint chair")
    # chair.paint_uniform_color([1, 0.706, 0])
    # o3d.visualization.draw_geometries([chair])
    # print("")
    #
    # """

