
import numpy as np
import open3d as o3d
import os

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

    


    path='/home/ryan/Documents/Data/3D/cache/nbv/'
    pcd=o3d.io.read_point_cloud(os.path.join(path,'8.ply'))
    #pcd=o3d.io.read_triangle_mesh(os.path.join(path,'8.ply'))
    #pcd.paint_uniform_color([1, 0.706, 0])
    #o3d.visualization.draw_geometries([pcd])

    alpha = 0.1
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    # 可视化重建结果

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.05)
    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    center=np.mean(indices,axis=0)
    print(center)
    #o3d.visualization.draw_geometries([voxel_grid])

    x = np.linspace(-1, 1, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.power(0.01-(np.power(mesh_x-center[0]*0.05, 2) + np.power(mesh_y-center[1]*0.05, 2)),0.5)+center[2]*0.05
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z, -1)
    print('xyz')
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd_c = o3d.geometry.PointCloud()
    pcd_c.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd,pcd_c])


    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.01)
    #pcd.paint_uniform_color([0.424, 0.781, 0.929])
    o3d.visualization.draw_geometries([voxel_grid])
    i=9
    """
    for item in os.listdir(path):
        pcd = o3d.io.read_point_cloud(os.path.join(path,item))
        if not pcd.is_empty():
            print(item)
            print(np.asarray(pcd.points).shape)
            i=9
            #o3d.visualization.draw_geometries([pcd])
    """
    """
    #ok
    alpha = 0.1
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    # 可视化重建结果
    o3d.visualization.draw_geometries([mesh], window_name="点云重建",
                                    width=800, 
                                    height=600,
                                    mesh_show_back_face=True) 
    i=0
    """
    
    
    
    
    #ok
    pcd.paint_uniform_color([0.424, 0.781, 0.929])
    # Poisson surface reconstruction
    radius = 0.001  # 搜索半径
    max_nn = 5  # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 法线估计
    # BPA重建
    radii = [0.005, 0.01, 0.02, 0.04]
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # 可视化重建结果
    o3d.visualization.draw_geometries([pcd,bpa_mesh], window_name="点云重建",
                                    width=800,  
                                    height=600, 
                                    mesh_show_back_face=True,point_show_normal=True)
    
    
    
    pcd.paint_uniform_color([0.424, 0.781, 0.929])
    # Poisson surface reconstruction
    radius = 0.01  # 搜索半径
    max_nn = 10  # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 法线估计

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    print(mesh)
    # 可视化重建结果
    o3d.visualization.draw_geometries([mesh], window_name="点云重建",
                                    width=800,
                                    height=600,
                                    mesh_show_back_face=True)
    

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh], zoom=0.664,
                                    front=[-0.4761, -0.4698, -0.7434],
                                    lookat=[1.8900, 3.2596, 0.9284],
                                    up=[0.2304, -0.8825, 0.4101])
    

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=5)
    o3d.visualization.draw_geometries([downpcd])

    """
    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])

    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.normals)[:10, :])
    print("")

    print("Load a polygon volume and use it to crop the original point cloud")
    vol = o3d.visualization.read_selection_polygon_volume(
        "../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    o3d.visualization.draw_geometries([chair])
    print("")

    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([chair])
    print("")

    """

    