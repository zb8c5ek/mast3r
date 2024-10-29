__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
import numpy as np
import trimesh

def project_mesh_to_depth_map(mesh, camera_intrinsics, camera_extrinsics, image_size):
    """
    Project a 3D mesh to generate a depth map.

    Args:
    - mesh (trimesh.Trimesh): The 3D mesh.
    - camera_intrinsics (numpy.ndarray): Camera intrinsic matrix (3x3).
    - camera_extrinsics (numpy.ndarray): Camera extrinsic matrix (4x4).
    - image_size (tuple): Size of the output depth map (height, width).

    Returns:
    - numpy.ndarray: The generated depth map.
    """
    height, width = image_size
    depth_map = np.zeros((height, width), dtype=np.float32)

    # Generate rays from the camera through each pixel
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Create a grid of pixel coordinates
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    directions = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], axis=-1)

    # Transform directions to world coordinates
    directions = directions @ camera_extrinsics[:3, :3].T
    origins = np.tile(camera_extrinsics[:3, 3], (height, width, 1))

    # Cast rays and find intersections with the mesh
    ray_origins = origins.reshape(-1, 3)
    ray_directions = directions.reshape(-1, 3)

    # Access the geometry within the scene
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    locations, index_ray, _ = mesh.ray.intersects_location(ray_origins, ray_directions)

    # Store the intersection points as depth values
    for loc, ray_idx in zip(locations, index_ray):
        y, x = divmod(ray_idx, width)
        depth_map[y, x] = loc[2]  # Use the z-coordinate as the depth value

    return depth_map


if __name__ == '__main__':
    # fp_mesh = '/d_disk/RunningData/ZhiNengDao/20-from-2075-to-94-720P_160/blob3recon/blob_0-start2075_end2077/scene_mesh.glb'
    # mesh = trimesh.load(fp_mesh)
    # focal = 308.37515
    # cam2world = np.array([[ 0.04749054, -0.39668703,  0.9167246 ,  0.14828019],
    #                       [ 0.11363892,  0.91394717,  0.3895982 ,  0.06373303],
    #                       [-0.9923864 ,  0.08567338,  0.08848292, -0.07369868],
    #                       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # img_size = (336, 512)
    # depth_map = project_mesh_to_depth_map(
    #     mesh, np.array([[focal, 0, img_size[1] // 2], [0, focal, img_size[0] // 2], [0, 0, 1]]), cam2world, img_size
    # )
    # test on a sphere mesh
    import time
    start_time = time.time()
    mesh = trimesh.primitives.Sphere()

    # create some rays
    ray_origins = np.array([[0, 0, -5], [2, 2, -10]])
    ray_directions = np.array([[0, 0, 1], [0, 0, 1]])

    """
    Signature: mesh.ray.intersects_location(ray_origins,
                                            ray_directions,
                                            multiple_hits=True)
    Docstring:

    Return the location of where a ray hits a surface.

    Parameters
    ----------
    ray_origins:    (n,3) float, origins of rays
    ray_directions: (n,3) float, direction (vector) of rays


    Returns
    ---------
    locations: (n) sequence of (m,3) intersection points
    index_ray: (n,) int, list of ray index
    index_tri: (n,) int, list of triangle (face) indexes
    """

    # run the mesh- ray test
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions
    )

    # stack rays into line segments for visualization as Path3D
    # ray_visualize = trimesh.load_path(
    #     np.hstack((ray_origins, ray_origins + ray_directions)).reshape(-1, 2, 3)
    # )
    #
    # # make mesh transparent- ish
    # mesh.visual.face_colors = [100, 100, 100, 100]

    # create a visualization scene with rays, hits, and mesh
    # scene = trimesh.Scene([mesh, ray_visualize, trimesh.points.PointCloud(locations)])

    # display the scene
    # scene.show()
    print("Time taken: ", time.time() - start_time)