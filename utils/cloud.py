from attr import attrs, attrib
import h5py
import math
import numpy as np
import open3d as o3d
from os.path import isfile
from sklearn.cluster import KMeans


DEPTH_MAP_KEY = "depth_map"
INTENSITY_MAP_KEY = "intensity_map"
HOR_FOV_KEY = "horizontal_fov_deg"
VER_FORV_KEY = "vertical_fov_deg"

@attrs
class RawData(object):
    depth_map = attrib()
    intensity_map = attrib()
    fov = attrib()


def load_cloud(path: str) -> RawData:
    assert isfile(path), f"{path} doesn't exist"
    f = h5py.File(path, 'r')
    depth_dataset = f[DEPTH_MAP_KEY]
    intensity_dataset = f[INTENSITY_MAP_KEY]
    dset_attrs = depth_dataset.attrs
    fov = dset_attrs[VER_FORV_KEY], dset_attrs[HOR_FOV_KEY]
    return RawData(
        depth_map=np.asarray(depth_dataset),
        intensity_map=np.asarray(intensity_dataset),
        fov=fov,
    )


def get_uniform_median(size: int) -> float:
    if size % 2:
        return size // 2
    else:
        return size / 2 + 0.5


def normalize_vector(v) -> np.ndarray:
    assert isinstance(v, (list, np.ndarray))
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    assert not np.isclose(np.abs(norm), 0.), "Vector has zero norm"
    return v / norm


def compute_rotation_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    assert len(vec1) == len(vec2) == 3
    vec1 = normalize_vector(vec1)
    vec2 = normalize_vector(vec2)
    dp = np.dot(vec1, vec2)
    assert np.sign(dp) > 0
    return np.arccos(dp)


def make_rotation_x(angle: float) -> np.ndarray:
    return np.asarray([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ])


def make_rotation_y(angle: float) -> np.ndarray:
    return np.asarray([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1., 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])


def compute_rot_matrix_xy(x_angle: float, y_angle: float) -> np.ndarray:
    return np.dot(
        make_rotation_x(x_angle),
        make_rotation_y(y_angle),
    )


def compute_cam2world_transform(cam_axis_xyz: np.ndarray, translation_xyz=None) -> np.ndarray:
    if translation_xyz is None:
        translation_xyz = [0, 0, 0]
    x, y, z = normalize_vector(cam_axis_xyz)
    
    x_rot = compute_rotation_angle([0, 0, 1], [0, y, z]) * np.sign(-y)
    y_rot = compute_rotation_angle([0, y, z], [x, y, z]) * np.sign(x)    
    rotation = compute_rot_matrix_xy(x_rot, y_rot)
    
    transform = np.zeros(shape=(4, 4), dtype=np.float32)
    transform[:3, :3] = rotation
    transform[0, 3] = translation_xyz[0]
    transform[1, 3] = translation_xyz[1]
    transform[2, 3] = translation_xyz[2]
    transform[3, 3] = 1
    return transform


def apply_transform(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_cnt = xyz.shape[0]   
    xyz_homog = np.hstack([xyz, np.ones(shape=[points_cnt, 1])])
    return np.dot(transform, xyz_homog.T).T[:, :-1]


def compute_cloud(depth_map: np.ndarray,
                  intensity_map: np.ndarray,
                  fov: np.ndarray,
                  cam_axis_dir: np.ndarray):
    mask = np.logical_not(np.isnan(depth_map))
    map_shape = depth_map.shape
    pixel_angles = np.asarray(fov) / np.asarray(map_shape)
    pixel_angles = np.radians(pixel_angles)
    
    y_idx, x_idx =  np.indices(map_shape, dtype=np.float32)
    y_offset, x_offset = get_uniform_median(map_shape[0]), get_uniform_median(map_shape[1])
    
    y_idx = (y_idx - y_offset) * pixel_angles[0]
    x_idx = (x_idx - x_offset) * pixel_angles[1]
    
    xyzi = np.stack([x_idx, y_idx, depth_map, intensity_map], axis=0)
    xyzi = np.transpose(xyzi, [1, 2, 0])

    xyzi = xyzi[mask]
    z = xyzi[:, 2]
    xyzi[:, 0] = z * np.tan(xyzi[:, 0])
    xyzi[:, 1] = z * np.tan(xyzi[:, 1])
    
    cam2world = compute_cam2world_transform(cam_axis_dir)   
    xyzi[:, :3] = apply_transform(xyzi[:, :3], cam2world)
    return xyzi


def clusterize_cloud(cloud: np.ndarray, clusters=2) -> list:
    assert cloud.shape[1] >= 3

    coords = cloud[:, :3]
    clusterizer = KMeans(n_clusters=clusters, random_state=0)
    clusterizer.fit(coords)
    
    output = []
    for cluster_id in range(clusters):
        mask = clusterizer.labels_ == cluster_id
        output.append(cloud[mask])
    return output


def estimate_sphere_by_points(cloud: np.ndarray) -> tuple:
    xyz = cloud[:, :3]
    points_cnt = xyz.shape[0]
    coeffs = np.ones((points_cnt, 4))
    coeffs[:, 0:3] = xyz

    b_vector = np.sum(np.multiply(xyz, xyz), axis=1)
    sol, _, _, _ = np.linalg.lstsq(coeffs, b_vector)
    radius = math.sqrt(
        (sol[0] * sol[0] / 4.0) +
        (sol[1] * sol[1] / 4.0) +
        (sol[2] * sol[2] / 4.0) +
        sol[3]
    )
    x, y, z = sol[0] / 2.0, sol[1] / 2.0, sol[2] / 2.0
    return np.asarray([x, y, z]), radius


def segment_cuboid(cloud: np.ndarray, expected_faces=3, dist_thresh=0.15) -> tuple:
    xyz = cloud[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    plane_equations = []
    segmented_clouds = []
    for _ in range(expected_faces):
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=100)
        plane_equations.append(plane_model)
        inlier_cloud = pcd.select_by_index(inliers)
        segmented_clouds.append(np.asarray(inlier_cloud.points))
        pcd = pcd.select_by_index(inliers, invert=True)
    
    return plane_equations, segmented_clouds


def is_cloud_cuboid(cloud: np.ndarray) -> bool:
    total_points = len(cloud)
    _, segments = segment_cuboid(cloud, expected_faces=3, dist_thresh=0.05)
    points_at_plane_faces = sum([len(x) for x in segments])
    segmented_share = points_at_plane_faces / total_points
    return segmented_share > 0.5


def estimate_cuboid_by_points(cloud: np.ndarray):
    xyz = cloud[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    obb = pcd.get_oriented_bounding_box()
    whl = obb.extent
    xyz = obb.center
    return np.asarray(xyz), np.asarray(whl)


def cuboid_area(whl):
    w, h, l = whl
    return 2 * (w * h + w * l + l * h)


def cuboid_volume(whl):
    w, h, l = whl
    return w * h * l


def sphere_area(r):
    return 4 * np.pi * r ** 2


def sphere_volume(r):
    return np.pi * r ** 3 * 4. / 3. 


def save_cloud_to_h5(cloud, path):
    hf = h5py.File(path, "w")
    hf.create_dataset("cloud", data=cloud)
    hf.close()
