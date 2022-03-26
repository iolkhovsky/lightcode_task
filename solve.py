import argparse
from os.path import join
import logging

from numpy import save

from utils.cloud import *
from utils.common import force_create_folder, remove_file
from utils.visualization import visualize_cloud


def parse_args():
    parser = argparse.ArgumentParser(description='Test task solver')
    parser.add_argument("--cloud_path", type=str, default=join("data", "cuboid-sphere.hdf5"))
    parser.add_argument("--camera_dir", nargs="+", type=float, default=[0.5, 0.5, 1.])
    parser.add_argument("--output", type=str, default="output")
    return parser.parse_args()


def run(args):
    logger = logging.getLogger("Task solver")
    logger.setLevel(logging.DEBUG)
    remove_file("solve.log")
    fh = logging.FileHandler("solve.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Starting solver script with the next args:")
    for arg in vars(args):
        logger.info(f"\t{arg}: {getattr(args, arg)}")
    
    force_create_folder(args.output)

    logger.info("Task #1: point cloud analysis")
    raw_data = load_cloud(args.cloud_path)

    cloud = compute_cloud(
        depth_map=raw_data.depth_map,
        intensity_map=raw_data.intensity_map,
        fov=raw_data.fov,
        cam_axis_dir=args.camera_dir,
    )
    visualize_cloud(cloud, savepath=join(args.output, "pointcloud.png"))
    save_cloud_to_h5(cloud, path=join(args.output, "pointcloud.hdf5"))

    cloud_objects = clusterize_cloud(cloud)
    for obj_cloud in cloud_objects:
        if is_cloud_cuboid(obj_cloud):
            label = "cuboid"
            xyz, whl = estimate_cuboid_by_points(obj_cloud)
            area = cuboid_area(whl)
            volume = cuboid_volume(whl)
        else:
            label = "sphere"
            xyz, r = estimate_sphere_by_points(obj_cloud)
            area = sphere_area(r)
            volume = sphere_volume(r)
        visualize_cloud(obj_cloud, savepath=join(args.output, f"{label}.png"))
        save_cloud_to_h5(obj_cloud, path=join(args.output, f"{label}.hdf5"))
        logger.info(f"Detected object: {label}")
        logger.info(f"Centroid [xyz]: {xyz}")
        logger.info(f"Surface area: {area}")
        logger.info(f"Volume: {volume}")


if __name__ == "__main__":
    run(parse_args())
