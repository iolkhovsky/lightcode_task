import argparse
from os.path import join
import logging

from numpy import save

from utils.image import *
from utils.cloud import *
from utils.common import *
from utils.visualization import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test task solver')
    parser.add_argument("--cloud_path", type=str, default=join("data", "cuboid-sphere.hdf5"),
        help="Absolute path to the H5 file with depth & intensity maps")
    parser.add_argument("--camera_dir", nargs="+", type=float, default=[0.5, 0.5, 1.],
        help="List of 3 float numbers representing Camera axis direction in world coordinates")
    parser.add_argument("--image_path", type=str, default=join("data", "cuboid-sphere.png"),
        help="Absolute path to the png file with scene image")
    parser.add_argument("--output", type=str, default="output",
        help="Absolute path to the output folder (for generated artifacts)")
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
    visualize_cloud(cloud, savepath=join(args.output, "pointcloud.png"), title="Full cloud")
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
        visualize_cloud(obj_cloud, savepath=join(args.output, f"{label}.png"), title=label)
        save_cloud_to_h5(obj_cloud, path=join(args.output, f"{label}.hdf5"))
        logger.info(f"Detected object: {label}")
        logger.info(f"Centroid [xyz]: {xyz}")
        logger.info(f"Surface area: {area}")
        logger.info(f"Volume: {volume}")
    
    logger.info("Task #2: image analysis")
    img = read_image(args.image_path)
    circles = detect_circles(img)
    assert len(circles) == 1
    circle_face_area = circle_area(circles[0])
    logger.info(f"Sphere main face area: {circle_face_area}")
    processed_img = draw_circles(img, circles)
    cuboid_face, cuboid_face_area = detect_cuboid_face(img)
    logger.info(f"Cuboid main face area: {cuboid_face_area}")
    processed_img = draw_contours(processed_img, cuboid_face)
    write_image(processed_img, join(args.output, "processed.jpg"))


if __name__ == "__main__":
    run(parse_args())
