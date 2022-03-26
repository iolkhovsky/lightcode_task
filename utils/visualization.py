import matplotlib.pyplot as plt
import numpy as np


def visualize_cloud(cloud, color="red", savepath=None):
    fig = plt.figure(figsize=(10, 6)) 
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=color, alpha=0.5, s=0.25) 

    x_min, x_max = np.min(cloud[:, 0]), np.max(cloud[:, 0])
    xc = 0.5 * (x_max + x_min)
    y_min, y_max = np.min(cloud[:, 1]), np.max(cloud[:, 1])
    yc = 0.5 * (y_max + y_min)
    z_min, z_max = np.min(cloud[:, 2]), np.max(cloud[:, 2])
    zc = 0.5 * (z_max + z_min)

    vis_cube_half_side = max(
        (x_max - x_min),
        (y_max - y_min),
        (z_max - z_min),
    ) * 1.2 / 2

    ax.set_xlim([xc - vis_cube_half_side, xc + vis_cube_half_side])
    ax.set_ylim([yc - vis_cube_half_side, yc + vis_cube_half_side])
    ax.set_zlim([zc - vis_cube_half_side, zc + vis_cube_half_side])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
