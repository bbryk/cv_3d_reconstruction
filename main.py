import pyvista as pv
from pyvista import examples

import torch
import torchvision
# from trimesh.ray.ray_pyembree import RayMeshIntersector
import numpy as np

import trimesh

def compute_sdf(mesh, points):
    # For each point, we'll cast a ray in an arbitrary direction (e.g., towards the +z direction)
    # and count the number of intersections
    directions = np.tile([0, 0, 1], (len(points), 1))  # Rays towards the +z direction

    # Perform the ray-mesh intersection test
    intersections = mesh.ray.intersects_first(ray_origins=points, ray_directions=directions)

    # Points with no intersections are inside the mesh
    inside = np.isnan(intersections)

    # Compute distances from points to mesh
    distances = mesh.nearest.signed_distance(points)

    # Flip signs for points inside the mesh
    distances[inside] *= -1

    return -1*distances
# import pytorch3d
filename = 'test_task_meshes/30.obj'




# Load a mesh
mesh = trimesh.load_mesh(filename)


# Scale the mesh
# scale_factor = 100  # Adjust this as needed
# mesh.apply_scale(scale_factor)
points = np.array([[0,0,80], [0,-110,8]])

mesh.fill_holes()
print(mesh.bounds)
# sdf = compute_sdf(mesh, points)
# print(sdf)
#
# # Create a point cloud
# cloud = trimesh.points.PointCloud(points)
#
# # Create a scene with the mesh and the point cloud
# scene = trimesh.Scene([mesh, cloud])
#
# # Show the scene
# scene.show()















import itertools
import os
def generate_points_batch(x_range, y_range, z_range, step_size, batch_size):
    # Create the grid
    x = np.arange(*x_range, step_size)
    y = np.arange(*y_range, step_size)
    z = np.arange(*z_range, step_size)

    # Create a generator for all combinations of x, y, z
    all_points = itertools.product(x, y, z)

    while True:
        # Generate a batch of points
        batch_points = []
        for _ in range(batch_size):
            try:
                point = next(all_points)
                batch_points.append(point)
            except StopIteration:
                break

        if not batch_points:
            break

        yield np.array(batch_points)


x_range = (-0.3, 0.3)
y_range = (-1.5, 0.55)
z_range = (-1.5, 1.5)
step_size = 0.02
batch_size = 1000

i = 0

training_data_list = []
for points_batch in generate_points_batch(x_range, y_range, z_range, step_size, batch_size):
    i+=1
    print(i)
    sdf_batch = compute_sdf(mesh, points_batch)

    training_data_batch = np.concatenate([points_batch, sdf_batch[:, None]], axis=1)

    training_data_list.append(training_data_batch)

training_data = np.concatenate(training_data_list, axis=0)

np.save(os.path.join("", '31training_data.npy'), training_data)

print("Training data has been saved.")
