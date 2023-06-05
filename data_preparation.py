import pyvista as pv
from pyvista import examples

import torch
import torchvision
import numpy as np

import trimesh

import itertools
import os
def compute_sdf(mesh, points):

    directions = np.tile([0, 0, 1], (len(points), 1))  # Rays towards the +z direction

    intersections = mesh.ray.intersects_first(ray_origins=points, ray_directions=directions)

    inside = np.isnan(intersections)

    distances = mesh.nearest.signed_distance(points)

    distances[inside] *= -1

    return -1*distances
filename = 'test_task_meshes/30.obj'



mesh = trimesh.load_mesh(filename)
mesh.fill_holes()















def generate_points_batch(x_range, y_range, z_range, step_size, batch_size):
    x = np.arange(*x_range, step_size)
    y = np.arange(*y_range, step_size)
    z = np.arange(*z_range, step_size)

    all_points = itertools.product(x, y, z)

    while True:
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

np.save(os.path.join("", 'training_data.npy'), training_data)

print("Training data has been saved.")
