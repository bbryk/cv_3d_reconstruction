import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

import torch

from data_preparation import compute_sdf

import copy

from scipy.spatial import Delaunay

import trimesh


class FFNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.layers = [
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
        ]
        for i in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)






filename = 'test_task_meshes/30.obj'

mesh = trimesh.load_mesh(filename)

input_size = 3
hidden_size = 256
output_size = 1
num_layers = 8

model_trained = FFNN(input_size, hidden_size, output_size, num_layers)
model_trained.load_state_dict(torch.load('FFNN_v5_new_data.pth'))

def sample_around_surface():
    areas = mesh.area_faces

    areas = areas.copy()

    areas /= areas.sum()


    num_points = 1000

    chosen_faces = np.random.choice(len(mesh.faces), size=num_points, p=areas)

    points = np.zeros((num_points, 3))

    for i, face_index in enumerate(chosen_faces):
        vertices = mesh.vertices[mesh.faces[face_index]]


        u = np.random.uniform()
        v = np.random.uniform()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        w = 1 - u - v

        point = u * vertices[0] + v * vertices[1] + w * vertices[2]
        points[i] = point

    noise = np.random.normal(scale=0.02, size=points.shape)
    points += noise

    return points


def sample_inside():
    min_bound, max_bound = mesh.bounds
    max_bound = copy.copy(max_bound)
    min_bound = copy.copy(min_bound)


    size = max_bound - min_bound

    density = 10000

    num_points = int(np.prod(size) * density)

    points = np.random.uniform(low=min_bound, high=max_bound, size=(num_points, 3))

    tri = Delaunay(mesh.vertices)

    mask = tri.find_simplex(points) >= 0

    points_inside = points[mask]

    points = points_inside
    return  points

points = sample_inside()

real_sdf = []
predicted_sdf = []
c = 0
for i in range(len(points)):
    point = np.array(points[i])
    point_tensor = torch.from_numpy(point).float().unsqueeze(0)

    if np.sign(model_trained(point_tensor).item()) == np.sign(compute_sdf(mesh, [points[i]])):
        c += 1
    real_sdf.append(compute_sdf(mesh, [points[i]]))
    predicted_sdf.append(model_trained(point_tensor).item())

print(f"Accuracy: {c / len(points)}")
real_sdf = np.array(real_sdf)
predicted_sdf = np.array(predicted_sdf)

real_labels = (real_sdf >= 0).astype(int)  # 1 if SDF is positive, 0 if negative
predicted_labels = (predicted_sdf >= 0).astype(int)  # 1 if SDF is positive, 0 if negative

real_labels = real_labels.flatten()
predicted_labels = predicted_labels.flatten()

from sklearn.metrics import f1_score
f1 = f1_score(real_labels, predicted_labels)
print('F1 Score:', f1)


