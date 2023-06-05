import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

import torch


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

    return -1 * distances


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


# Define loss function
def deep_sdf_loss(pred_sdf, gt_sdf, delta):
    clamp_pred_sdf = torch.clamp(pred_sdf, -delta, delta)
    clamp_gt_sdf = torch.clamp(gt_sdf, -delta, delta)
    return F.l1_loss(clamp_pred_sdf, clamp_gt_sdf)


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

    return -1 * distances


# import pytorch3d
filename = 'test_task_meshes/30.obj'

# Load a mesh
mesh = trimesh.load_mesh(filename)

# Scale the mesh
# scale_factor = 100  # Adjust this as needed
# mesh.apply_scale(scale_factor)
points = np.array([[0.05, -0.24, 0.03], [0, 0, 0]])

mesh.fill_holes()

# Create a point cloudS
cloud = trimesh.points.PointCloud(points)

# Create a scene with the mesh and the point cloud
scene = trimesh.Scene([mesh, cloud])

# Show the scene
scene.show()

print(mesh.bounds)

sdf_real = compute_sdf(mesh, points)
print(f"Ground Truth SDF: {sdf_real}")

input_size = 3
hidden_size = 256
output_size = 1
num_layers = 8

model_trained = FFNN(input_size, hidden_size, output_size, num_layers)
model_trained.load_state_dict(torch.load('FFNN_v5_new_data.pth'))

# import torch.quantization
#
# # Define the model architecture
# model = FFNN(input_size, hidden_size, output_size, num_layers)
#
# # Quantize the model
# quantized_model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )
#
# # Load the state dict into the quantized model
# quantized_model.load_state_dict(torch.load('quantized_FFNN_v4.pth'))

# First, compute the area of each face in the mesh
# First, compute the area of each face in the mesh

def sample():
    areas = mesh.area_faces

    # Create a copy of areas
    areas = areas.copy()

    # Next, normalize the areas so they sum to 1
    areas /= areas.sum()

    # Continue with the rest of the code...

    # Decide on how many points you want to sample
    num_points = 1000

    # Sample face indices using the areas as probabilities
    chosen_faces = np.random.choice(len(mesh.faces), size=num_points, p=areas)

    # For each chosen face, sample a random point
    points = np.zeros((num_points, 3))

    for i, face_index in enumerate(chosen_faces):
        # Get the vertices of the face
        vertices = mesh.vertices[mesh.faces[face_index]]

        # Compute the barycentric coordinates of a random point in the triangle
        # Barycentric coordinates are a form of homogeneous coordinates used to
        # interpolate across triangles.
        u = np.random.uniform()
        v = np.random.uniform()
        if u + v > 1:  # we're outside the triangle, re-sample
            u = 1 - u
            v = 1 - v
        w = 1 - u - v

        # Compute the coordinates of the random point
        point = u * vertices[0] + v * vertices[1] + w * vertices[2]
        points[i] = point

    # Add noise to the points
    noise = np.random.normal(scale=0.02, size=points.shape)
    points += noise

    print(points)

import copy

from scipy.spatial import Delaunay

# Find the bounding box of the object
min_bound, max_bound = mesh.bounds
max_bound = copy.copy(max_bound)
min_bound = copy.copy(min_bound)


# Compute the size of the bounding box
size = max_bound - min_bound

# Decide on the density of points you want to sample
density = 10000  # This is the number of points per unit volume you want to sample

# Compute the total number of points
num_points = int(np.prod(size) * density)

# Sample points in the bounding box
points = np.random.uniform(low=min_bound, high=max_bound, size=(num_points, 3))

# Create a Delaunay triangulation of the mesh vertices
tri = Delaunay(mesh.vertices)

# Use the Delaunay triangulation to perform a point-in-mesh test
mask = tri.find_simplex(points) >= 0

# Keep only the points that are inside the mesh
points_inside = points[mask]

# print(points_inside)
points = points_inside
# data = np.load('training_data_updated.npy')
#
# # Separate points and SDF values
# points = data[:, :3]
# sdf = data[:, 3]
#
#
# point = np.array([0.05, -0.24, 0.03])
# # point = np.array([0,-0.3,0.05])
# point_tensor = torch.from_numpy(point).float().unsqueeze(0)
#
#
#
real_sdf = []
predicted_sdf = []
c = 0
for i in range(len(points)):
    point = np.array(points[i])
    point_tensor = torch.from_numpy(point).float().unsqueeze(0)
    # with torch.no_grad():  # We don't need gradients for prediction
    #     print(model_trained(point_tensor).item())
    # print(compute_sdf(mesh, [points[i]]))
    # print("\n\n")
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

# print(real_labels[:100].flatten())
# print(predicted_labels[:100])
# inc = 0
# for i in range(len(real_labels)):
#     if real_labels[i] != predicted_labels[i]:
#         print(i)
#         inc+=1
# print(i)
