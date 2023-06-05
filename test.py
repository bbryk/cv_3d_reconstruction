import numpy as np
# data = np.load('training_data.npy')
#
# # Separate points and SDF values
# points = data[:, :3]
# sdf = data[:, 3]
#
# print(points)
# print(sdf)
#
# a = data[data<0]
# print(len(a))

import torch
import torch.nn as nn
import torch.nn.functional as F



import torch

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


######################################################################












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
filename = 'test_task_meshes/31.obj'

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


#####################################################################

data = np.load('31training_data.npy')

# Separate points and SDF values
points = data[:, :3]
sdf = data[:, 3]

# Convert to PyTorch tensors
input_data = torch.from_numpy(points).float()
target_data = torch.from_numpy(sdf).unsqueeze(-1).float()  # unsqueeze(-1) to add an extra dimension

# Initialize the model and the optimizer
# Assume you have a training dataset with input features 'X_train' and targets 'y_train'
# Assume these are numpy arrays. You would typically load these from your .npy file.
X_train = input_data  # example random data, replace with your data
y_train = target_data  # example random data, replace with your data

# Convert your dataset to PyTorch tensors
X_train_tensor = X_train
y_train_tensor = y_train

input_size = 3
hidden_size = 256
output_size = 1
num_layers = 8
# model = FFNN(input_size, hidden_size, output_size, num_layers)
#
# loss_fn = torch.nn.L1Loss()  # This is mean absolute error (L1) loss. You may want to adjust this as per your needs.
# optimizer = torch.optim.Adam(model.parameters())
# losses = []
# num_epochs = 500
# for epoch in range(num_epochs):
#     y_pred = model(X_train_tensor)
#
#     loss = loss_fn(y_pred, y_train_tensor)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
#
#
# import matplotlib.pyplot as plt
#
# plt.plot(losses, marker='o')
#
# # Labeling the axes and providing a title
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss per Epoch')
#
# # Display the plot
# plt.show()
# torch.save(model.state_dict(), '31FFNN.pth')
#
# print("Model has been saved.")
# #
#
# # print(model([-100, 30, 120]))
#
# model_trained = FFNN(input_size, hidden_size, output_size, num_layers)
# model_trained.load_state_dict(torch.load('31FFNN.pth'))
#
# import torch.quantization
#
# # Quantize the model
# quantized_model = torch.quantization.quantize_dynamic(
#     model_trained, {torch.nn.Linear}, dtype=torch.qint8
# )
#
# # Save the quantized model
# torch.save(quantized_model.state_dict(), 'quantized_31FFNN.pth')


# point = np.array([0, 0, 0])
# # point = np.array([0,-0.3,0.05])
# point_tensor = torch.from_numpy(point).float().unsqueeze(0)
#
# with torch.no_grad():  # We don't need gradients for prediction
#     print(model_trained(point_tensor))

model_trained = FFNN(input_size, hidden_size, output_size, num_layers)
model_trained.load_state_dict(torch.load('31FFNN.pth'))
import numpy as np
import trimesh
import torch
from skimage import measure
from skimage.measure import marching_cubes
# Load your trained model
model = FFNN(input_size=3, hidden_size=256, output_size=1, num_layers=8)
model.load_state_dict(torch.load('31FFNN.pth'))

# Generate a 3D grid of points
grid_size = 50  # The number of points in each dimension of the grid
x = np.linspace(-1.5, 1.5, grid_size)  # Adjust these values to match the space your object occupies
y = np.linspace(-1.5, 1.5, grid_size)
z = np.linspace(-1.5, 1.5, grid_size)
X, Y, Z = np.meshgrid(x, y, z)

# Evaluate the SDF at each point in the grid
points_to_predict = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)



# # Predict the SDF values for all points
# sdf_values = model(torch.from_numpy(points_to_predict).float()).detach().numpy().reshape(X.shape)
#
# # Use Marching Cubes algorithm to create a surface mesh
# verts, faces, _, _ = marching_cubes(sdf_values, 0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
#
# # Create a Trimesh object from the vertices and faces
# mesh = trimesh.Trimesh(vertices=verts, faces=faces)
#
# # Visualize the mesh
# mesh.show()

# Predict the SDF values for all points
# sdf_values = compute_sdf(mesh, torch.from_numpy(points_to_predict).float()).detach().numpy().reshape(X.shape)
sdf_values = compute_sdf(mesh, torch.from_numpy(points_to_predict).float()).reshape(X.shape)

# Find the min and max SDF values
min_sdf = np.min(sdf_values)
max_sdf = np.max(sdf_values)

# Use Marching Cubes algorithm to create a surface mesh
# Use an appropriate level value
level = 0.0 if min_sdf < 0 and max_sdf > 0 else max_sdf
verts, faces, _, _ = marching_cubes(sdf_values, level, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))

# Create a Trimesh object from the vertices and faces
mesh = trimesh.Trimesh(vertices=verts, faces=faces)

# Visualize the mesh
mesh.show()