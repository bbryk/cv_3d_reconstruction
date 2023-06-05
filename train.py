
import torch.nn.functional as F



import torch
import numpy as np

import trimesh
from data_preparation import compute_sdf
import numpy as np
import trimesh
import torch
from skimage import measure
from skimage.measure import marching_cubes
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

# Scale the mesh
# scale_factor = 100  # Adjust this as needed
# mesh.apply_scale(scale_factor)
points = np.array([[0.05, -0.24, 0.03], [0, 0, 0]])

mesh.fill_holes()

cloud = trimesh.points.PointCloud(points)

scene = trimesh.Scene([mesh, cloud])

scene.show()



data = np.load('training_data.npy')

points = data[:, :3]
sdf = data[:, 3]

input_data = torch.from_numpy(points).float()
target_data = torch.from_numpy(sdf).unsqueeze(-1).float()  # unsqueeze(-1) to add an extra dimension


X_train = input_data
y_train = target_data

X_train_tensor = X_train
y_train_tensor = y_train

input_size = 3
hidden_size = 256
output_size = 1
num_layers = 8
model = FFNN(input_size, hidden_size, output_size, num_layers)

loss_fn = torch.nn.L1Loss()  # This is mean absolute error (L1) loss. You may want to adjust this as per your needs.
optimizer = torch.optim.Adam(model.parameters())
losses = []
num_epochs = 500
for epoch in range(num_epochs):
    y_pred = model(X_train_tensor)

    loss = loss_fn(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


import matplotlib.pyplot as plt

plt.plot(losses, marker='o')

# Labeling the axes and providing a title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')

# Display the plot
plt.show()
torch.save(model.state_dict(), 'FFNN.pth')

print("Model has been saved.")


model_trained = FFNN(input_size, hidden_size, output_size, num_layers)
model_trained.load_state_dict(torch.load('FFNN.pth'))

import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model_trained, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 'quantized_FFNN.pth')



model_trained = FFNN(input_size, hidden_size, output_size, num_layers)
model_trained.load_state_dict(torch.load('FFNN.pth'))

model = FFNN(input_size=3, hidden_size=256, output_size=1, num_layers=8)
model.load_state_dict(torch.load('FFNN.pth'))




#The code below reconstruct 3D object using predicted SDF

grid_size = 50
x = np.linspace(-1.5, 1.5, grid_size)
y = np.linspace(-1.5, 1.5, grid_size)
z = np.linspace(-1.5, 1.5, grid_size)
X, Y, Z = np.meshgrid(x, y, z)

points_to_predict = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)


sdf_values = compute_sdf(mesh, torch.from_numpy(points_to_predict).float()).reshape(X.shape)

min_sdf = np.min(sdf_values)
max_sdf = np.max(sdf_values)


level = 0.0 if min_sdf < 0 and max_sdf > 0 else max_sdf
verts, faces, _, _ = marching_cubes(sdf_values, level, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))

mesh = trimesh.Trimesh(vertices=verts, faces=faces)

mesh.show()