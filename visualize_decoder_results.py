import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
# Import your updated model
from WholeGraspPose.models.models_handweights import FullBodyGraspNet
from utils.train_helper import point2point_signed


cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
with open(cfg_path, 'r') as f:
    cfg_dict = yaml.safe_load(f)

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

cfg = Config(cfg_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'logs/GraspPose/partbased_embedding_02/snapshots/TR00_E040_net.pt'
model = FullBodyGraspNet(cfg)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

data = np.load('dataset/GraspPose/train/s1/bowl_pass_1.npz', allow_pickle=True)

markers = data['markers_143'][0:1]           # Shape: (1, 143, 3)
transf_transl = data['transf_transl'][0:1]   # Shape: (1, 3)
verts_object = data['verts_object'][0:1]     # Shape: (1, 2048, 3)
contacts_object = data['contact_object'][0:1] # Shape: (1, 2048)
contacts_markers = data['contact_body'][0:1] # Shape: (1, 143)
normal_object = data['normal_object'][0:1]   # Shape: (1, 2048, 3)
global_orient_object = data['global_orient_object'][0:1] # Shape: (1, 3)

from smplx.lbs import batch_rodrigues

rotmat = batch_rodrigues(torch.tensor(global_orient_object).view(-1, 3)).view([global_orient_object.shape[0], 9]).numpy()

markers = torch.tensor(markers).float().to(device)               # Shape: (1, 143, 3)
transf_transl = torch.tensor(transf_transl).float().to(device)   # Shape: (1, 3)
verts_object = torch.tensor(verts_object).float().to(device)     # Shape: (1, 2048, 3)
contacts_object = torch.tensor(contacts_object).float().to(device) # Shape: (1, 2048)
contacts_markers = torch.tensor(contacts_markers).float().to(device) # Shape: (1, 143)
normal_object = torch.tensor(normal_object).float().to(device)   # Shape: (1, 2048, 3)
rotmat = torch.tensor(rotmat).float().to(device)                 # Shape: (1, 9)


# Repeat rotation matrix to match the number of object vertices
rotmat_expanded = rotmat[:, :6].unsqueeze(1).repeat(1, verts_object.shape[1], 1)  # Shape: (1, 2048, 6)

# Concatenate normals and rotation matrices
feat_object = torch.cat([normal_object, rotmat_expanded], dim=-1)  # Shape: (1, 2048, 9)

# Permute to match expected input shape: (batch_size, 3, num_points)
verts_object = verts_object.permute(0, 2, 1)      # Shape: (1, 3, 2048)
feat_object = feat_object.permute(0, 2, 1)        # Shape: (1, 9, 2048)

contacts_object = contacts_object.unsqueeze(1)  # Shape: (1, 1, 2048)

# Calculate marker-object distance (example calculation, adjust as needed)
# Compute marker_object_distance using point2point_signed
_, marker_object_distance, _, _ = point2point_signed(
    x=markers,                    # Shape: (N, 143, 3)
    y=verts_object,               # Shape: (N, 2048, 3)
    transform_distances=True
)
# marker_object_distance shape: (N, 143)

# Ensure marker_object_distance is on the same device
marker_object_distance = marker_object_distance.to(device)


with torch.no_grad():
    # Prepare the object condition using the pointnet encoder
    object_cond = model.pointnet(l0_xyz=verts_object, l0_points=feat_object)

    # Encode to get the distribution over z
    z_dist = model.encode(
        object_cond=object_cond,
        verts_object=verts_object,
        feat_object=feat_object,
        contacts_object=contacts_object,
        markers=markers,
        contacts_markers=contacts_markers,
        transf_transl=transf_transl,
        marker_object_distance=marker_object_distance
    )

    # Sample z
    z_sample = z_dist.rsample()

    # Use the marker_net decoder
    markers_xyz_pred, markers_p_pred = model.marker_net.dec(
        Z=z_sample,
        cond_object=object_cond,
        transf_transl=transf_transl
    )

# Move tensors to CPU for visualization
markers_xyz_pred_cpu = markers_xyz_pred.cpu().numpy()[0]  # Shape: (143, 3)
markers_p_pred_cpu = markers_p_pred.cpu().numpy()[0]      # Shape: (143,)

# Plot the predicted markers
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot markers with colors based on predicted probabilities
sc = ax.scatter(
    markers_xyz_pred_cpu[:, 0],
    markers_xyz_pred_cpu[:, 1],
    markers_xyz_pred_cpu[:, 2],
    c=markers_p_pred_cpu, cmap='jet', s=20
)

# Optionally, plot the object vertices for reference
verts_object_cpu = verts_object.cpu().numpy()[0].T  # Shape: (2048, 3)
ax.scatter(
    verts_object_cpu[:, 0],
    verts_object_cpu[:, 1],
    verts_object_cpu[:, 2],
    color='gray', s=1, alpha=0.5
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Predicted Markers from MarkerNet Decoder')

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Marker Probabilities', rotation=270, labelpad=15)

plt.show()
