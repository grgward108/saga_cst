import torch
import numpy as np
import json
from smplx.lbs import batch_rodrigues
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from matplotlib import cm

# Load configuration from YAML file
cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
with open(cfg_path, 'r') as f:
    cfg_dict = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load sample data
data = np.load('dataset/GraspPose/train/s1/banana_eat_1.npz', allow_pickle=True)

# Extract relevant fields (use only the first frame)
markers = data['markers_143'][0:1]  # Shape: (1, 143, 3) for the first frame
transf_transl = data['transf_transl'][0:1]  # Shape: (1, 3)
verts_object = data['verts_object'][0:1]  # Shape: (1, 2048, 3)
contacts_object = data['contact_object'][0:1]  # Shape: (1, 2048)
contacts_markers = data['contact_body'][0:1, :]  # Shape: (1, 143)
normal_object = data['normal_object'][0:1]  # Shape: (1, 2048, 3)
global_orient_object = data['global_orient_object'][0:1]  # Shape: (1, 3)
rotmat = batch_rodrigues(torch.tensor(global_orient_object).view(-1, 3)).view([global_orient_object.shape[0], 9]).numpy()

# Convert to torch tensors and move to device
markers = torch.tensor(markers).float().to(device)
transf_transl = torch.tensor(transf_transl).float().to(device)
verts_object = torch.tensor(verts_object).float().to(device)
contacts_object = torch.tensor(contacts_object).float().to(device)
contacts_markers = torch.tensor(contacts_markers).float().to(device)
normal_object = torch.tensor(normal_object).float().to(device)
rotmat = torch.tensor(rotmat).float().to(device)

# Move tensors to CPU for plotting
markers_cpu = markers.cpu().numpy()
contacts_markers_cpu = contacts_markers.cpu().numpy()
transf_transl_cpu = transf_transl.cpu().numpy()
verts_object_cpu = verts_object.cpu().numpy()  # Convert object vertices to CPU

# Compute feat_object for the first frame
feat_object = torch.cat([normal_object, rotmat[:, :6].view(-1, 1, 6).repeat(1, verts_object.shape[1], 1)], dim=-1)

# Load markerset indices and compute part_labels
with open('body_utils/smplx_markerset.json') as f:
    markerset = json.load(f)['markersets']
    markers_idx = []
    for marker in markerset:
        if marker['type'] not in ['palm_5']:
            markers_idx += list(marker['indices'].values())

# Map markers to parts
body_part_groups = {
    'head_and_neck': [2819, 3076, 1795, 2311, 1043, 919, 8985, 1696, 1703, 9002, 8757, 2383, 2898, 3035, 2148, 9066, 8947, 2041, 2813],
    'trunk': [4391, 4297, 5615, 5944, 5532, 5533, 5678, 7145],
    'right_upper_limb': [7179, 7028, 7115, 7251, 7274, 7293, 6778, 7036],
    'left_upper_limb': [4509, 4245, 4379, 4515, 4538, 4557, 4039, 4258],
    'right_hand': [8001, 7781, 7750, 7978, 7756, 7884, 7500, 7419, 7984, 7633, 7602, 7667, 7860, 8082, 7351, 7611, 7867, 7423, 7357, 7396, 7443, 7446, 7536, 7589, 7618, 7625, 7692, 7706, 7730, 7748, 7789, 7847, 7858, 7924, 7931, 7976, 8039, 8050, 8087, 8122],
    'left_hand': [4897, 5250, 4931, 5124, 5346, 4615, 5321, 4875, 5131, 4683, 4686, 4748, 5268, 5045, 5014, 5242, 5020, 5149, 4628, 4641, 4660, 4690, 4691, 4710, 4750, 4885, 4957, 4970, 5001, 5012, 5082, 5111, 5179, 5193, 5229, 5296, 5306, 5315, 5353, 5387],
    'left_legs': [5857, 5893, 5899, 3479, 3781, 3638, 3705, 5761, 8852, 5726],
    'right_legs': [8551, 8587, 8593, 6352, 6539, 6401, 6466, 8455, 8634, 8421],
}

def map_marker_to_part(marker_indices):
    part_labels = np.zeros(len(marker_indices))
    for part_index, (part_name, part_indices) in enumerate(body_part_groups.items()):
        for marker_idx in part_indices:
            if marker_idx in marker_indices:
                mapped_index = marker_indices.index(marker_idx)
                part_labels[mapped_index] = part_index + 1  # Labels start from 1
    return part_labels

part_labels = map_marker_to_part(markers_idx)
part_labels = torch.tensor(part_labels).long().unsqueeze(0).to(device)

# Instantiate the model
from WholeGraspPose.models.models_handweights import FullBodyGraspNet

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

cfg = Config(cfg_dict)

model = FullBodyGraspNet(cfg)

# Load the checkpoint
model_path = 'logs/GraspPose/cross_attention_hand_weights/snapshots/TR00_E005_net.pt'
checkpoint = torch.load(model_path, map_location=device)
filtered_state_dict = {k: v for k, v in checkpoint.items() if 'marker_net' in k}

# Load the filtered state dictionary into the model
model.load_state_dict(filtered_state_dict, strict=False)
model = model.to(device)
model.eval()

# Prepare object_cond using the pointnet encoder for the first frame
l0_xyz = verts_object.permute(0, 2, 1)  # (1, 3, 2048)
l0_points = feat_object.permute(0, 2, 1)  # (1, 64, 2048)
object_cond = model.pointnet(l0_xyz=l0_xyz, l0_points=l0_points)

# Call marker_net.enc to get attention weights (for the first frame)
with torch.no_grad():
    X, attn_weights = model.marker_net.enc(
        cond_object=object_cond,
        markers=markers,
        contacts_markers=contacts_markers,
        transf_transl=transf_transl,
        part_labels=part_labels,
        return_attention=True
    )

# Process attention weights for the first frame
attn_weights_np = attn_weights.squeeze(0).squeeze(0).cpu().detach().numpy()  # Shape: (n_markers,)
part_labels_np = part_labels.squeeze(0).cpu().numpy()  # Shape: (n_markers,)

# Heatmap color mapping
norm = plt.Normalize(vmin=min(attn_weights_np), vmax=max(attn_weights_np))
cmap = plt.colormaps.get_cmap('jet')

# Visualize the first frame of body markers with heatmap
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each marker with its corresponding attention-based color and smaller size
for i in range(143):  # Loop through all markers
    ax.scatter(
        markers_cpu[0, i, 0], markers_cpu[0, i, 1], markers_cpu[0, i, 2], 
        color=cmap(norm(attn_weights_np[i])), 
        s=20,  # Reduced size
        label='Body Markers' if i == 0 else ""  # Label only the first marker for the legend
    )

sc = ax.scatter(
    markers_cpu[0, :, 0], markers_cpu[0, :, 1], markers_cpu[0, :, 2], 
    c=attn_weights_np, cmap='jet', s=20  # Reduced size for markers
)


# Add object vertices with a smaller size and gray color
ax.scatter(
    verts_object_cpu[0, :, 0], verts_object_cpu[0, :, 1], verts_object_cpu[0, :, 2], 
    color='gray', label='Object Vertices', s=1  # Smaller object vertex size
)

cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Attention Weights', rotation=270, labelpad=15)

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Attention Heatmap for Body Markers and Object Vertices (First Frame)')
ax.legend()  # Add the legend
plt.show()
