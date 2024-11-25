import numpy as np
import torch
import pickle
import sys
import os

# Add two levels up to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from utils.train_helper import point2point_signed

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of objects to process
objects = ["toothpaste", "camera", "wineglass", "binoculars", "mug"]

rhand_verts_indices = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142]

# Directory template for loading marker results
datadirectory_template = '../../results/testdump/GraspPose/{object}/markers_results.npy'

# Process objects
for obj in objects:
    try:
        datadirectory = datadirectory_template.format(object=obj)
        print(f"Loading data for {obj} from {datadirectory}")
        samples_results = np.load(datadirectory, allow_pickle=True).item()

        print(f"keys in samples_results: {list(samples_results.keys())}")
        markers_gen = torch.tensor(samples_results['markers_gen']).to(device)
        verts_object = torch.tensor(samples_results['verts_object']).to(device)
        normals_object = torch.tensor(samples_results['normal_object']).to(device)

        # Check for NaNs
        assert not torch.isnan(markers_gen).any(), "markers_gen contains NaNs!"
        assert not torch.isnan(verts_object).any(), "verts_object contains NaNs!"
        assert not torch.isnan(normals_object).any(), "normals_object contains NaNs!"

        # Check indexing
        assert all(0 <= idx < markers_gen.size(1) for idx in rhand_verts_indices), "rhand_verts_indices out of bounds!"

        # Initialize variables for averaging
        total_interpenetration_depth = 0.0
        total_contact_points = 0
        num_samples = len(markers_gen)

        # Process each sample
        for sample_idx in range(num_samples):
            try:
                # Use all body markers
                rhand_verts = markers_gen[sample_idx, :, :]  # Shape: [143, 3]
                verts_object_sample = verts_object[sample_idx]  # Shape: [2048, 3]
                normals_object_sample = normals_object[sample_idx]  # Shape: [2048, 3]

                # Add batch dimension
                rhand_verts = rhand_verts.unsqueeze(0)  # Shape: [1, 143, 3]
                verts_object_sample = verts_object_sample.unsqueeze(0)  # Shape: [1, 2048, 3]
                normals_object_sample = normals_object_sample.unsqueeze(0)  # Shape: [1, 2048, 3]
                o2h_signed, h2o_signed, o2h_idx, h2o_idx = point2point_signed(
                    rhand_verts, verts_object_sample, None, normals_object_sample
                )

                # Calculate interpenetration depth
                interpenetration_depth_cm = torch.sum(torch.abs(h2o_signed[h2o_signed < 0])) * 100
                total_interpenetration_depth += interpenetration_depth_cm.item()

                # Calculate contact points
                contact = torch.where((h2o_signed < 0.001) * (h2o_signed > -0.001) == True)[0].size()[0]
                total_contact_points += contact

                print(f"Sample {sample_idx + 1}: Interpenetration depth = {interpenetration_depth_cm:.2f} cm")
                print(f"Sample {sample_idx + 1}: Contact points = {contact}")

            except Exception as e:
                print(f"Error processing sample {sample_idx + 1}: {e}")
                continue

        # Compute and print the averages
        if num_samples > 0:
            avg_interpenetration_depth = total_interpenetration_depth / num_samples
            avg_contact_points = total_contact_points / num_samples
            print(f"\nAverage interpenetration depth for {obj}: {avg_interpenetration_depth:.2f} cm")
            print(f"Average contact points for {obj}: {avg_contact_points:.2f}\n")
        else:
            print(f"\nNo valid samples for object {obj}.\n")

    except FileNotFoundError:
        print(f"File not found for object: {obj}")
    except Exception as e:
        print(f"An error occurred for object {obj}: {e}")
