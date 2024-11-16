import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as filters
import smplx
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.Pivots import Pivots
from utils.Quaternions import Quaternions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRAB_DataLoader(data.Dataset):
    def __init__(self, clip_seconds=8, clip_fps=30, normalize=False, split='train', markers_type=None, is_debug=False, log_dir='', mask_prob=0.15):
        """
        Simplified GRAB DataLoader for body and hand markers.
        """
        self.clip_seconds = clip_seconds
        self.clip_len = clip_seconds * clip_fps + 2  # T+2 frames for each clip
        self.data_dict_list = []
        self.normalize = normalize
        self.clip_fps = clip_fps
        self.split = split  # train/test
        self.is_debug = is_debug
        self.markers_type = markers_type
        self.log_dir = log_dir
        self.mask_prob = mask_prob

        assert self.markers_type is not None

        # Parse marker types (e.g., f0_p5)
        f, p = self.markers_type.split('_')
        finger_n, palm_n = int(f[1:]), int(p[1:])

        # Load marker IDs
        with open('./body_utils/smplx_markerset.json') as f:
            markerset = json.load(f)['markersets']
            self.markers_ids = []
            for marker in markerset:
                if marker['type'] == 'finger' and finger_n == 0:
                    continue
                elif 'palm' in marker['type']:
                    if palm_n == 5 and marker['type'] == 'palm_5':
                        self.markers_ids += list(marker['indices'].values())
                    elif palm_n == 22 and marker['type'] == 'palm':
                        self.markers_ids += list(marker['indices'].values())
                    else:
                        continue
                else:
                    self.markers_ids += list(marker['indices'].values())

        self.segmentation = {
            1: [2819, 3076, 1795, 2311, 1043, 919, 8985, 1696, 1703, 9002, 8757, 2383, 2898, 3035, 2148, 9066, 8947, 2041, 2813],
            2: [4391, 4297, 5615, 5944, 5532, 5533, 5678, 7145],
            3: [7179, 7028, 7115, 7251, 7274, 7293, 6778, 7036],
            4: [4509, 4245, 4379, 4515, 4538, 4557, 4039, 4258],
            5: [8001, 7781, 7750, 7978, 7756, 7884, 7500, 7419, 7984, 7633, 7602, 7667, 7860, 8082, 7351, 7611, 7867, 7423, 7357, 7396, 7443, 7446, 7536, 7589, 7618, 7625, 7692, 7706, 7730, 7748, 7789, 7847, 7858, 7924, 7931, 7976, 8039, 8050, 8087, 8122],
            6: [4897, 5250, 4931, 5124, 5346, 4615, 5321, 4875, 5131, 4683, 4686, 4748, 5268, 5045, 5014, 5242, 5020, 5149, 4628, 4641, 4660, 4690, 4691, 4710, 4750, 4885, 4957, 4970, 5001, 5012, 5082, 5111, 5179, 5193, 5229, 5296, 5306, 5315, 5353, 5387],
            7: [5857, 5893, 5899, 3479, 3781, 3638, 3705, 5761, 8852, 5726],
            8: [8551, 8587, 8593, 6352, 6539, 6401, 6466, 8455, 8634, 8421],
        }

    def map_markers_to_parts(self, markers_ids):
        """
        Map marker indices to part labels based on segmentation.
        """
        part_labels = np.zeros(len(markers_ids), dtype=int)
        for part, indices in self.segmentation.items():
            for marker_id in indices:
                if marker_id in markers_ids:
                    part_labels[markers_ids.index(marker_id)] = part
        return part_labels

    def divide_clip(self, dataset_name='GraspMotion', data_dir=None):
        npz_fnames = sorted(glob.glob(os.path.join(data_dir, dataset_name) + '/*.npz'))
        for npz_fname in npz_fnames:
            cdata = np.load(npz_fname, allow_pickle=True)
            fps = int(cdata['framerate'])

            # Set sampling rate based on FPS
            if fps == 150:
                sample_rate = 5
            elif fps == 120:
                sample_rate = 4
            elif fps == 60:
                sample_rate = 2
            else:
                continue

            clip_len = self.clip_seconds * fps + sample_rate + 1
            N = cdata['n_frames']

            # Adjust sequence length
            if N >= clip_len:
                seq_transl = cdata['body'][()]['params']['transl']
            else:
                diff = clip_len - N
                seq_transl = np.concatenate([np.repeat(cdata['body'][()]['params']['transl'][0].reshape(1, -1), diff, axis=0), cdata['body'][()]['params']['transl']], axis=0)

            data_dict = {
                'body': {
                    'transl': seq_transl[-clip_len:][::sample_rate],
                },
                'gender': str(cdata['gender']),
                'framerate': fps
            }
            self.data_dict_list.append(data_dict)

    def read_data(self, datasets, data_dir):
        for dataset_name in tqdm(datasets):
            self.divide_clip(dataset_name, data_dir)
        self.n_samples = len(self.data_dict_list)
        print(f'[INFO] Loaded {self.n_samples} samples.')


    def create_body_repr(self, body_params, markers_ids, smplx_model_path, gender):
        """
        Create body representation by passing body parameters through the SMPL-X model.
        """
        bs = body_params['transl'].shape[0]

        # Ensure 'betas' is correctly shaped
        if 'betas' in body_params and body_params['betas'].shape[0] != bs:
            body_params['betas'] = np.repeat(body_params['betas'], bs, axis=0)
        elif 'betas' not in body_params:
            body_params['betas'] = np.zeros((bs, 10))

        for param_name in body_params:
            body_params[param_name] = torch.from_numpy(body_params[param_name]).float().to(device)
            # Uncomment for debugging:
            # print(f"{param_name} shape: {body_params[param_name].shape}")

        # Initialize SMPL-X model with use_pca=True and num_pca_comps=24
        body_model = smplx.create(
            model_path=smplx_model_path,
            model_type='smplx',
            gender=gender,
            batch_size=bs,
            use_pca=True,       # Use PCA for hand poses
            num_pca_comps=24    # Match the number of PCA components in your data
        ).to(device)

        with torch.no_grad():
            smplx_output = body_model(return_verts=True, **body_params)
            markers = smplx_output.vertices[:, markers_ids, :]  # [T, N_markers, 3]

        return markers.cpu().numpy()


    def create_body_hand_repr(self, smplx_model_path):
        """
        Generate normalized marker positions and part labels for body and hand.
        """
        self.clip_img_list = []
        self.part_labels_list = []  # To store part labels for each sample

        for i in tqdm(range(self.n_samples)):
            body_params = self.data_dict_list[i]['body']
            gender = self.data_dict_list[i]['gender']  # Get gender for the sample

            # Ensure 'betas' are present in body_params
            if 'betas' not in body_params:
                if 'betas' in self.data_dict_list[i]:
                    body_params['betas'] = self.data_dict_list[i]['betas']
                else:
                    # Use zeros if betas are not available
                    bs = body_params['transl'].shape[0]
                    body_params['betas'] = np.zeros((bs, 10))

            # Compute markers using SMPL-X model
            markers = self.create_body_repr(body_params, self.markers_ids, smplx_model_path, gender)

            # Normalize markers
            markers_mean = markers.mean(axis=(0, 1), keepdims=True)
            markers_std = markers.std(axis=(0, 1), keepdims=True)
            normalized_markers = (markers - markers_mean) / (markers_std + 1e-8)

            # Map markers to part labels
            part_labels = self.map_markers_to_parts(self.markers_ids)

            # Store normalized data and part labels
            self.clip_img_list.append(normalized_markers)  # [seq_len, N_markers, 3]
            self.part_labels_list.append(part_labels)

        self.clip_img_list = np.asarray(self.clip_img_list)  # [N_samples, seq_len, N_markers, 3]
        self.part_labels_list = np.asarray(self.part_labels_list)  # [N_samples, N_markers]



    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        clip_img = self.clip_img_list[index]  # [seq_len, N_markers, 3]
        part_labels = self.part_labels_list[index]  # [N_markers]

        # Convert to tensor
        clip_img = torch.from_numpy(clip_img).float()  # [seq_len, N_markers, 3]
        part_labels = torch.from_numpy(part_labels).long()  # [N_markers]

        # Apply random masking
        mask = torch.rand(clip_img.shape[:2]) < self.mask_prob  # [seq_len, N_markers]
        mask = mask.unsqueeze(-1).expand_as(clip_img)  # [seq_len, N_markers, 3]
        masked_clip_img = clip_img.clone()
        masked_clip_img[mask] = 0  # Zero out masked values

        return masked_clip_img, clip_img, part_labels






