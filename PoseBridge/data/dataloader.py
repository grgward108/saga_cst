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

    def create_body_hand_repr(self, smplx_model_path=None):
        """
        Generate normalized marker positions for body and hand.
        """
        self.clip_img_list = []
        self.traj_gt_list = []

        for i in tqdm(range(self.n_samples)):
            body_param_ = self.data_dict_list[i]['body']
            body_param_['transl'] = torch.from_numpy(body_param_['transl']).float().to(device)

            markers = body_param_['transl']  # Replace this with marker computation logic
            markers_np = markers.cpu().numpy()

            # Normalize markers
            markers_mean = markers_np.mean(axis=(0, 1), keepdims=True)
            markers_std = markers_np.std(axis=(0, 1), keepdims=True)
            normalized_markers = (markers_np - markers_mean) / (markers_std + 1e-8)

            # Store normalized data
            self.clip_img_list.append(normalized_markers)

        self.clip_img_list = np.asarray(self.clip_img_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        """
        Return marker data without channels.
        """
        clip_img = self.clip_img_list[index]  # Shape: [sequence_length, features]
        clip_img = torch.from_numpy(clip_img).float().permute(1, 0)  # Reshape to [features, sequence_length]

        # Apply random masking
        mask = torch.rand(clip_img.shape) < self.mask_prob
        masked_clip_img = clip_img.clone()
        masked_clip_img[mask] = 0

        return masked_clip_img, clip_img



