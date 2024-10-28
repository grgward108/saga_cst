import gc
import json
import os
import time

import numpy as np
import torch
from smplx.lbs import batch_rodrigues
from torch.utils import data
from utils.train_helper import point2point_signed

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 gender=None,
                 motion_intent=None,
                 object_class=['all'],
                 dtype=torch.float32,
                 data_type = 'markers_143'):

        super().__init__()

        print('Preparing {} data...'.format(ds_name.upper()))
        self.sbj_idxs = []
        self.objs_frames = {}
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.gender = gender
        self.motion_intent = motion_intent
        self.object_class = object_class
        self.data_type = data_type

        self.body_part_groups = {
            'head_and_neck': [2819, 3076, 1795, 2311, 1043, 919, 8985, 1696, 1703, 9002, 8757, 2383, 2898, 3035, 2148, 9066, 8947, 2041, 2813],
            'trunk': [4391, 4297, 5615, 5944, 5532, 5533, 5678, 7145],
            'right_upper_limb': [7179, 7028, 7115, 7251, 7274, 7293, 6778, 7036],
            'left_upper_limb': [4509, 4245, 4379, 4515, 4538, 4557, 4039, 4258],
            'right_hand': [8001, 7781, 7750, 7978, 7756, 7884, 7500, 7419, 7984, 7633, 7602, 7667, 7860, 8082, 7351, 7611, 7867, 7423, 7357, 7396, 7443, 7446, 7536, 7589, 7618, 7625, 7692, 7706, 7730, 7748, 7789, 7847, 7858, 7924, 7931, 7976, 8039, 8050, 8087, 8122],
            'left_hand': [4897, 5250, 4931, 5124, 5346, 4615, 5321, 4875, 5131, 4683, 4686, 4748, 5268, 5045, 5014, 5242, 5020, 5149, 4628, 4641, 4660, 4690, 4691, 4710, 4750, 4885, 4957, 4970, 5001, 5012, 5082, 5111, 5179, 5193, 5229, 5296, 5306, 5315, 5353, 5387],
            'left_legs': [5857, 5893, 5899, 3479, 3781, 3638, 3705, 5761, 8852, 5726],
            'right_legs': [8551, 8587, 8593, 6352, 6539, 6401, 6466, 8455, 8634, 8421],
        }

        with open('body_utils/smplx_markerset.json') as f:
            markerset = json.load(f)['markersets']
            self.markers_idx = []
            for marker in markerset:
                if marker['type'] not in ['palm_5']:   # 'palm_5' contains selected 5 markers per palm, but for training we use 'palm' set where there are 22 markers per palm. 
                    self.markers_idx += list(marker['indices'].values())
        self.ds = self.load_full_data(self.ds_path)
        

    def map_marker_to_part(self, marker_indices):
        """
        Function to map markers to specific body parts based on markers_idx.
        """
        part_labels = np.zeros(len(marker_indices))  # Initialize part labels for each marker as 0 (unassigned)

        # Loop through each body part and assign a label
        for part_index, (part_name, part_indices) in enumerate(self.body_part_groups.items()):
            for marker_idx in part_indices:
                if marker_idx in marker_indices:
                    mapped_index = marker_indices.index(marker_idx)
                    part_labels[mapped_index] = part_index + 1 

        return part_labels
        

    def load_full_data(self, path):
        rec_list = []
        output = {}

        markers_list = []
        transf_transl_list = []
        verts_object_list = []
        contacts_object_list = []
        normal_object_list = []
        transl_object_list = []
        global_orient_object_list = []
        rotmat_list = []
        contacts_markers_list = []
        part_labels_list = []
        body_list = {}
        for key in ['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'expression']:
            body_list[key] = []
            
        subsets_dict = {'male':['s1', 's2', 's8', 's9', 's10'],
                       'female': ['s3', 's4', 's5', 's6', 's7']}
        subsets = subsets_dict[self.gender]

        print('loading {} dataset: {}'.format(self.gender, subsets))
        for subset in subsets:
            subset_path = os.path.join(path, subset)
            rec_list += [os.path.join(subset_path, i) for i in os.listdir(subset_path)]

        index = 0

        for rec in rec_list:
            data = np.load(rec, allow_pickle=True)

            ## select object
            obj_name = rec.split('/')[-1].split('_')[0]
            if 'all' not in self.object_class:
                if obj_name not in self.object_class:
                    continue

            verts_object_list.append(data['verts_object'])
            markers_list.append(data[self.data_type])
            transf_transl_list.append(data['transf_transl'])
            normal_object_list.append(data['normal_object'])
            global_orient_object_list.append(data['global_orient_object'])

            # Get SMPL-X markers
            marker_indices = self.markers_idx
            part_labels = self.map_marker_to_part(marker_indices)  # Map each marker to a body part
            num_frames = data['verts_object'].shape[0]
            part_labels_expanded = np.tile(part_labels, (num_frames, 1))
            part_labels_list.append(part_labels_expanded)


            orient = torch.tensor(data['global_orient_object'])
            rot_mats = batch_rodrigues(orient.view(-1, 3)).view([orient.shape[0], 9]).numpy()
            rotmat_list.append(rot_mats)

            object_contact = data['contact_object']
            markers_contact = data['contact_body'][:, self.markers_idx]
            object_contact_binary = (object_contact>0).astype(int)
            contacts_object_list.append(object_contact_binary)
            markers_contact_binary = (markers_contact>0).astype(int)
            contacts_markers_list.append(markers_contact_binary)

            # SMPLX parameters (optional)
            for key in data['body'][()].keys():
                body_list[key].append(data['body'][()][key])

            sbj_id = rec.split('/')[-2]
            self.sbj_idxs += [sbj_id]*data['verts_object'].shape[0]
            if obj_name in self.objs_frames.keys():
                self.objs_frames[obj_name] += list(range(index, index+data['verts_object'].shape[0]))
            else:
                self.objs_frames[obj_name] = list(range(index, index+data['verts_object'].shape[0]))
            index += data['verts_object'].shape[0]
        output['transf_transl'] = torch.tensor(np.concatenate(transf_transl_list, axis=0))
        output['markers'] = torch.tensor(np.concatenate(markers_list, axis=0))              # (B, 143, 3)
        output['verts_object'] = torch.tensor(np.concatenate(verts_object_list, axis=0))    # (B, 2048, 3)
        output['marker_object_distance'] = point2point_signed(output['markers'], output['verts_object'])
        print(output['marker_object_distance'].shape)
        output['contacts_object'] = torch.tensor(np.concatenate(contacts_object_list, axis=0))    # (B, 2048, 3)
        output['contacts_markers'] = torch.tensor(np.concatenate(contacts_markers_list, axis=0))    # (B, 2048, 3)
        output['normal_object'] = torch.tensor(np.concatenate(normal_object_list, axis=0))    # (B, 2048, 3)
        output['global_orient_object'] = torch.tensor(np.concatenate(global_orient_object_list, axis=0))    # (B, 2048, 3)
        output['rotmat'] = torch.tensor(np.concatenate(rotmat_list, axis=0))    # (B, 2048, 3)
        output['part_labels'] = torch.tensor(np.concatenate(part_labels_list, axis=0))  # Part-based labels (B, 143)

        # SMPLX parameters
        output['smplxparams'] = {}
        for key in ['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'expression']:
            output['smplxparams'][key] = torch.tensor(np.concatenate(body_list[key], axis=0))

        return output

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]

    def __getitem__(self, idx):

        try:
            data_out = {}

            data_out['markers'] = self.ds['markers'][idx]
            data_out['contacts_markers'] = self.ds['contacts_markers'][idx]

            data_out['part_labels'] = self.ds['part_labels'][idx].long()
            data_out['marker_object_distance'] = self.ds['marker_object_distance'][idx]
            print(data_out['marker_object_distance'].shape)
            data_out['verts_object'] = self.ds['verts_object'][idx]
            data_out['normal_object'] = self.ds['normal_object'][idx]
            data_out['global_orient_object'] = self.ds['global_orient_object'][idx]
            data_out['transf_transl'] = self.ds['transf_transl'][idx]
            data_out['contacts_object'] = self.ds['contacts_object'][idx]
            if len(data_out['verts_object'].shape) == 2:
                data_out['feat_object'] = torch.cat([self.ds['normal_object'][idx], self.ds['rotmat'][idx, :6].view(1, 6).repeat(2048, 1)], -1)
            else:
                data_out['feat_object'] = torch.cat([self.ds['normal_object'][idx], self.ds['rotmat'][idx, :6].view(-1, 1, 6).repeat(1, 2048, 1)], -1)

            # You may want to uncomment it when you need smplxparams!!!
            data_out['smplxparams'] = {}
            for key in ['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'expression']:
                data_out['smplxparams'][key] = self.ds['smplxparams'][key][idx]

            # Random rotation augmentation
            bsz = 1
            theta = torch.FloatTensor(np.random.uniform(-np.pi / 6, np.pi / 6, bsz))
            orient = torch.zeros((bsz, 3))
            orient[:, -1] = theta
            rot_mats = batch_rodrigues(orient.view(-1, 3)).view([bsz, 3, 3])
            if len(data_out['verts_object'].shape) == 3:
                data_out['markers'] = torch.matmul(data_out['markers'][:, :, :3], rot_mats.squeeze())
                data_out['verts_object'] = torch.matmul(data_out['verts_object'][:, :, :3], rot_mats.squeeze())
                data_out['normal_object'][:, :, :3] = torch.matmul(data_out['normal_object'][:, :, :3], rot_mats.squeeze())
            else:
                data_out['markers'] = torch.matmul(data_out['markers'][:, :3], rot_mats.squeeze())
                data_out['verts_object'] = torch.matmul(data_out['verts_object'][:, :3], rot_mats.squeeze())
                data_out['normal_object'][:, :3] = torch.matmul(data_out['normal_object'][:, :3], rot_mats.squeeze())

            return data_out

        except IndexError as e:
            logging.error(f"IndexError accessing index {idx}: {e}")
            raise
