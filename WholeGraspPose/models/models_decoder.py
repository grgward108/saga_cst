import sys
import math  # Added for sqrt

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from WholeGraspPose.models.pointnet import (PointNetFeaturePropagation,
                                            PointNetSetAbstraction)


class ResBlock(nn.Module):
    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class PointNetEncoder(nn.Module):

    def __init__(self,
                 hc,
                 in_feature):

        super(PointNetEncoder, self).__init__()
        self.hc = hc
        self.in_feature = in_feature

        self.enc_sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=self.in_feature, mlp=[self.hc, self.hc*2], group_all=False)
        self.enc_sa2 = PointNetSetAbstraction(npoint=128, radius=0.25, nsample=64, in_channel=self.hc*2 + 3, mlp=[self.hc*2, self.hc*4], group_all=False)
        self.enc_sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=self.hc*4 + 3, mlp=[self.hc*4, self.hc*8], group_all=True)

    def forward(self, l0_xyz, l0_points):

        l1_xyz, l1_points = self.enc_sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.enc_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.enc_sa3(l2_xyz, l2_points)
        x = l3_points.view(-1, self.hc*8)
        
        return l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, x

class MarkerNet(nn.Module):
    def __init__(self, cfg, n_neurons=1024, in_cond=1024, latentD=16, in_feature=143*3, **kwargs):
        super(MarkerNet, self).__init__()

        self.cfg = cfg
        self.latentD = latentD
        self.in_cond = in_cond
        self.in_feature = in_feature  # Total number of marker features (number of markers * features per marker)
        self.n_markers = int(in_feature / 3)  # Assuming each marker has 3 features (x, y, z)
        self.n_neurons = n_neurons
        self.obj_cond_feature = 1 if cfg.cond_object_height else 0

        # Transformer parameters
        self.d_model = 128  # Embedding dimension for the transformer
        self.nhead = 8      # Number of attention heads in the transformer
        self.num_layers = 4 # Number of transformer encoder layers
        self.dim_feedforward = 256  # Dimension of the feedforward network in the transformer

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.n_markers)

        # Input embedding layer
        input_dim = 3 + self.obj_cond_feature  # Input features per marker (3 for x, y, z, plus object height if used)
        self.input_proj = nn.Linear(input_dim, self.d_model)  # Projects marker inputs to the embedding dimension

        self.object_cond_proj = nn.Linear(self.in_cond, self.d_model)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        # Define separate biases for left and right hands
        self.left_hand_attention_bias = nn.Parameter(torch.tensor(1.0))  # Learnable left hand bias
        self.right_hand_attention_bias = nn.Parameter(torch.tensor(1.0))  # Learnable right hand bias


        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Decoder: Project latent vector to sequence embeddings
        self.z_to_seq = nn.Linear(latentD, self.n_markers * self.d_model)

        # Transformer decoder layers (using encoder layers for simplicity)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward, batch_first=True)
        self.transformer_decoder_layers = nn.TransformerEncoder(decoder_layer, num_layers=self.num_layers)

        # Output projection layers
        self.output_proj_xyz = nn.Linear(self.d_model, 3)
        self.output_proj_p = nn.Linear(self.d_model, 1)
        self.p_output = nn.Sigmoid()  # Activation function for probabilities

        # Residual blocks are removed since we're using transformer layers in the decoder

    def enc(self, cond_object, markers, contacts_markers, transf_transl, part_labels):
        _, _, _, _, _, object_cond = cond_object

        bs = markers.size(0)
        markers = markers.view(bs, self.n_markers, 3).float()
        part_labels = part_labels.view(bs, self.n_markers)

        contacts_markers = contacts_markers.float()
        transf_transl = transf_transl.float()

        if self.obj_cond_feature == 1:
            object_height = transf_transl[:, -1, None].unsqueeze(1).repeat(1, self.n_markers, 1)
            markers = torch.cat([markers, object_height], dim=-1)

        # Project marker features to embedding dimension
        marker_embeds = self.input_proj(markers)  # Shape: (bs, n_markers, d_model)
        marker_embeds = self.pos_encoder(marker_embeds)

        # Transformer Encoder
        transformer_output = self.transformer_encoder(marker_embeds)  # Shape: (bs, n_markers, d_model)

        # Project object condition to embedding dimension
        object_cond_proj = self.object_cond_proj(object_cond)  # Shape: (bs, d_model)
        object_cond_proj = object_cond_proj.unsqueeze(1)  # Shape: (bs, 1, d_model)

        # Custom Cross-Attention with Hand Bias
        query = self.q_proj(object_cond_proj)  # (bs, 1, d_model)
        key = self.k_proj(transformer_output)    # (bs, n_markers, d_model)
        value = self.v_proj(transformer_output)  # (bs, n_markers, d_model)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)  # (bs, 1, n_markers)

        # Assuming hand labels are 5 for left hand and 6 for right hand
        left_hand_labels = torch.tensor([5], device=part_labels.device)
        right_hand_labels = torch.tensor([6], device=part_labels.device)

        is_left_hand_marker = (part_labels.unsqueeze(-1) == left_hand_labels).any(dim=-1)  # (bs, n_markers)
        is_right_hand_marker = (part_labels.unsqueeze(-1) == right_hand_labels).any(dim=-1)  # (bs, n_markers)

        left_hand_mask = is_left_hand_marker.float().unsqueeze(1)  # (bs, 1, n_markers)
        right_hand_mask = is_right_hand_marker.float().unsqueeze(1)  # (bs, 1, n_markers)

        # Adjust attention scores with separate biases
        adjusted_attn_scores = attn_scores + left_hand_mask * self.left_hand_attention_bias + right_hand_mask * self.right_hand_attention_bias


        # Compute attention weights
        attn_weights = F.softmax(adjusted_attn_scores, dim=-1)  # (bs, 1, n_markers)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)  # (bs, 1, d_model)

        # Flatten attn_output
        attn_output_flat = attn_output.view(bs, -1)  # (bs, d_model)

        # Pass through a fully connected layer or further processing
        X = F.relu(attn_output_flat)

        return X

    def dec(self, Z, cond_object, transf_transl):
        bs = Z.size(0)

        # Project Z to sequence embeddings
        z_seq = self.z_to_seq(Z).view(bs, self.n_markers, self.d_model)  # Shape: (bs, n_markers, d_model)

        # Apply positional encoding
        z_seq = self.pos_encoder(z_seq)

        # Pass through transformer decoder layers
        transformer_output = self.transformer_decoder_layers(z_seq)  # Shape: (bs, n_markers, d_model)

        # Predict positions and probabilities
        xyz_pred = self.output_proj_xyz(transformer_output)  # Shape: (bs, n_markers, 3)
        p_pred = self.p_output(self.output_proj_p(transformer_output).squeeze(-1))  # Shape: (bs, n_markers)

        # Flatten the xyz predictions to match the expected output shape
        xyz_pred = xyz_pred.view(bs, -1)

        return xyz_pred, p_pred


class ContactNet(nn.Module):
    def __init__(self, cfg, latentD=16, hc=64, object_feature=6, **kwargs):
        super(ContactNet, self).__init__()
        self.latentD = latentD
        self.hc = hc
        self.object_feature  = object_feature

        self.enc_pointnet = PointNetEncoder(self.hc, self.object_feature+1)

        self.dec_fc1 = nn.Linear(self.latentD, self.hc*2)
        self.dec_bn1 = nn.BatchNorm1d(self.hc*2)
        self.dec_drop1 = nn.Dropout(0.1)
        self.dec_fc2 = nn.Linear(self.hc*2, self.hc*4)
        self.dec_bn2 = nn.BatchNorm1d(self.hc*4)
        self.dec_drop2 = nn.Dropout(0.1)
        self.dec_fc3 = nn.Linear(self.hc*4, self.hc*8)
        self.dec_bn3 = nn.BatchNorm1d(self.hc*8)
        self.dec_drop3 = nn.Dropout(0.1)

        self.dec_fc4 = nn.Linear(self.hc*8+self.latentD, self.hc*8)
        self.dec_bn4 = nn.BatchNorm1d(self.hc*8)
        self.dec_drop4 = nn.Dropout(0.1)

        self.dec_fp3 = PointNetFeaturePropagation(in_channel=self.hc*8+self.hc*4, mlp=[self.hc*8, self.hc*4])
        self.dec_fp2 = PointNetFeaturePropagation(in_channel=self.hc*4+self.hc*2, mlp=[self.hc*4, self.hc*2])
        self.dec_fp1 = PointNetFeaturePropagation(in_channel=self.hc*2+self.object_feature, mlp=[self.hc*2, self.hc*2])

        self.dec_conv1 = nn.Conv1d(self.hc*2, self.hc*2, 1)
        self.dec_conv_bn1 = nn.BatchNorm1d(self.hc*2)
        self.dec_conv_drop1 = nn.Dropout(0.1)
        self.dec_conv2 = nn.Conv1d(self.hc*2, 1, 1)

        self.dec_output = nn.Sigmoid()

    def enc(self, contacts_object, verts_object, feat_object):
        l0_xyz = verts_object[:, :3, :]
        l0_points = torch.cat([feat_object, contacts_object], 1) if feat_object is not None else contacts_object
        _, _, _, _, _, x = self.enc_pointnet(l0_xyz, l0_points)

        return x  

    def dec(self, z, cond_object, verts_object, feat_object):
        l0_xyz = verts_object[:, :3, :]
        l0_points = feat_object

        l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points = cond_object

        l3_points = torch.cat([l3_points, z], 1)
        l3_points = self.dec_drop4(F.relu(self.dec_bn4(self.dec_fc4(l3_points)), inplace=True))
        l3_points = l3_points.view(l3_points.size()[0], l3_points.size()[1], 1)

        l2_points = self.dec_fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.dec_fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        if l0_points is None:
            l0_points = self.dec_fp1(l0_xyz, l1_xyz, l0_xyz, l1_points)
        else:
            l0_points = self.dec_fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        feat =  F.relu(self.dec_conv_bn1(self.dec_conv1(l0_points)), inplace=True)
        x = self.dec_conv_drop1(feat)
        x = self.dec_conv2(x)
        x = self.dec_output(x)

        return x


class FullBodyGraspNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(FullBodyGraspNet, self).__init__()

        self.cfg = cfg
        self.latentD = cfg.latentD
        self.in_feature_list = {}
        self.in_feature_list['joints'] = 127*3
        self.in_feature_list['markers_143'] = 143*3
        self.in_feature_list['markers_214'] = 214*3
        self.in_feature_list['markers_593'] = 593*3

        self.in_feature = self.in_feature_list[cfg.data_representation]

        self.marker_net = MarkerNet(cfg, n_neurons=cfg.n_markers, in_cond=cfg.pointnet_hc*8, latentD=cfg.latentD, in_feature=self.in_feature)
        self.contact_net = ContactNet(cfg, latentD=cfg.latentD, hc=cfg.pointnet_hc, object_feature=cfg.obj_feature)

        self.pointnet = PointNetEncoder(hc=cfg.pointnet_hc, in_feature=cfg.obj_feature)
        # encoder fusion
        self.enc_fusion = ResBlock(self.marker_net.d_model + self.cfg.pointnet_hc * 8, cfg.n_neurons)

        self.enc_mu = nn.Linear(cfg.n_neurons, cfg.latentD)
        self.enc_var = nn.Linear(cfg.n_neurons, cfg.latentD)


    def encode(self, object_cond, verts_object, feat_object, contacts_object, markers, contacts_markers, transf_transl, part_labels):
        # marker branch
        marker_feat = self.marker_net.enc(object_cond, markers, contacts_markers, transf_transl, part_labels)   # [B, n_neurons=1024]

        # contact branch
        contact_feat = self.contact_net.enc(contacts_object, verts_object, feat_object)  # [B, hc*8]

        # fusion
        X = torch.cat([marker_feat, contact_feat], dim=-1)
        X = self.enc_fusion(X, True)

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))


    def decode(self, Z, object_cond, verts_object, feat_object, transf_transl):

        bs = Z.shape[0]
        # marker_branch
        markers_xyz_pred, markers_p_pred = self.marker_net.dec(Z, object_cond, transf_transl)

        # contact branch
        contact_pred = self.contact_net.dec(Z, object_cond, verts_object, feat_object)

        return markers_xyz_pred.view(bs, -1, 3), markers_p_pred, contact_pred

    def forward(self, verts_object, feat_object, contacts_object, markers, contacts_markers, transf_transl, part_labels, **kwargs):
        object_cond = self.pointnet(l0_xyz=verts_object, l0_points=feat_object)
        z = self.encode(object_cond, verts_object, feat_object, contacts_object, markers, contacts_markers, transf_transl, part_labels)
        z_s = z.rsample()

        markers_xyz_pred, markers_p_pred, object_p_pred = self.decode(z_s, object_cond, verts_object, feat_object, transf_transl)

        results = {'markers': markers_xyz_pred, 'contacts_markers': markers_p_pred, 'contacts_object': object_p_pred, 'object_code': object_cond[-1], 'mean': z.mean, 'std': z.scale}

        return results


    def sample(self, verts_object, feat_object, transf_transl, seed=None):
        bs = verts_object.shape[0]
        if seed is not None:
            np.random.seed(seed)
        dtype = verts_object.dtype
        device = verts_object.device
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0., 1., size=(bs, self.latentD)) #generates random latent vectors Zgen using standard normal distribution
            Zgen = torch.tensor(Zgen,dtype=dtype).to(device)

        object_cond = self.pointnet(l0_xyz=verts_object, l0_points=feat_object)

        return self.decode(Zgen, object_cond, verts_object, feat_object, transf_transl)
