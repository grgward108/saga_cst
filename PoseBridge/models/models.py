import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.fftpack import dct, idct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoseBridgeTransformer(nn.Module):
    def __init__(self, n_markers=143, d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=256, latent_dim=16):
        super(PoseBridgeTransformer, self).__init__()
        
        self.n_markers = n_markers
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Learnable latent vector (single vector per sequence)
        self.latent_vec = nn.Parameter(torch.randn(1, latent_dim, device=device))
        
        # Input projection layer for marker positions (e.g., 3D coordinates)
        self.input_proj = nn.Linear(3, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Latent projection for interpolation and sequence generation
        self.latent_to_seq = nn.Linear(latent_dim, n_markers * d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection layers
        self.output_proj_xyz = nn.Linear(d_model, 3)  # Output 3D coordinates for each marker
        
    def encode(self, markers):
        """
        Encodes input marker positions into a latent representation using the transformer encoder.
        
        markers: (batch_size, n_markers, 3) - 3D marker positions
        """
        # Project input markers into embedding dimension
        markers = markers.view(-1, self.n_markers, 3)  # Ensure input shape
        marker_embeds = self.input_proj(markers).permute(1, 0, 2)  # Shape: (n_markers, batch_size, d_model)
        
        # Pass through encoder
        encoder_output = self.transformer_encoder(marker_embeds)  # Shape: (n_markers, batch_size, d_model)
        return encoder_output.permute(1, 0, 2)  # Shape: (batch_size, n_markers, d_model)
    
    def decode(self, z_seq, memory):
        """
        Decodes intermediate marker positions using the transformer decoder.
        
        z_seq: (batch_size, n_markers, d_model) - Sequence embeddings from latent vector
        memory: Encoder output to use as memory for the decoder
        """
        z_seq = z_seq.permute(1, 0, 2)  # Shape: (n_markers, batch_size, d_model)
        transformer_output = self.transformer_decoder(z_seq, memory.permute(1, 0, 2))
        transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (batch_size, n_markers, d_model)
        
        # Project to output 3D coordinates
        xyz_pred = self.output_proj_xyz(transformer_output)
        return xyz_pred.view(-1, self.n_markers, 3)  # Shape: (batch_size, n_markers, 3)
    
    def apply_dct_smoothing(self, latent_vec_sequence, keep_frequencies=10):
        """
        Applies DCT on the latent vector sequence and keeps only the low-frequency components.
        
        latent_vec_sequence: Tensor representing the latent sequence over time.
        keep_frequencies: Number of low-frequency components to retain.
        
        Returns a smoothed latent vector sequence.
        """
        # Convert to numpy for DCT
        latent_array = latent_vec_sequence.cpu().numpy()
        
        # Apply DCT along the sequence dimension (axis=0)
        latent_dct = dct(latent_array, axis=0, norm='ortho')
        
        # Zero out high frequencies
        latent_dct[keep_frequencies:] = 0
        
        # Inverse DCT to get the smoothed sequence
        smoothed_latent_array = idct(latent_dct, axis=0, norm='ortho')
        
        # Convert back to tensor
        return torch.tensor(smoothed_latent_array, device=device)
    
    def forward(self, start_markers, end_markers, num_frames=10, keep_frequencies=10):
        """
        Forward pass for PoseBridge to generate interpolated frames between start and end markers.
        
        start_markers: (batch_size, n_markers, 3) - Starting marker positions
        end_markers: (batch_size, n_markers, 3) - Ending marker positions
        num_frames: Number of interpolated frames to generate
        keep_frequencies: Number of DCT frequencies to retain for smoothing
        """
        # Encode start and end markers
        start_encoding = self.encode(start_markers)
        end_encoding = self.encode(end_markers)
        
        # Generate a sequence of latent vectors for interpolation
        latent_vec_sequence = torch.cat(
            [alpha * self.latent_vec + (1 - alpha) * self.latent_vec for alpha in torch.linspace(0, 1, num_frames)]
        )
        
        # Apply DCT-based smoothing to the latent vector sequence
        smoothed_latent_sequence = self.apply_dct_smoothing(latent_vec_sequence, keep_frequencies)
        
        # Project latent sequence to sequence embeddings
        z_seq = self.latent_to_seq(smoothed_latent_sequence).view(-1, self.n_markers, self.d_model)
        
        # Concatenate start and end encodings as memory for decoder
        memory = torch.cat([start_encoding, end_encoding], dim=1)
        
        # Decode each frame
        interpolated_frames = [self.decode(z, memory) for z in z_seq]
        return torch.stack(interpolated_frames)  # Shape: (num_frames, batch_size, n_markers, 3)
