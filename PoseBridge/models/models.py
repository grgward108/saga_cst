import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.partbased_segment import get_segmentation, create_marker_to_part

class PoseBridge(nn.Module):
    def __init__(self, n_markers, marker_dim=3, model_dim=128, num_heads=8, num_layers=4, seq_len=62):
        super(PoseBridge, self).__init__()
        self.n_markers = n_markers        # Number of markers per frame
        self.marker_dim = marker_dim      # Features per marker (x, y, z)
        self.model_dim = model_dim        # Embedding dimension for the transformer
        self.num_heads = num_heads        # Number of attention heads
        self.num_layers = num_layers      # Number of transformer layers
        self.seq_len = seq_len            # Sequence length (number of frames)

        # Learnable part embeddings
        self.num_parts = 8  # Example: Total number of parts (e.g., head, arms, legs, etc.)
        self.part_embedding = nn.Embedding(self.num_parts, self.model_dim)

        # Input embedding layer for spatial transformer
        self.input_proj = nn.Linear(self.marker_dim, self.model_dim)

        # Spatial transformer encoder (like MarkerNet)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=self.num_heads, dim_feedforward=256, batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Temporal transformer encoder
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_markers * self.model_dim, nhead=self.num_heads, dim_feedforward=512, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=self.num_layers)

        # Output layer to map back to original marker positions
        self.output_layer = nn.Linear(self.n_markers * self.model_dim, self.n_markers * self.marker_dim)

    def forward(self, x, part_labels):
        """
        Forward pass through PoseBridge.
        
        Args:
            x: [batch_size, seq_len, n_markers, marker_dim]
            part_labels: [n_markers] (marker-to-part mapping for embedding)
        Returns:
            Output: [batch_size, seq_len, n_markers, marker_dim]
        """

        batch_size, seq_len, n_markers, marker_dim = x.size()

        # Flatten each frame into markers: [batch_size * seq_len, n_markers, marker_dim]
        x = x.reshape(batch_size * seq_len, n_markers, marker_dim)

        # Project marker features to embedding dimension: [batch_size * seq_len, n_markers, model_dim]
        marker_embeds = self.input_proj(x)

        # Add part-based embeddings
        part_labels = part_labels.unsqueeze(0).expand(batch_size * seq_len, -1).to(x.device)  # [batch_size * seq_len, n_markers]
        part_embeds = self.part_embedding(part_labels)  # [batch_size * seq_len, n_markers, model_dim]
        marker_embeds = marker_embeds + part_embeds  # Add part-based embeddings to marker embeddings

        # Apply spatial transformer: [batch_size * seq_len, n_markers, model_dim]
        marker_embeds = self.spatial_transformer(marker_embeds)

        # Flatten markers for temporal transformer: [batch_size, seq_len, n_markers * model_dim]
        x = marker_embeds.view(batch_size, seq_len, -1)

        # Apply temporal transformer: [batch_size, seq_len, n_markers * model_dim]
        x = self.temporal_transformer(x)

        # Map back to original marker dimensions: [batch_size, seq_len, n_markers * marker_dim]
        x = self.output_layer(x)

        # Reshape to [batch_size, seq_len, n_markers, marker_dim]
        x = x.view(batch_size, seq_len, n_markers, marker_dim)

        return x


class SpatialTransformer(nn.Module):
    def __init__(self, n_markers, model_dim, num_layers, num_heads):
        super(SpatialTransformer, self).__init__()
        self.n_markers = n_markers
        self.model_dim = model_dim
        # Input embedding layer
        input_marker_dim = 3  # Assuming 3 features per marker (x, y, z)

        self.input_proj = nn.Linear(input_marker_dim, model_dim)

        # Part embeddings
        segmentation = get_segmentation()
        marker_to_part = create_marker_to_part(segmentation, n_markers)
        self.num_parts = max(segmentation.keys()) + 1
        self.register_buffer('marker_to_part', marker_to_part)
        self.part_embedding = nn.Embedding(self.num_parts, model_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, markers):
        # markers: [batch_size * seq_len, n_markers, marker_features]

        bs_seq = markers.size(0)
        markers = markers.float()


        # Input projection
        marker_embeds = self.input_proj(markers)  # [bs_seq, n_markers, model_dim]

        # Add part embeddings
        part_labels = self.marker_to_part.unsqueeze(0).expand(bs_seq, -1).to(markers.device)
        part_embeds = self.part_embedding(part_labels)
        marker_embeds = marker_embeds + part_embeds

        # Transformer Encoder
        marker_embeds = marker_embeds.permute(1, 0, 2)  # [n_markers, bs_seq, model_dim]
        transformer_output = self.transformer_encoder(marker_embeds)
        transformer_output = transformer_output.permute(1, 0, 2)  # [bs_seq, n_markers, model_dim]

        return transformer_output  # [bs_seq, n_markers, model_dim]



class TemporalTransformer(nn.Module):
    def __init__(self, model_dim, num_layers, num_heads):
        super(TemporalTransformer, self).__init__()
        self.model_dim = model_dim

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(model_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch_size, seq_len, model_dim]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # Transformer expects [seq_len, batch_size, model_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, model_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, model_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, model_dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

