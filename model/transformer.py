# from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile
import pdb
import math


class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, num_heads=8, dim_feedforward=1024, num_layers=3):
        super(TransformerBottleneck, self).__init__()
        
        # Patch embedding: Flatten spatial dimensions and map to desired feature dimension
        self.patch_embed = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reshape back to original dimensions after Transformer processing
        self.unpatch_embed = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        # x: (B, C, H, W)
        batch_size, C, H, W = x.shape
        
        # Patch Embedding: Flatten the spatial dimensions
        x = self.patch_embed(x).view(batch_size, C, -1).permute(2, 0, 1)  # (N, B, C)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # (N, B, C)
        
        # Reshape back to (B, C, H, W)
        x = x.permute(1, 2, 0).view(batch_size, C, H, W)
        x = self.unpatch_embed(x)
        
        return x
