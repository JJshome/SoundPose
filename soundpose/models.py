"""
Models module for SoundPose.

This module contains the transformer-based models used for feature extraction
and anomaly detection in the SoundPose framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for the transformer encoder.
    
    Attributes:
        hidden_size (int): Size of the hidden representations
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the multi-head self-attention module.
        
        Args:
            hidden_size: Size of the hidden representations
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        assert self.head_size * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Linear layers for query, key, value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the multi-head self-attention module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape for multi-head attention
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.head_size)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, value)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.output(context)
        
        return output


class FeedForward(nn.Module):
    """
    Feed-forward network used in the transformer encoder.
    
    Attributes:
        hidden_size (int): Size of the input and output
        ff_size (int): Size of the inner feed-forward layer
        dropout (float): Dropout probability
    """
    
    def __init__(self, hidden_size: int, ff_size: int, dropout: float = 0.1):
        """
        Initialize the feed-forward network.
        
        Args:
            hidden_size: Size of the input and output
            ff_size: Size of the inner feed-forward layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_size, ff_size)
        self.fc2 = nn.Linear(ff_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single layer of the transformer encoder.
    
    Attributes:
        hidden_size (int): Size of the hidden representations
        num_heads (int): Number of attention heads
        ff_size (int): Size of the inner feed-forward layer
        dropout (float): Dropout probability
    """
    
    def __init__(self, hidden_size: int, num_heads: int, ff_size: int, dropout: float = 0.1):
        """
        Initialize the transformer encoder layer.
        
        Args:
            hidden_size: Size of the hidden representations
            num_heads: Number of attention heads
            ff_size: Size of the inner feed-forward layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, ff_size, dropout)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Self-attention with residual connection and layer normalization
        attention_output = self.self_attention(self.norm1(x), mask)
        x = x + self.dropout(attention_output)
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder model for feature extraction.
    
    Attributes:
        model_type (str): Type of model ('voice', 'mechanical', etc.)
        hidden_size (int): Size of the hidden representations
        num_layers (int): Number of transformer encoder layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        model_type: str,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        """
        Initialize the transformer encoder.
        
        Args:
            model_type: Type of model ('voice', 'mechanical', etc.)
            hidden_size: Size of the hidden representations
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input embedding
        self.input_projection = nn.Linear(128, hidden_size)  # Assuming 128 mel bins
        
        # Positional encoding
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(1000, hidden_size),  # Max sequence length of 1000
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_size=hidden_size * 4,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_seq_len: int, hidden_size: int) -> torch.Tensor:
        """
        Create positional encoding for the transformer.
        
        Args:
            max_seq_len: Maximum sequence length
            hidden_size: Size of the hidden representations
            
        Returns:
            Positional encoding tensor of shape (1, max_seq_len, hidden_size)
        """
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(np.log(10000.0) / hidden_size))
        
        pos_encoding = torch.zeros(1, max_seq_len, hidden_size)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        
        Args:
            x: Input tensor of shape (batch_size, n_mels, seq_len)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Transpose to get (batch_size, seq_len, n_mels)
        x = x.transpose(1, 2)
        
        # Get sequence length
        _, seq_len, _ = x.size()
        
        # Apply input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x


class GenerativeModel(nn.Module):
    """
    Generative model for anomaly detection.
    
    This model uses the features extracted by the transformer encoder
    to generate expected normal features, which are then compared with
    the actual features to detect anomalies.
    
    Attributes:
        model_type (str): Type of model ('voice', 'mechanical', etc.)
        latent_dim (int): Size of the latent space
        hidden_size (int): Size of the hidden representations
        device (torch.device): Device to use for computation
    """
    
    def __init__(
        self,
        model_type: str,
        latent_dim: int = 256,
        hidden_size: int = 768,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the generative model.
        
        Args:
            model_type: Type of model ('voice', 'mechanical', etc.)
            latent_dim: Size of the latent space
            hidden_size: Size of the hidden representations
            device: Device to use for computation
        """
        super().__init__()
        
        self.model_type = model_type
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.device = device
        
        # Encoder network (transformer features -> latent space)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, latent_dim * 2),  # mu and logvar
        )
        
        # Decoder network (latent space -> transformer features)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the features to the latent space.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple of (mu, logvar) of shape (batch_size, seq_len, latent_dim)
        """
        encoder_output = self.encoder(x)
        mu, logvar = encoder_output.chunk(2, dim=-1)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent space.
        
        Args:
            mu: Mean tensor of shape (batch_size, seq_len, latent_dim)
            logvar: Log variance tensor of shape (batch_size, seq_len, latent_dim)
            
        Returns:
            Sampled tensor of shape (batch_size, seq_len, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation to the feature space.
        
        Args:
            z: Latent tensor of shape (batch_size, seq_len, latent_dim)
            
        Returns:
            Decoded tensor of shape (batch_size, seq_len, hidden_size)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generative model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Encode to latent space
        mu, logvar = self.encode(x)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode to feature space
        x_reconstructed = self.decode(z)
        
        return x_reconstructed
    
    def loss_function(self, x: torch.Tensor, x_reconstructed: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute the VAE loss function.
        
        Args:
            x: Original input tensor of shape (batch_size, seq_len, hidden_size)
            x_reconstructed: Reconstructed tensor of shape (batch_size, seq_len, hidden_size)
            mu: Mean tensor of shape (batch_size, seq_len, latent_dim)
            logvar: Log variance tensor of shape (batch_size, seq_len, latent_dim)
            
        Returns:
            Total loss value
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconstructed, x, reduction="mean")
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + 0.1 * kl_loss
        
        return total_loss
