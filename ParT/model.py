"""Particle Transformer (ParT) in PyTorch.

Converted and adapted from TensorFlow reference implementation.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def prepare_interaction(x: torch.Tensor) -> torch.Tensor:
    """Prepare the features for interaction matrix U.

    Args:
        x : torch.Tensor
            Input tensor of shape (N, L, 3), where N is the batch size,
            L is the number of particles, and 3 is the feature dimension
            corresponding to (pt_rel, delta_eta, delta_phi).
    Returns:
        torch.Tensor
            Output tensor of shape (N, 3, L, L), where N is the batch size,
            L is the number of particles, and 3 is the feature dimension
            corresponding to (delta, kt, z).
    """
    # Expand dimensions for broadcasting
    x_i = x.unsqueeze(-2)  # (N, L, 1, 3)
    x_j = x.unsqueeze(-3)  # (N, 1, L, 3)

    # Split features
    pt_rel_i, delta_eta_i, delta_phi_i = torch.unbind(x_i, dim=-1)  # (N, L, 1)
    pt_rel_j, delta_eta_j, delta_phi_j = torch.unbind(x_j, dim=-1)  # (N, 1, L)

    # Calculate delta and mod delta_phi to [-pi, pi]
    delta_eta_diff = delta_eta_i - delta_eta_j
    delta_phi_diff = (delta_phi_i - delta_phi_j + np.pi) % (2 * np.pi) - np.pi
    delta = torch.sqrt(delta_eta_diff ** 2 + delta_phi_diff ** 2)  # (N, L, L)

    # Calculate kt and z
    pt_rel_min = torch.minimum(pt_rel_i, pt_rel_j)  # (N, L, L)
    kt = pt_rel_min * delta  # (N, L, L)
    z = pt_rel_min / torch.clamp(pt_rel_i + pt_rel_j, min=1e-12)  # (N, L, L)

    # Stack and clamp values to avoid numerical issues
    features = torch.stack([delta, kt, z], dim=-3)  # (N, 3, L, L)
    features = torch.clamp(features, min=1e-9)
    return torch.log(features)


class ParticleFeatureEmbedding(nn.Module):
    """Embedding network for individual particle features."""

    def __init__(self, input_dim: int, embedding_dims: list[int]):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims

        layers = []
        in_dim = input_dim
        for embed_dim in embedding_dims:
            layers.append(nn.LayerNorm(in_dim))
            layers.append(nn.Linear(in_dim, embed_dim))
            layers.append(nn.GELU())
            in_dim = embed_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, L, input_dim) -> Output shape: (N, L, embed_dim_last)
        return self.network(x)


class InteractionMatrixEmbedding(nn.Module):
    """Embedding network for pairwise interaction matrix features."""

    def __init__(self, input_dim: int, embedding_dims: list[int]):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dims = embedding_dims

        layers = []
        in_dim = input_dim
        for embed_dim in embedding_dims:
            layers.append(nn.Conv2d(in_dim, embed_dim, kernel_size=1))
            layers.append(nn.GELU())
            in_dim = embed_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, input_dim, L, L) -> Output shape: (N, embed_dim_last, L, L)
        return self.network(x)


class MultiheadAttention(nn.Module):
    """Custom Multihead Attention module supporting padding masks, 

    relative attention masks, and head scaling.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, 
                 use_bias: bool = False, use_head_scale: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        self.use_head_scale = use_head_scale
        if use_head_scale:
            self.gamma = nn.Parameter(torch.ones(num_heads))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # query shape: (N, L_q, E)
        # key shape: (N, L_k, E)
        # value shape: (N, L_v, E)
        batch_size = query.size(0)

        # Projections & reshape to (N, H, L, D)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (N, H, L_q, L_k)

        # Apply padding mask (True means mask out/ignore)
        if key_padding_mask is not None:
            # Expand key_padding_mask from (N, L_k) to (N, 1, 1, L_k)
            scores_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(scores_mask, -1e9)
            
            # Mask out corresponding values
            v_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # (N, 1, L_k, 1)
            v = v.masked_fill(v_mask, 0.0)

        # Apply custom attention mask (e.g. interaction matrix U)
        if attn_mask is not None:
            scores = scores + attn_mask

        # Softmax & dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum
        weighted_sum = torch.matmul(attn_weights, v)  # (N, H, L_q, D)

        # Apply head scale
        if self.use_head_scale:
            weighted_sum = weighted_sum * self.gamma.view(1, -1, 1, 1)

        # Concatenate heads & final linear projection
        weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(weighted_sum)
        
        return output


class AttentionBlock(nn.Module):
    """Transformer Attention Block (either Self-Attention or Class-Attention)."""

    def __init__(self, embed_dim: int, num_heads: int, fc_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.attention = MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.post_attn_norm = nn.LayerNorm(embed_dim)
        
        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.post_fc_norm = nn.LayerNorm(fc_dim)
        
        self.fc1 = nn.Linear(embed_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, embed_dim)
        self.activation = nn.GELU()
        
        self.act_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual scaling parameter
        self.lambda_residual = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor, x_clt: torch.Tensor | None = None, 
                attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Case A: Class Attention Block
        if x_clt is not None:
            # Concatenate class token mask if key_padding_mask exists
            if key_padding_mask is not None:
                clt_padding_mask = torch.zeros(key_padding_mask.size(0), 1, dtype=torch.bool, device=key_padding_mask.device)
                key_padding_mask = torch.cat([clt_padding_mask, key_padding_mask], dim=1)

            residual = x_clt
            
            # Combine class token with particle representations
            combined_x = torch.cat([x_clt, x], dim=1)  # (N, 1+L, E)
            combined_x = self.pre_attn_norm(combined_x)
            
            # Query is class token, Keys/Values are combined representation
            x = self.attention(
                query=x_clt, 
                key=combined_x, 
                value=combined_x, 
                key_padding_mask=key_padding_mask
            )
        # Case B: Particle Self-Attention Block
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attention(
                query=x, 
                key=x, 
                value=x, 
                attn_mask=attn_mask, 
                key_padding_mask=key_padding_mask
            )

        x = self.post_attn_norm(x)
        x = self.dropout1(x)
        x = x + residual

        # Feed Forward Network
        residual = x
        x = self.pre_fc_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.act_dropout(x)
        x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        # Residual connection with scale
        x = x + residual * self.lambda_residual
        
        return x


class ParticleTransformer(nn.Module):
    """Standard Particle Transformer module."""

    def __init__(self, score_dim: int, parameters: dict):
        super().__init__()
        self.score_dim = score_dim
        self.model_params = parameters

        # Particle Feature Embedding
        self.par_embedding = ParticleFeatureEmbedding(
            input_dim=parameters['ParEmbed']['input_dim'],
            embedding_dims=parameters['ParEmbed']['embed_dim']
        )

        atte_embed_dim = parameters['ParEmbed']['embed_dim'][-1]

        # Particle Attention Blocks
        self.par_atte_blocks = nn.ModuleList([
            AttentionBlock(
                embed_dim=atte_embed_dim,
                **parameters['ParAtteBlock']
            ) for _ in range(parameters['num_ParAtteBlock'])
        ])

        # Class Attention Blocks
        self.class_atte_blocks = nn.ModuleList([
            AttentionBlock(
                embed_dim=atte_embed_dim,
                **parameters['ClassAtteBlock']
            ) for _ in range(parameters['num_ClassAtteBlock'])
        ])

        # Learnable Class Token
        self.class_token = nn.Parameter(torch.empty(1, 1, atte_embed_dim))
        init.trunc_normal_(self.class_token, std=0.02)

        self.layer_norm = nn.LayerNorm(atte_embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(atte_embed_dim, atte_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.final_layer = nn.Linear(atte_embed_dim, score_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, L, D) where D is feature dimension
        batch_size = x.size(0)

        # Create padding mask (True where pt value is NaN)
        key_padding_mask = torch.isnan(x[..., 0])  # (N, L)

        # Fill NaN values with zeros for the numerical operations
        x = torch.where(key_padding_mask.unsqueeze(-1), torch.zeros_like(x), x)

        # Particle Embedding
        x = self.par_embedding(x)  # (N, L, E)

        # Particle Self-Attention blocks
        for block in self.par_atte_blocks:
            x = block(x, x_clt=None, attn_mask=None, key_padding_mask=key_padding_mask)

        # Class Attention blocks
        class_token = self.class_token.expand(batch_size, -1, -1)  # (N, 1, E)
        for block in self.class_atte_blocks:
            class_token = block(x, x_clt=class_token, key_padding_mask=key_padding_mask)

        # Final classification head
        class_token = self.layer_norm(class_token).squeeze(1)  # (N, E)
        class_token = self.fc(class_token)
        class_token = self.final_layer(class_token)
        
        return class_token


class ParT_Baseline(ParticleTransformer):
    """Baseline Particle Transformer config."""

    def __init__(self, num_channels: int = 3):
        hyperparameters = {
            "ParEmbed": {
                "input_dim": 3 + num_channels,  # (pt, eta, phi_rel) + type one-hot
                "embed_dim": [64, 512, 64]
            },
            "ParAtteBlock": {
                "num_heads": 8,
                "fc_dim": 512,
                "dropout": 0.1
            },
            "ClassAtteBlock": {
                "num_heads": 8,
                "fc_dim": 512,
                "dropout": 0.0
            },
            "num_ParAtteBlock": 6,
            "num_ClassAtteBlock": 2
        }
        super().__init__(score_dim=1, parameters=hyperparameters)


class ParT_Light(ParticleTransformer):
    """Lightweight Particle Transformer config."""

    def __init__(self, num_channels: int = 3):
        hyperparameters = {
            "ParEmbed": {
                "input_dim": 3 + num_channels,  # (pt, eta, phi_rel) + type one-hot
                "embed_dim": [64, 256, 64]
            },
            "ParAtteBlock": {
                "num_heads": 4,
                "fc_dim": 256,
                "dropout": 0.1
            },
            "ClassAtteBlock": {
                "num_heads": 4,
                "fc_dim": 256,
                "dropout": 0.0
            },
            "num_ParAtteBlock": 3,
            "num_ClassAtteBlock": 1
        }
        super().__init__(score_dim=1, parameters=hyperparameters)


if __name__ == "__main__":
    # Test forward pass with mock data
    print("Testing PyTorch Particle Transformer (ParT) model...")
    model = ParT_Light(num_channels=3)
    
    # Batch size = 4, sequence length = 5, feature dimensions = 6
    mock_input = torch.randn(4, 5, 6)
    
    # Simulate padding (some NaNs in the first column)
    mock_input[2, 3:, 0] = float('nan')
    mock_input[3, 4:, 0] = float('nan')
    
    output = model(mock_input)
    print(f"Input shape: {mock_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 1)
    print("Test passed successfully!")
