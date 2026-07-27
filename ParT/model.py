"""Particle Transformer (ParT) in PyTorch.

Converted and adapted from TensorFlow reference implementation.
"""

import math
from typing import Any, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def prepare_interaction(x: torch.Tensor, pt_log_scale: bool = True, interaction_type: str = "default") -> torch.Tensor:
    """Prepare the features for interaction matrix U.

    Args:
        x : torch.Tensor
            Input tensor of shape (N, L, 3), where N is the batch size,
            L is the number of particles, and 3 is the feature dimension
            corresponding to (pt_rel, delta_eta, delta_phi).
        pt_log_scale : bool
            Whether the input pt is in log-scale. If True, it will be
            exponentiated back to linear scale to compute physics quantities (kt, z).
        interaction_type : str
            Type of interaction features to compute:
            - "default": log(delta), log(kt), log(z)
            - "eta_phi_dr": (|delta_eta|, |delta_phi|, delta_R) in raw scale.
    Returns:
        torch.Tensor
            Output tensor of shape (N, 3, L, L), where N is the batch size,
            L is the number of particles, and 3 is the feature dimension.
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

    if interaction_type == "eta_phi_dr":
        abs_delta_eta = torch.abs(delta_eta_diff)
        abs_delta_phi = torch.abs(delta_phi_diff)
        features = torch.stack([abs_delta_eta, abs_delta_phi, delta], dim=-3)  # (N, 3, L, L)
        return features

    # Exponentiate back to linear scale if log scale is used for pt
    if pt_log_scale:
        pt_i = torch.exp(pt_rel_i)
        pt_j = torch.exp(pt_rel_j)
    else:
        pt_i = pt_rel_i
        pt_j = pt_rel_j

    # Calculate kt and z in linear scale
    pt_min = torch.minimum(pt_i, pt_j)  # (N, L, L)
    kt = pt_min * delta  # (N, L, L)
    z = pt_min / torch.clamp(pt_i + pt_j, min=1e-12)  # (N, L, L)

    # Stack and clamp values to avoid numerical issues
    features = torch.stack([delta, kt, z], dim=-3)  # (N, 3, L, L)
    features = torch.clamp(features, min=1e-9)
    return torch.log(features)


class ParticleFeatureEmbedding(nn.Module):
    """Embedding network for individual particle features.

    Separately embeds continuous particle kinematics (taking 3/4 of embed_dim)
    and particle type features (taking 1/4 of embed_dim), then concatenates them
    into a total embed_dim token representation.
    Example:
      embed_dim = 32 -> continuous: 24, type: 8
      embed_dim = 64 -> continuous: 48, type: 16
    """

    def __init__(
        self,
        kinematic_dim: int,
        num_channels: int,
        embed_dim: int = 64,
        fc_dim: int = 256,
    ):
        super().__init__()
        self.kinematic_dim = kinematic_dim
        self.num_channels = num_channels

        # Type embedding takes 1/4 of total embed_dim, continuous kinematics takes 3/4
        self.type_embed_dim = max(1, embed_dim // 4)
        self.continuous_embed_dim = embed_dim - self.type_embed_dim

        # 1. Continuous particle kinematics embedding (3/4 of total embed_dim)
        kin_layers = [
            nn.LayerNorm(kinematic_dim),
            nn.Linear(kinematic_dim, fc_dim),
            nn.GELU(),
            nn.LayerNorm(fc_dim),
            nn.Linear(fc_dim, self.continuous_embed_dim),
            nn.GELU(),
        ]
        self.continuous_network = nn.Sequential(*kin_layers)

        # 2. Type information embedding (1/4 of total embed_dim)
        self.type_network = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, self.type_embed_dim),
            nn.GELU(),
        )

        self.out_dim = self.continuous_embed_dim + self.type_embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (N, L, kinematic_dim + num_channels)
        x_kin = x[..., : self.kinematic_dim]
        x_type = x[..., self.kinematic_dim :]

        emb_kin = self.continuous_network(x_kin)  # (N, L, 3/4 * embed_dim)
        emb_type = self.type_network(x_type)      # (N, L, 1/4 * embed_dim)

        # Concatenate continuous particle kinematics and type embeddings (N, L, embed_dim)
        return torch.cat([emb_kin, emb_type], dim=-1)


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
        # dimension of particle embedding
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

        # Apply custom attention mask (e.g. interaction matrix U)
        if attn_mask is not None:
            scores = scores + attn_mask

        # Apply padding mask (True means mask out/ignore)
        if key_padding_mask is not None:
            # Expand key_padding_mask from (N, L_k) to (N, 1, 1, L_k)
            scores_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(scores_mask, -1e9)
            
            # Mask out corresponding values
            v_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # (N, 1, L_k, 1)
            v = v.masked_fill(v_mask, 0.0)

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
    """Standard Particle Transformer module with separate particle kinematics (3/4 embed_dim)
    and type (1/4 embed_dim) embeddings concatenated into full embed_dim token representation.
    """

    def __init__(self, score_dim: int, parameters: dict):
        super().__init__()
        self.score_dim = score_dim
        self.model_params = parameters
        self.pt_log_scale = parameters.get('pt_log_scale', True)
        self.interaction_type = parameters.get('interaction_type', 'default')

        kinematic_dim = parameters.get('kinematic_dim', 4 if parameters.get('use_mass', False) else 3)
        num_channels = parameters.get('num_channels', parameters['ParEmbed']['input_dim'] - kinematic_dim)
        
        # Get target embed_dim (e.g. 32 for micro, 64 for light/wide/deep)
        embed_dim_cfg = parameters['ParEmbed']['embed_dim']
        if isinstance(embed_dim_cfg, list):
            target_embed_dim = embed_dim_cfg[-1]
        else:
            target_embed_dim = int(embed_dim_cfg)

        fc_dim = parameters['ParAtteBlock'].get('fc_dim', 256)

        # Particle Feature Embedding (continuous: 3/4 embed_dim, type: 1/4 embed_dim)
        self.par_embedding = ParticleFeatureEmbedding(
            kinematic_dim=kinematic_dim,
            num_channels=num_channels,
            embed_dim=target_embed_dim,
            fc_dim=fc_dim,
        )

        atte_embed_dim = self.par_embedding.out_dim  # Exactly equals target_embed_dim
        num_heads = parameters['ParAtteBlock']['num_heads']

        # Pairwise interaction embedding mapping (N, 3, L, L) -> (N, num_heads, L, L)
        self.inter_embedding = InteractionMatrixEmbedding(
            input_dim=3,
            embedding_dims=[64, 64, num_heads]
        )

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

        # Fill NaN values with zeros for numerical operations
        x = torch.where(key_padding_mask.unsqueeze(-1), torch.zeros_like(x), x)

        # Extract coordinate features (pt, eta, phi) for pairwise interaction
        coords = x[..., :3].clone()
        u = prepare_interaction(coords, pt_log_scale=self.pt_log_scale, interaction_type=self.interaction_type)  # (N, 3, L, L)
        attn_mask = self.inter_embedding(u)  # (N, num_heads, L, L)

        # Particle Embedding (Separately embeds kinematics -> 3/4 and type -> 1/4, concat -> full embed_dim)
        x = self.par_embedding(x)  # (N, L, E)

        # Particle Self-Attention blocks (passing attn_mask)
        for block in self.par_atte_blocks:
            x = block(x, x_clt=None, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

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

    def __init__(self, num_channels: int = 3, pt_log_scale: bool = True, interaction_type: str = "default", use_mass: bool = False):
        kinematic_dim = 4 if use_mass else 3
        hyperparameters = {
            "pt_log_scale": pt_log_scale,
            "interaction_type": interaction_type,
            "use_mass": use_mass,
            "kinematic_dim": kinematic_dim,
            "num_channels": num_channels,
            "ParEmbed": {
                "input_dim": kinematic_dim + num_channels,
                "embed_dim": 64
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

    def __init__(self, num_channels: int = 3, pt_log_scale: bool = True, interaction_type: str = "default", use_mass: bool = False):
        kinematic_dim = 4 if use_mass else 3
        hyperparameters = {
            "pt_log_scale": pt_log_scale,
            "interaction_type": interaction_type,
            "use_mass": use_mass,
            "kinematic_dim": kinematic_dim,
            "num_channels": num_channels,
            "ParEmbed": {
                "input_dim": kinematic_dim + num_channels,
                "embed_dim": 64
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


def create_model_from_config(config: Any, num_channels: int) -> ParticleTransformer:
    """Factory function to build ParticleTransformer model directly from config object/dict,
    
    supporting ParT_Baseline and ParT_Light as bases with direct YAML hyperparameter overrides.
    """
    model_type = getattr(config, "model_structure", "ParT_Light")
    pt_log_scale = getattr(config, "pt_log_scale", True)
    interaction_type = getattr(config, "interaction_type", "default")
    use_mass = getattr(config, "use_mass", False)
    kinematic_dim = 4 if use_mass else 3

    # Base defaults according to model_structure
    if model_type == "ParT_Baseline":
        base_params = {
            "pt_log_scale": pt_log_scale,
            "interaction_type": interaction_type,
            "use_mass": use_mass,
            "kinematic_dim": kinematic_dim,
            "num_channels": num_channels,
            "ParEmbed": {"input_dim": kinematic_dim + num_channels, "embed_dim": 64},
            "ParAtteBlock": {"num_heads": 8, "fc_dim": 512, "dropout": 0.1},
            "ClassAtteBlock": {"num_heads": 8, "fc_dim": 512, "dropout": 0.0},
            "num_ParAtteBlock": 6,
            "num_ClassAtteBlock": 2,
        }
    else:  # Default ParT_Light (or Custom based on ParT_Light)
        base_params = {
            "pt_log_scale": pt_log_scale,
            "interaction_type": interaction_type,
            "use_mass": use_mass,
            "kinematic_dim": kinematic_dim,
            "num_channels": num_channels,
            "ParEmbed": {"input_dim": kinematic_dim + num_channels, "embed_dim": 64},
            "ParAtteBlock": {"num_heads": 4, "fc_dim": 256, "dropout": 0.1},
            "ClassAtteBlock": {"num_heads": 4, "fc_dim": 256, "dropout": 0.0},
            "num_ParAtteBlock": 3,
            "num_ClassAtteBlock": 1,
        }

    # 1. Override using direct YAML top-level attributes if provided
    if getattr(config, "num_ParAtteBlock", None) is not None:
        base_params["num_ParAtteBlock"] = getattr(config, "num_ParAtteBlock")
    if getattr(config, "num_ClassAtteBlock", None) is not None:
        base_params["num_ClassAtteBlock"] = getattr(config, "num_ClassAtteBlock")
    if getattr(config, "num_heads", None) is not None:
        num_heads = getattr(config, "num_heads")
        base_params["ParAtteBlock"]["num_heads"] = num_heads
        base_params["ClassAtteBlock"]["num_heads"] = num_heads
    if getattr(config, "embed_dim", None) is not None:
        base_params["ParEmbed"]["embed_dim"] = getattr(config, "embed_dim")
    if getattr(config, "fc_dim", None) is not None:
        fc_dim = getattr(config, "fc_dim")
        base_params["ParAtteBlock"]["fc_dim"] = fc_dim
        base_params["ClassAtteBlock"]["fc_dim"] = fc_dim
    if getattr(config, "dropout", None) is not None:
        dropout = getattr(config, "dropout")
        base_params["ParAtteBlock"]["dropout"] = dropout

    # 2. Override using model_params dictionary if provided
    if hasattr(config, "model_params") and isinstance(config.model_params, dict):
        mp = config.model_params
        if "num_ParAtteBlock" in mp:
            base_params["num_ParAtteBlock"] = mp["num_ParAtteBlock"]
        if "num_ClassAtteBlock" in mp:
            base_params["num_ClassAtteBlock"] = mp["num_ClassAtteBlock"]
        if "num_heads" in mp:
            base_params["ParAtteBlock"]["num_heads"] = mp["num_heads"]
            base_params["ClassAtteBlock"]["num_heads"] = mp["num_heads"]
        if "embed_dim" in mp:
            base_params["ParEmbed"]["embed_dim"] = mp["embed_dim"]
        if "fc_dim" in mp:
            base_params["ParAtteBlock"]["fc_dim"] = mp["fc_dim"]
            base_params["ClassAtteBlock"]["fc_dim"] = mp["fc_dim"]
        if "dropout" in mp:
            base_params["ParAtteBlock"]["dropout"] = mp["dropout"]

    return ParticleTransformer(score_dim=1, parameters=base_params)


if __name__ == "__main__":
    from types import SimpleNamespace

    # Test forward pass with mock data
    print("Testing PyTorch Particle Transformer (ParT) models with 3/4 + 1/4 Embeddings...")
    mock_input_32 = torch.randn(4, 5, 6)
    mock_input_64 = torch.randn(4, 5, 6)

    # Test micro (embed_dim=32 -> kin:24, type:8)
    micro_cfg = SimpleNamespace(
        model_structure="ParT_Light",
        interaction_type="eta_phi_dr",
        pt_log_scale=True,
        num_ParAtteBlock=2,
        num_ClassAtteBlock=1,
        num_heads=2,
        embed_dim=32,
        fc_dim=128,
        dropout=0.1
    )
    model_micro = create_model_from_config(micro_cfg, num_channels=3)
    out_micro = model_micro(mock_input_32)
    assert out_micro.shape == (4, 1)
    print(f"  [PASS] exp_micro (embed_dim=32 -> kin:24, type:8) -> Output shape: {out_micro.shape}")

    # Test light/wide/deep (embed_dim=64 -> kin:48, type:16)
    light_cfg = SimpleNamespace(
        model_structure="ParT_Light",
        interaction_type="eta_phi_dr",
        pt_log_scale=True,
        num_ParAtteBlock=3,
        num_ClassAtteBlock=1,
        num_heads=4,
        embed_dim=64,
        fc_dim=256,
        dropout=0.1
    )
    model_light = create_model_from_config(light_cfg, num_channels=3)
    out_light = model_light(mock_input_64)
    assert out_light.shape == (4, 1)
    print(f"  [PASS] exp_light (embed_dim=64 -> kin:48, type:16) -> Output shape: {out_light.shape}")

    print("\nAll dynamic 3/4 + 1/4 embedding model tests passed successfully!")
