"""EventCNN model architecture implemented in PyTorch.

1:1 reproduction of the original TensorFlow EventCNN architecture from higgs_production/CNN:
- Input BatchNorm2d
- Block 1: Conv2D(32) -> Conv2D(64) -> MaxPool2D(3)
- Block 2: 2x Conv2D(64, same padding) with Residual Add
- Block 3: 2x Conv2D(64, same padding) with Residual Add
- Block 4: Conv2D(64) -> GlobalAveragePooling2D -> Flatten
- Dense Head: Linear(256) -> Dropout -> Linear(128) -> Dropout -> Linear(128) -> Linear(1)
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ResidualBlock(nn.Module):
    """Residual convolutional block with two 3x3 Conv2d layers with padding=1."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        return out + residual


class EventCNN(nn.Module):
    """Event-level Convolutional Neural Network for particle physics event images."""

    def __init__(
        self,
        in_channels: int = 3,
        n_filters: int = 64,
        dense_hidden_dim: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters

        # Input Normalization
        self.bn_input = nn.BatchNorm2d(in_channels)

        # Block 1: Initial feature extraction & spatial reduction
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters // 2, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters // 2, n_filters, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )

        # Block 2 & Block 3: Residual feature refinement
        self.res_block2 = ResidualBlock(channels=n_filters)
        self.res_block3 = ResidualBlock(channels=n_filters)

        # Block 4: Final convolution before global pooling
        self.conv_final = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(n_filters, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, dense_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_hidden_dim, dense_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(dense_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Logits of shape (batch_size, 1)
        """
        # (B, C, H, W)
        x = self.bn_input(x)
        x1 = self.block1(x)
        x2 = self.res_block2(x1)
        x3 = self.res_block3(x2)
        feat = self.conv_final(x3)
        logits = self.classifier(feat)
        return logits


def create_model_from_config(config: Any) -> EventCNN:
    """Instantiate EventCNN using config object or dictionary."""
    if hasattr(config, "to_dict"):
        cfg = config.to_dict()
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = config.__dict__

    in_channels = cfg.get("num_channels", 3)
    n_filters = cfg.get("n_CNN_filters", 64)
    dense_hidden_dim = cfg.get("dense_hidden_dim", 128)
    dropout_rate = cfg.get("dropout", 0.1)

    return EventCNN(
        in_channels=in_channels,
        n_filters=n_filters,
        dense_hidden_dim=dense_hidden_dim,
        dropout_rate=dropout_rate,
    )
