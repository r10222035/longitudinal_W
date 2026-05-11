"""
Neural network model for WW polarization state classification.

Implements fully-connected feed-forward DNN with Swish (SiLU) activation,
following Ref. [96] weight initialization strategy.
"""

import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.init as init


# ============================================================================
# WEIGHT INITIALIZATION (Ref. [96])
# ============================================================================

def init_swish_weights(
    module: nn.Module,
    var_scale: float = 2.952,
) -> None:
    """
    Initialize weights and biases following Ref. [96] for Swish/SiLU networks.
    
    - Weights: Truncated normal distribution with variance = var_scale / n_in
    - Biases: Normal distribution with std = 0.2
    
    Args:
        module: PyTorch module to initialize
        var_scale: Variance scaling factor (default 2.952 from paper)
    """
    if isinstance(module, nn.Linear):
        n_in = module.in_features
        
        # Truncated normal for weights
        # std = sqrt(2.952 / n_in), truncated at ±2σ
        std = math.sqrt(var_scale / n_in)
        init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-2.0,  # lower bound in std units
            b=2.0,   # upper bound in std units
        )
        
        # Normal distribution for biases
        if module.bias is not None:
            init.normal_(module.bias, mean=0.0, std=0.2)


# ============================================================================
# NORMALIZATION LAYER AS NN MODULE
# ============================================================================

class NormalizationLayer(nn.Module):
    """
    Feature normalization layer that stores fitted statistics as buffers.
    
    This layer normalizes input features using pre-fitted mean and std.
    Statistics are typically fit on the training set and frozen during training.
    """
    
    def __init__(self, n_features: int, mean: Optional[torch.Tensor] = None, 
                 std: Optional[torch.Tensor] = None):
        """
        Args:
            n_features: Number of input features
            mean: Pre-fitted mean (optional, can be set later via set_statistics)
            std: Pre-fitted standard deviation (optional, can be set later via set_statistics)
        """
        super().__init__()
        
        if mean is not None:
            assert mean.shape == (n_features,), "mean shape mismatch"
            self.register_buffer("mean", mean)
        else:
            self.register_buffer("mean", torch.zeros(n_features))
        
        if std is not None:
            assert std.shape == (n_features,), "std shape mismatch"
            self.register_buffer("std", std)
        else:
            self.register_buffer("std", torch.ones(n_features))
    
    def set_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Update normalization statistics."""
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize features."""
        return (x - self.mean) / (self.std + 1e-8)


# ============================================================================
# MAIN MODEL
# ============================================================================

class PolarizationDNN(nn.Module):
    """
    Fully-connected feed-forward DNN for WW polarization state classification.
    
    Architecture:
    - Input layer: Feature normalization
    - Hidden layers: Linear(w, w) -> BatchNorm1d -> SiLU -> Dropout (repeated)
    - Output layer: Linear(w, 1) for binary classification logit
    
    Weight initialization follows Ref. [96]:
    - Weights: truncated normal with var = 2.952 / n_in
    - Biases: normal with std = 0.2
    """
    
    def __init__(
        self,
        n_features: int = 32,
        hidden_width: int = 128,
        n_hidden_layers: int = 4,
        dropout_rate: float = 0.3,
        init_var_scale: float = 2.952,
    ):
        """
        Args:
            n_features: Number of input features (default 32 for physics features)
            hidden_width: Number of neurons in each hidden layer (constant width)
            n_hidden_layers: Number of hidden layer blocks
            dropout_rate: Dropout probability
            init_var_scale: Variance scaling for weight initialization (Ref. [96])
        """
        super().__init__()
        
        self.n_features = n_features
        self.hidden_width = hidden_width
        self.n_hidden_layers = n_hidden_layers
        self.dropout_rate = dropout_rate
        
        # Normalization layer (statistics will be set after fitting on train set)
        self.normalization = NormalizationLayer(n_features)
        
        # Hidden layers
        self.hidden_blocks = nn.ModuleList()
        
        for i in range(n_hidden_layers):
            in_features = n_features if i == 0 else hidden_width
            out_features = hidden_width
            
            block = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.SiLU(),  # Swish activation
                nn.Dropout(dropout_rate),
            )
            self.hidden_blocks.append(block)
        
        # Output layer: binary classification logit
        self.output_layer = nn.Linear(hidden_width, 1)
        
        # Initialize weights following Ref. [96]
        self._init_weights(init_var_scale)
    
    def _init_weights(self, var_scale: float = 2.952) -> None:
        """Initialize all weights and biases."""
        for block in self.hidden_blocks:
            for module in block.modules():
                init_swish_weights(module, var_scale)
        
        init_swish_weights(self.output_layer, var_scale)
    
    def set_normalization_statistics(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """
        Set the normalization statistics (typically fitted on training set).
        
        Args:
            mean: Feature means (shape: n_features)
            std: Feature standard deviations (shape: n_features)
        """
        self.normalization.set_statistics(mean, std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, n_features)
        
        Returns:
            logits: Binary classification logits (batch_size, 1)
        """
        # Apply normalization
        x = self.normalization(x)
        
        # Pass through hidden blocks
        for block in self.hidden_blocks:
            x = block(x)
        
        # Output layer
        logits = self.output_layer(x)
        
        return logits


# ============================================================================
# MODEL CREATION HELPER
# ============================================================================

def create_model(
    n_features: int = 32,
    hidden_width: int = 128,
    n_hidden_layers: int = 4,
    dropout_rate: float = 0.3,
    init_var_scale: float = 2.952,
    device: torch.device = torch.device("cpu"),
) -> PolarizationDNN:
    """
    Create a PolarizationDNN model on specified device.
    
    Args:
        n_features: Number of input features
        hidden_width: Width of hidden layers
        n_hidden_layers: Number of hidden layer blocks
        dropout_rate: Dropout probability
        init_var_scale: Variance scaling for weight initialization
        device: Torch device (cpu or cuda)
    
    Returns:
        Initialized model on specified device
    """
    model = PolarizationDNN(
        n_features=n_features,
        hidden_width=hidden_width,
        n_hidden_layers=n_hidden_layers,
        dropout_rate=dropout_rate,
        init_var_scale=init_var_scale,
    )
    model = model.to(device)
    return model
