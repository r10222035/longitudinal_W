"""
Training loop and utilities for WW polarization DNN.

Includes loss computation with sample weights, metrics, and checkpoint management.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from model import PolarizationDNN


# ============================================================================
# LOSS AND METRICS
# ============================================================================

def compute_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute sample-weighted binary cross-entropy loss.
    
    Physics weights are applied per-sample before averaging.
    
    Args:
        logits: Model outputs (batch_size, 1) or (batch_size,)
        labels: Target labels (batch_size,) with values 0 or 1
        weights: Physics cross-section weights (batch_size,)
    
    Returns:
        Scalar loss tensor
    """
    # Ensure shapes are compatible
    logits = logits.squeeze(-1)  # (batch_size,)
    labels = labels.float()       # (batch_size,)
    weights = weights.float()     # (batch_size,)
    
    # Binary cross-entropy with logits (per-sample, unreduced)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    unreduced_loss = bce_loss(logits, labels)  # (batch_size,)
    
    # Apply physics weights
    weighted_loss = unreduced_loss * weights

    # Weighted mean keeps the optimization scale stable while preserving
    # the intended relative weighting between samples.
    loss = weighted_loss.sum() / torch.clamp(weights.sum(), min=1e-12)
    
    return loss


def compute_roc_auc(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute ROC-AUC score with optional sample weighting.
    
    Args:
        logits: Model logits (n_samples,)
        labels: Target labels (n_samples,) with values 0 or 1
        weights: Optional sample weights for sklearn (n_samples,)
    
    Returns:
        ROC-AUC score (or 0.5 if only one class present)
    """
    # Handle case where only one class is present
    if len(np.unique(labels)) < 2:
        return 0.5  # Undefined ROC-AUC; return neutral value
    
    # Convert logits to probabilities via sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # Compute ROC-AUC
    auc = roc_auc_score(labels, probs, sample_weight=weights)
    
    return auc


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute multiple metrics for evaluation.
    
    Args:
        logits: Model logits (n_samples,)
        labels: Target labels (n_samples,)
        weights: Optional sample weights (n_samples,)
    
    Returns:
        Dictionary with metrics
    """
    roc_auc = compute_roc_auc(logits, labels, weights)
    
    metrics = {
        "roc_auc": roc_auc,
    }
    
    return metrics


# ============================================================================
# TRAINING LOOP
# ============================================================================

class Trainer:
    """Training orchestrator for DNN model."""
    
    def __init__(
        self,
        model: PolarizationDNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 0.001,
        max_epochs: int = 200,
        early_stopping_patience: int = 20,
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Args:
            model: DNN model to train
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            device: Torch device
            learning_rate: Adam learning rate
            max_epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience (epochs)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # Checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_roc_auc": [],
            "val_roc_auc": [],
            "test_roc_auc": None,
        }
        
        self.best_val_auc = -1.0
        self.epochs_without_improvement = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            (loss, roc_auc)
        """
        self.model.train()
        
        all_logits = []
        all_labels = []
        all_weights = []
        total_loss = 0.0
        n_batches = 0
        
        for batch_features, batch_labels, batch_weights, batch_event_ids in self.train_loader:
            # Move to device
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_weights = batch_weights.to(self.device)
            
            # Forward pass
            logits = self.model(batch_features)
            
            # Compute loss
            loss = compute_weighted_loss(logits, batch_labels, batch_weights)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            n_batches += 1
            
            # Store for AUC computation
            all_logits.append(logits.detach().cpu().numpy().squeeze())
            all_labels.append(batch_labels.cpu().numpy())
            all_weights.append(batch_weights.cpu().numpy())
        
        # Average loss
        avg_loss = total_loss / n_batches
        
        # Compute ROC-AUC
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_weights = np.concatenate(all_weights)
        
        roc_auc = compute_roc_auc(all_logits, all_labels, all_weights)
        
        return avg_loss, roc_auc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate model on validation set.
        
        Returns:
            (loss, roc_auc)
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_weights = []
        total_loss = 0.0
        n_batches = 0
        
        for batch_features, batch_labels, batch_weights, batch_event_ids in self.val_loader:
            # Move to device
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_weights = batch_weights.to(self.device)
            
            # Forward pass
            logits = self.model(batch_features)
            
            # Compute loss
            loss = compute_weighted_loss(logits, batch_labels, batch_weights)
            total_loss += loss.item()
            n_batches += 1
            
            # Store for AUC computation
            all_logits.append(logits.cpu().numpy().squeeze())
            all_labels.append(batch_labels.cpu().numpy())
            all_weights.append(batch_weights.cpu().numpy())
        
        # Average loss
        avg_loss = total_loss / n_batches
        
        # Compute ROC-AUC
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_weights = np.concatenate(all_weights)
        
        roc_auc = compute_roc_auc(all_logits, all_labels, all_weights)
        
        return avg_loss, roc_auc
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "test") -> Dict[str, float]:
        """
        Evaluate model on a given data split.
        
        Args:
            loader: DataLoader (test or val)
            split_name: Name of split for logging
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_weights = []
        
        for batch_features, batch_labels, batch_weights, batch_event_ids in loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_weights = batch_weights.to(self.device)
            
            logits = self.model(batch_features)
            
            all_logits.append(logits.cpu().numpy().squeeze())
            all_labels.append(batch_labels.cpu().numpy())
            all_weights.append(batch_weights.cpu().numpy())
        
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_weights = np.concatenate(all_weights)
        
        metrics = compute_metrics(all_logits, all_labels, all_weights)
        
        return metrics
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self) -> Dict:
        """
        Full training loop with early stopping.
        
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {self.max_epochs} epochs...")
        print(f"Early stopping patience: {self.early_stopping_patience} epochs\n")
        
        for epoch in range(self.max_epochs):
            # Train
            train_loss, train_auc = self.train_epoch()
            
            # Validate
            val_loss, val_auc = self.validate()
            
            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_roc_auc"].append(train_auc)
            self.history["val_loss"].append(val_loss)
            self.history["val_roc_auc"].append(val_auc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"Train Loss: {train_loss:.6f} AUC: {train_auc:.6f} | "
                    f"Val Loss: {val_loss:.6f} AUC: {val_auc:.6f}"
                )
            
            # Early stopping check
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_without_improvement = 0
                
                # Save best checkpoint
                self.save_checkpoint(self.best_checkpoint_path)
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(
                        f"\nEarly stopping triggered at epoch {epoch+1}. "
                        f"No improvement for {self.early_stopping_patience} epochs."
                    )
                    break
        
        # Load best model
        self.load_checkpoint(self.best_checkpoint_path)
        
        # Evaluate on test set
        test_metrics = self.evaluate(self.test_loader, "test")
        self.history["test_roc_auc"] = test_metrics["roc_auc"]
        
        print(f"\nBest validation ROC-AUC: {self.best_val_auc:.6f}")
        print(f"Test ROC-AUC: {test_metrics['roc_auc']:.6f}")
        
        return self.history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_loss_history(history: Dict, output_path: str) -> None:
    """
    Plot training and validation loss.
    
    Args:
        history: Training history dictionary from Trainer
        output_path: Path to save the PDF plot
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_auc_history(history: Dict, output_path: str) -> None:
    """
    Plot training and validation ROC-AUC.
    
    Args:
        history: Training history dictionary from Trainer
        output_path: Path to save the PDF plot
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(history["train_roc_auc"], label="Train AUC", linewidth=2)
    ax.plot(history["val_roc_auc"], label="Val AUC", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Training and Validation AUC")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def save_metrics_json(metrics: Dict, output_path: str) -> None:
    """Save metrics to JSON file, recursively converting numpy types."""
    def convert_to_json_compatible(obj):
        """Recursively convert numpy types to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_compatible(v) for v in obj]
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return obj
    
    json_metrics = convert_to_json_compatible(metrics)
    
    with open(output_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"Saved metrics to {output_path}")
