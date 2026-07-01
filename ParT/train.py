"""Training loop and utilities for WW polarization Particle Transformer (ParT).

Includes Trainer class, metrics calculation, and logging/checkpointing.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Add workspace directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DNN.train import (
    compute_weighted_loss,
    compute_roc_auc,
    compute_metrics,
    plot_loss_history,
    plot_auc_history,
    save_metrics_json,
)


class Trainer:
    """Training orchestrator for Particle Transformer (ParT) model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 0.001,
        max_epochs: int = 200,
        early_stopping_patience: int = 20,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        
        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # Warmup and LR Scheduler
        self.warmup_epochs = 5
        self.warmup_start_lr = 1e-6
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=max_epochs - self.warmup_epochs, 
            eta_min=1e-6
        )
        
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
        """Train for one epoch.

        Returns:
            (loss, roc_auc)
        """
        self.model.train()
        
        all_logits = []
        all_labels = []
        all_weights = []
        total_weighted_loss = 0.0
        total_weight_sum = 0.0
        
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
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
            
            # Gradient clipping to stabilize Transformer training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate global weighted loss and weight sum for tracking
            logits_sq = logits.squeeze(-1)
            labels_f = batch_labels.float()
            weights_f = batch_weights.float()
            unreduced_loss = bce_loss(logits_sq, labels_f)
            total_weighted_loss += (unreduced_loss * weights_f).sum().item()
            total_weight_sum += weights_f.sum().item()
            
            # Store for AUC computation
            all_logits.append(logits.detach().cpu().numpy().squeeze())
            all_labels.append(batch_labels.cpu().numpy())
            all_weights.append(batch_weights.cpu().numpy())
        
        # Average loss
        avg_loss = total_weighted_loss / max(total_weight_sum, 1e-12)
        
        # Compute ROC-AUC
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_weights = np.concatenate(all_weights)
        
        roc_auc = compute_roc_auc(all_logits, all_labels, all_weights)
        
        return avg_loss, roc_auc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model on validation set.

        Returns:
            (loss, roc_auc)
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_weights = []
        total_weighted_loss = 0.0
        total_weight_sum = 0.0
        
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        for batch_features, batch_labels, batch_weights, batch_event_ids in self.val_loader:
            # Move to device
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_weights = batch_weights.to(self.device)
            
            # Forward pass
            logits = self.model(batch_features)
            
            # Accumulate global weighted loss and weight sum for tracking
            logits_sq = logits.squeeze(-1)
            labels_f = batch_labels.float()
            weights_f = batch_weights.float()
            unreduced_loss = bce_loss(logits_sq, labels_f)
            total_weighted_loss += (unreduced_loss * weights_f).sum().item()
            total_weight_sum += weights_f.sum().item()
            
            # Store for AUC computation
            all_logits.append(logits.cpu().numpy().squeeze())
            all_labels.append(batch_labels.cpu().numpy())
            all_weights.append(batch_weights.cpu().numpy())
        
        # Average loss
        avg_loss = total_weighted_loss / max(total_weight_sum, 1e-12)
        
        # Compute ROC-AUC
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_weights = np.concatenate(all_weights)
        
        roc_auc = compute_roc_auc(all_logits, all_labels, all_weights)
        
        return avg_loss, roc_auc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a given data split."""
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

    @torch.no_grad()
    def get_predictions(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions (probabilities), true labels, and weights.
        
        Returns:
            Tuple of (probabilities, labels, weights)
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        all_weights = []
        
        for batch_features, batch_labels, batch_weights, batch_event_ids in loader:
            batch_features = batch_features.to(self.device)
            logits = self.model(batch_features)
            
            all_logits.append(logits.cpu().numpy().squeeze())
            all_labels.append(batch_labels.numpy())
            all_weights.append(batch_weights.numpy())
            
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        all_weights = np.concatenate(all_weights)
        
        probs = 1.0 / (1.0 + np.exp(-all_logits))
        
        return probs, all_labels, all_weights

    def save_checkpoint(self, checkpoint_path: str, epoch: Optional[int] = None) -> None:
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)
        if epoch is not None:
            print(f"Saved checkpoint to {checkpoint_path} (Epoch {epoch + 1})")
        else:
            print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        print(f"Loaded checkpoint from {checkpoint_path}")

    def train(self) -> Dict:
        """Full training loop with early stopping.

        Returns:
            Training history dictionary
        """
        print(f"Starting training for {self.max_epochs} epochs...")
        print(f"Early stopping patience: {self.early_stopping_patience} epochs\n")
        
        t_start = time.time()
        
        for epoch in range(self.max_epochs):
            # Apply linear warmup if in warmup epochs
            if epoch < self.warmup_epochs:
                lr = self.warmup_start_lr + (self.learning_rate - self.warmup_start_lr) * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
            # Train
            train_loss, train_auc = self.train_epoch()
            
            # Validate
            val_loss, val_auc = self.validate()
            
            # Step scheduler if warmup is finished
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
                
            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_roc_auc"].append(train_auc)
            self.history["val_loss"].append(val_loss)
            self.history["val_roc_auc"].append(val_auc)
            
            # Print progress every 10 epochs or at the first epoch
            if epoch == 0 or (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"LR: {current_lr:.2e} | "
                    f"Train Loss: {train_loss:.6f} AUC: {train_auc:.6f} | "
                    f"Val Loss: {val_loss:.6f} AUC: {val_auc:.6f}"
                )
            
            # Early stopping check
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_without_improvement = 0
                
                # Save best checkpoint
                self.save_checkpoint(self.best_checkpoint_path, epoch=epoch)
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(
                        f"\nEarly stopping triggered at epoch {epoch+1}. "
                        f"No improvement for {self.early_stopping_patience} epochs."
                    )
                    break
        
        t_end = time.time()
        self.history["duration_seconds"] = t_end - t_start
        
        # Load best model
        self.load_checkpoint(self.best_checkpoint_path)
        
        # Evaluate on test set
        test_metrics = self.evaluate(self.test_loader)
        self.history["test_roc_auc"] = test_metrics["roc_auc"]
        
        print(f"\nBest validation ROC-AUC: {self.best_val_auc:.6f}")
        print(f"Test ROC-AUC: {test_metrics['roc_auc']:.6f}")
        
        return self.history


def plot_score_distribution(
    probs: np.ndarray,
    labels: np.ndarray,
    task_name: str,
    output_path: str,
) -> None:
    """Plot score distribution for signal and background.
    
    Args:
        probs: Model predicted probabilities (n_samples,)
        labels: True labels (n_samples,) with values 0 or 1
        task_name: Name of the task (e.g. 'EW_vs_Background')
        output_path: Path to save the PDF plot
    """
    if task_name == "EW_vs_Background":
        xlabel = r"ParT$_{W^{\pm}W^{\pm}}$ Score"
        sig_label = r"$W^{\pm}W^{\pm}\text{-EW}$"
        bg_label = r"Backgrounds"
    elif task_name == "PolState_LL_vs_LT_TT":
        xlabel = r"ParT$_{\mathrm{pol}}$ Score"
        sig_label = r"$W_{\mathrm{L}}^{\pm}W_{\mathrm{L}}^{\pm}\text{-EW}$"
        bg_label = r"$W_{\mathrm{L}}^{\pm}W_{\mathrm{T}}^{\pm}\text{-EW} + W_{\mathrm{T}}^{\pm}W_{\mathrm{T}}^{\pm}\text{-EW}$"
    elif task_name == "PolState_LL_LT_vs_TT":
        xlabel = r"ParT$_{\mathrm{pol}}$ Score"
        sig_label = r"$W_{\mathrm{L}}^{\pm}W_{\mathrm{L}}^{\pm}\text{-EW} + W_{\mathrm{L}}^{\pm}W_{\mathrm{T}}^{\pm}\text{-EW}$"
        bg_label = r"$W_{\mathrm{T}}^{\pm}W_{\mathrm{T}}^{\pm}\text{-EW}$"
    else:
        xlabel = f"ParT Score ({task_name})"
        sig_label = "Signal"
        bg_label = "Background"
        
    probs_bg = probs[labels == 0]
    probs_sg = probs[labels == 1]
    
    bins = np.linspace(0.0, 1.0, 21)
    
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    
    ax.hist(
        probs_sg,
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2.0,
        color="#E63928",
        linestyle="solid",
        label=sig_label
    )
    
    ax.hist(
        probs_bg,
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2.0,
        color="#3378FF",
        linestyle="dashed",
        label=bg_label
    )
        
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized events")
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=1.2, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.0, top=True, right=True)
    ax.minorticks_on()
    ax.legend(frameon=False)

    fig.savefig(output_path, format="pdf")
    print(f"Saved score distribution plot to {output_path}")
    plt.close(fig)

