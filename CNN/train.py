"""Training loop and utilities for WW polarization EventCNN.

Includes CNNTrainer orchestrator, weighted evaluation, and checkpoint/plot management.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
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


def plot_score_distribution(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    weights: np.ndarray,
    save_path: str,
    title: str = "CNN Prediction Score Distribution",
    nbins: int = 50,
):
    """Plot weighted prediction score distribution for signal vs background."""
    plt.figure(figsize=(7, 5))
    sig_mask = (y_true == 1)
    bg_mask = (y_true == 0)

    bins = np.linspace(0.0, 1.0, nbins + 1)
    if np.any(sig_mask):
        plt.hist(
            y_scores[sig_mask],
            bins=bins,
            weights=weights[sig_mask],
            density=True,
            histtype="step",
            color="red",
            linewidth=2,
            label="Signal",
        )
    if np.any(bg_mask):
        plt.hist(
            y_scores[bg_mask],
            bins=bins,
            weights=weights[bg_mask],
            density=True,
            histtype="step",
            color="blue",
            linewidth=2,
            label="Background",
        )

    plt.xlabel("CNN Prediction Probability")
    plt.ylabel("Normalized Event Count")
    plt.title(title)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


class CNNTrainer:
    """Training orchestrator for EventCNN model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0,
        max_epochs: int = 200,
        early_stopping_patience: int = 15,
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
        self.weight_decay = weight_decay

        # Adam optimizer matching original params.json
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

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
        total_loss = 0.0
        total_weight = 0.0

        all_logits = []
        all_labels = []
        all_weights = []

        for images, labels, weights, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            weights = weights.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images).squeeze(-1)
            loss = compute_weighted_loss(logits, labels, weights)
            loss.backward()
            self.optimizer.step()

            batch_weight = weights.sum().item()
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_weights.append(weights.detach().cpu().numpy())

        avg_loss = total_loss / (total_weight + 1e-8)
        logits_arr = np.concatenate(all_logits)
        labels_arr = np.concatenate(all_labels).astype(np.int64)
        weights_arr = np.concatenate(all_weights)

        auc = compute_roc_auc(logits_arr, labels_arr, weights_arr)
        return avg_loss, auc

    def evaluate(self, loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model on given DataLoader.

        Returns:
            (loss, roc_auc, y_true, y_scores, weights)
        """
        self.model.eval()
        total_loss = 0.0
        total_weight = 0.0

        all_logits = []
        all_labels = []
        all_weights = []

        with torch.no_grad():
            for images, labels, weights, _ in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                weights = weights.to(self.device, non_blocking=True)

                logits = self.model(images).squeeze(-1)
                loss = compute_weighted_loss(logits, labels, weights)

                batch_weight = weights.sum().item()
                total_loss += loss.item() * batch_weight
                total_weight += batch_weight

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_weights.append(weights.cpu().numpy())

        avg_loss = total_loss / (total_weight + 1e-8)
        logits_arr = np.concatenate(all_logits)
        labels_arr = np.concatenate(all_labels).astype(np.int64)
        weights_arr = np.concatenate(all_weights)

        auc = compute_roc_auc(logits_arr, labels_arr, weights_arr)
        y_scores = 1.0 / (1.0 + np.exp(-logits_arr))

        return avg_loss, auc, labels_arr, y_scores, weights_arr

    def train(self) -> Dict[str, Any]:
        """Run complete training with early stopping."""
        start_time = time.time()
        print(f"Starting EventCNN training on {self.device} (Max Epochs: {self.max_epochs})")

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_auc = self.train_epoch()
            val_loss, val_auc, _, _, _ = self.evaluate(self.val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_roc_auc"].append(train_auc)
            self.history["val_roc_auc"].append(val_auc)

            print(
                f"Epoch [{epoch:03d}/{self.max_epochs:03d}] "
                f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} || "
                f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

            # Checkpoint on best validation AUC
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_auc": val_auc,
                    },
                    self.best_checkpoint_path,
                )
                print(f"  --> Saved new best model checkpoint (Val AUC: {val_auc:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs (Patience: {self.early_stopping_patience})")
                    break

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f} seconds. Evaluating best checkpoint on test set...")

        # Load best model for testing
        if os.path.exists(self.best_checkpoint_path):
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best checkpoint from epoch {checkpoint.get('epoch')}")

        test_loss, test_auc, y_test, y_test_scores, test_weights = self.evaluate(self.test_loader)
        self.history["test_roc_auc"] = test_auc
        print(f"==================================================")
        print(f"Final Test Evaluation: Test Loss = {test_loss:.4f} | Test AUC = {test_auc:.4f}")
        print(f"==================================================")

        # Plot training curves
        plot_loss_history(
            self.history,
            output_path=os.path.join(self.checkpoint_dir, "loss_history.pdf"),
        )
        plot_auc_history(
            self.history,
            output_path=os.path.join(self.checkpoint_dir, "auc_history.pdf"),
        )
        plot_score_distribution(
            y_test,
            y_test_scores,
            test_weights,
            save_path=os.path.join(self.checkpoint_dir, "score_distribution.png"),
        )

        # Export metrics.json
        metrics = {
            "best_val_auc": float(self.best_val_auc),
            "test_roc_auc": float(test_auc),
            "test_loss": float(test_loss),
            "epochs_trained": len(self.history["train_loss"]),
            "training_time_sec": float(total_time),
        }
        save_metrics_json(metrics, os.path.join(self.checkpoint_dir, "metrics.json"))

        return {
            "metrics": metrics,
            "history": self.history,
            "y_test": y_test,
            "y_test_scores": y_test_scores,
            "test_weights": test_weights,
        }
