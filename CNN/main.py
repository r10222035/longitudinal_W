"""Main entry point for WW polarization EventCNN training.

Supports single fold and full 5-fold cross-validation with YAML configuration.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import yaml
import json

import numpy as np
import torch

# Add workspace and DNN directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DNN"))

from CNN.model import create_model_from_config
from CNN.data_loader import create_fold_loaders
from CNN.train import CNNTrainer


class CNNTrainingConfig:
    """Hyperparameter configuration for EventCNN training."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def load_cnn_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> CNNTrainingConfig:
    """Load config from YAML with optional overrides."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config_data[key] = value

    return CNNTrainingConfig(**config_data)


def run_single_fold(
    config: CNNTrainingConfig,
    i_fold: int,
    device: torch.device,
    base_output_dir: str,
) -> Dict[str, Any]:
    """Train and evaluate EventCNN on a single fold."""
    fold_dir = os.path.join(base_output_dir, f"fold_{i_fold}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"  Starting EventCNN Fold {i_fold} / Task: {config.task}")
    print(f"  Output Directory: {fold_dir}")
    print(f"=======================================================")

    # Create DataLoaders
    train_loader, val_loader, test_loader, meta = create_fold_loaders(
        parquet_dir=config.parquet_dir,
        i_fold=i_fold,
        task=config.task,
        batch_size=getattr(config, "batch_size", 128),
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=getattr(config, "pin_memory", True),
        weight_strategy=getattr(config, "weight_strategy", "hybrid"),
        balance_weights=getattr(config, "balance_signal_background_weights", True),
        res=getattr(config, "res", 40),
        normalize_per_event=getattr(config, "normalize_per_event", False),
        clean_duplicates=getattr(config, "clean_duplicates", True),
        dataset_fraction=getattr(config, "dataset_fraction", 1.0),
        dataset_seed=getattr(config, "dataset_seed", 42),
    )

    print(f"  Events: Train={meta['n_train']}, Val={meta['n_val']}, Test={meta['n_test']}")

    # Build EventCNN
    model = create_model_from_config(config)

    # Initialize Trainer
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=getattr(config, "learning_rate", 0.0001),
        weight_decay=getattr(config, "weight_decay", 0.0),
        max_epochs=getattr(config, "max_epochs", 200),
        early_stopping_patience=getattr(config, "early_stopping_patience", 15),
        checkpoint_dir=fold_dir,
    )

    results = trainer.train()
    return results["metrics"]


def main():
    parser = argparse.ArgumentParser(description="WW Polarization EventCNN Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--fold", type=int, default=None, help="Run specific fold (0-4). If None, runs all 5 folds")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU if available")
    parser.add_argument("--timestamp", type=str, default=None, help="Optional timestamp session tag")
    parser.add_argument("--parquet_dir", type=str, default=None, help="Override Parquet directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--res", type=int, default=None, help="Override image resolution")

    args = parser.parse_args()

    overrides = {
        "parquet_dir": args.parquet_dir,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "res": args.res,
    }

    config = load_cnn_config(args.config, overrides=overrides)

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"Active Device: {device}")

    ts = args.timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = Path(getattr(config, "output_dir", "./CNN/results"))

    parts = list(raw_output_dir.parts)
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            top_folder = parts[idx + 1]
            if not (top_folder.startswith(ts) or f"{ts}_" in top_folder):
                parts[idx + 1] = f"{ts}_{top_folder}"
        else:
            task_tag = str(getattr(config, "task", "task")).lower()
            parts.append(f"{ts}_{task_tag}")
    else:
        top_folder = parts[-1]
        if not (top_folder.startswith(ts) or f"{ts}_" in top_folder):
            parts[-1] = f"{ts}_{top_folder}"

    base_out = str(Path(*parts))
    os.makedirs(base_out, exist_ok=True)

    # Save active configuration
    config_dump_path = os.path.join(base_out, "run_config.json")
    with open(config_dump_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    folds_to_run = [args.fold] if args.fold is not None else list(range(5))
    all_fold_metrics = {}

    for f_idx in folds_to_run:
        m = run_single_fold(
            config=config,
            i_fold=f_idx,
            device=device,
            base_output_dir=base_out,
        )
        all_fold_metrics[f"fold_{f_idx}"] = m

    # Save cross-validation summary
    summary_path = os.path.join(base_out, "cv_summary.json")
    test_aucs = [m["test_roc_auc"] for m in all_fold_metrics.values()]
    summary = {
        "folds": all_fold_metrics,
        "mean_test_auc": float(np.mean(test_aucs)),
        "std_test_auc": float(np.std(test_aucs)) if len(test_aucs) > 1 else 0.0,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("EventCNN Cross-Validation Complete!")
    print(f"Summary: Mean Test AUC = {summary['mean_test_auc']:.4f} +/- {summary['std_test_auc']:.4f}")
    print(f"Results saved to: {base_out}")
    print("=======================================================")


if __name__ == "__main__":
    main()
