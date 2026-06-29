"""Main entry point for WW polarization Particle Transformer (ParT) training.

Orchestrates 5-fold cross-validation with PyTorch ParT training, validation, and testing.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import datetime
import time
import yaml

import numpy as np
import torch

# Add workspace and DNN directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DNN"))

from ParT.data_loader import create_fold_loaders
from ParT.model import ParT_Light, ParT_Baseline
from ParT.train import Trainer, plot_auc_history, plot_loss_history, save_metrics_json

from DNN.config import TASK_DEFINITIONS


class ParTTrainingConfig:
    """Hyperparameter configuration for Particle Transformer training."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def load_part_config(config_path: str, overrides: Dict[str, Any] = None) -> ParTTrainingConfig:
    """Load config from YAML with runtime overrides."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
        
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config_data[key] = value
                
    return ParTTrainingConfig(**config_data)


def get_dataset_sizes(dataset, config: ParTTrainingConfig) -> Dict[str, int]:
    """Count signal and background samples in the dataset using cached labels."""
    task_def = TASK_DEFINITIONS.get(config.task)
    sig_label = task_def["signal_label"] if task_def else 1
    bg_label = task_def["background_label"] if task_def else 0
    
    if hasattr(dataset, "datasets"):
        datasets = dataset.datasets
    else:
        datasets = [dataset]
        
    sig_count = 0
    bg_count = 0
    for ds in datasets:
        if hasattr(ds, "labels"):
            labels = ds.labels
            sig_count += int(np.sum(labels == sig_label))
            bg_count += int(np.sum(labels == bg_label))
            
    return {
        "signal": sig_count,
        "background": bg_count,
        "total": sig_count + bg_count,
    }


def train_single_fold(
    i_fold: int,
    config: ParTTrainingConfig,
    device: torch.device,
) -> Dict:
    """Train and evaluate ParT model for a single fold."""
    print(f"\n{'='*70}")
    print(f"FOLD {i_fold} / 5")
    print(f"{'='*70}\n")
    
    # Create output directory for this fold
    fold_output_dir = Path(config.output_dir) / f"fold_{i_fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = fold_output_dir / "checkpoints"
    
    # Create data loaders
    print(f"Loading data for fold {i_fold}...")
    train_loader, val_loader, test_loader = create_fold_loaders(
        parquet_dir=config.parquet_dir,
        i_fold=i_fold,
        task=config.task,
        weight_strategy=config.weight_strategy,
        pt_log_scale=config.pt_log_scale,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        balance_weights=config.balance_signal_background_weights,
    )
    
    print(
        f"Train: {len(train_loader.dataset)} samples | "
        f"Val: {len(val_loader.dataset)} samples | "
        f"Test: {len(test_loader.dataset)} samples\n"
    )
    
    # Create model
    print("Creating ParT model...")
    if config.model_structure == "ParT_Light":
        model = ParT_Light(num_channels=config.num_channels)
    elif config.model_structure == "ParT_Baseline":
        model = ParT_Baseline(num_channels=config.num_channels)
    else:
        raise ValueError(f"Unknown model structure: {config.model_structure}")
        
    print(f"Model architecture:")
    print(f"  - Model Type: {config.model_structure}")
    print(f"  - Input channels (types): {config.num_channels}")
    print(f"  - pt_log_scale: {config.pt_log_scale}\n")
    
    # Train model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    print("Starting training...\n")
    history = trainer.train()
    
    # Save results
    history_path = fold_output_dir / "training_history.json"
    save_metrics_json(history, str(history_path))
    
    # Plot training history
    loss_plot_path = fold_output_dir / "loss_history.pdf"
    auc_plot_path = fold_output_dir / "auc_history.pdf"
    plot_loss_history(history, str(loss_plot_path))
    plot_auc_history(history, str(auc_plot_path))
    
    # Prepare fold results
    fold_results = {
        "fold": i_fold,
        "best_val_roc_auc": trainer.best_val_auc,
        "test_roc_auc": history["test_roc_auc"],
        "output_dir": str(fold_output_dir),
        "data_size": {
            "train": get_dataset_sizes(train_loader.dataset, config),
            "val": get_dataset_sizes(val_loader.dataset, config),
            "test": get_dataset_sizes(test_loader.dataset, config),
        }
    }
    
    return fold_results


def run_cross_validation(config: ParTTrainingConfig, device: torch.device) -> None:
    """Run full 5-fold cross-validation."""
    start_time = datetime.datetime.now()
    t0 = time.time()

    print(f"\n{'='*70}")
    print("WW POLARIZATION STATE ParT - 5-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Task: {TASK_DEFINITIONS[config.task]['name']}")
    print(f"Weight strategy: {config.weight_strategy}")
    print(f"Device: {device}")
    print(f"Parquet dir: {config.parquet_dir}")
    print(f"Output dir: {config.output_dir}\n")
    
    # Set random seed for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train each fold
    all_results = []
    
    for i_fold in range(5):
        fold_results = train_single_fold(i_fold, config, device)
        all_results.append(fold_results)
    
    # Print cross-validation summary
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}\n")
    
    val_aucs = [r["best_val_roc_auc"] for r in all_results]
    test_aucs = [r["test_roc_auc"] for r in all_results]
    
    print("Per-fold results:")
    for result in all_results:
        print(
            f"  Fold {result['fold']}: "
            f"Val ROC-AUC = {result['best_val_roc_auc']:.6f}, "
            f"Test ROC-AUC = {result['test_roc_auc']:.6f}"
        )
    
    print(f"\nValidation ROC-AUC: {np.mean(val_aucs):.6f} ± {np.std(val_aucs):.6f}")
    print(f"Test ROC-AUC:       {np.mean(test_aucs):.6f} ± {np.std(test_aucs):.6f}\n")
    
    # Compute timing information
    end_time = datetime.datetime.now()
    duration_seconds = time.time() - t0
    m, s = divmod(int(duration_seconds), 60)
    h, m = divmod(m, 60)
    duration_formatted = f"{h}:{m:02d}:{s:02d}"
    
    time_info = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": float(duration_seconds),
        "duration_formatted": duration_formatted,
    }
    
    # Aggregate data sizes
    fold0_ds = all_results[0]["data_size"]
    total_signal = fold0_ds["train"]["signal"] + fold0_ds["val"]["signal"] + fold0_ds["test"]["signal"]
    total_background = fold0_ds["train"]["background"] + fold0_ds["val"]["background"] + fold0_ds["test"]["background"]
    total_samples = total_signal + total_background
    
    train_signals = [r["data_size"]["train"]["signal"] for r in all_results]
    train_backgrounds = [r["data_size"]["train"]["background"] for r in all_results]
    train_totals = [r["data_size"]["train"]["total"] for r in all_results]
    
    val_signals = [r["data_size"]["val"]["signal"] for r in all_results]
    val_backgrounds = [r["data_size"]["val"]["background"] for r in all_results]
    val_totals = [r["data_size"]["val"]["total"] for r in all_results]
    
    test_signals = [r["data_size"]["test"]["signal"] for r in all_results]
    test_backgrounds = [r["data_size"]["test"]["background"] for r in all_results]
    test_totals = [r["data_size"]["test"]["total"] for r in all_results]
    
    data_size_summary = {
        "total_dataset": {
            "signal": total_signal,
            "background": total_background,
            "total": total_samples
        },
        "train_mean": {
            "signal": float(np.mean(train_signals)),
            "background": float(np.mean(train_backgrounds)),
            "total": float(np.mean(train_totals))
        },
        "val_mean": {
            "signal": float(np.mean(val_signals)),
            "background": float(np.mean(val_backgrounds)),
            "total": float(np.mean(val_totals))
        },
        "test_mean": {
            "signal": float(np.mean(test_signals)),
            "background": float(np.mean(test_backgrounds)),
            "total": float(np.mean(test_totals))
        }
    }
    
    # Save summary
    summary = {
        "task": config.task,
        "n_folds": 5,
        "time_info": time_info,
        "data_size": data_size_summary,
        "config": config.to_dict(),
        "results": all_results,
        "summary": {
            "val_roc_auc_mean": float(np.mean(val_aucs)),
            "val_roc_auc_std": float(np.std(val_aucs)),
            "test_roc_auc_mean": float(np.mean(test_aucs)),
            "test_roc_auc_std": float(np.std(test_aucs)),
        },
    }
    
    summary_path = Path(config.output_dir) / "cv_summary.json"
    save_metrics_json(summary, str(summary_path))
    
    print(f"Summary saved to {summary_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train WW polarization ParT with 5-fold cross-validation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="ParT/configs/default_config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default=None,
        help="Override the Parquet batch directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "--weight_strategy",
        type=str,
        default="hybrid",
        choices=["hybrid"],
        help="Sample-weight strategy to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override maximum number of training epochs",
    )

    args = parser.parse_args()

    config = load_part_config(
        config_path=args.config,
        overrides={
            "parquet_dir": args.parquet_dir,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "weight_strategy": args.weight_strategy,
            "max_epochs": args.epochs,
        },
    )

    # Align with the directory naming structure of DNN
    config.output_dir = str(Path(config.output_dir) / config.weight_strategy)

    # Determine device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Run training
    run_cross_validation(config, device)


if __name__ == "__main__":
    main()
