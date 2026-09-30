"""Smoke and verification test script for EventCNN pipeline.

Can run with synthetic data or a tiny slice of real Parquet data.
Usage:
    python CNN/test_cnn_pipeline.py --synthetic
    python CNN/test_cnn_pipeline.py --real --parquet_dir Sample/Parquet/batch_lowlevel_constituent_mg_sample
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add workspace directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CNN.model import EventCNN
from CNN.data_loader import pixelize_eflow_events, CNNFoldDataset
from CNN.train import CNNTrainer


def test_synthetic_pipeline():
    print("==================================================")
    print("Running Synthetic EventCNN Smoke Test...")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test Device: {device}")

    # 1. Test EventCNN architecture forward & backward pass
    batch_size = 8
    res = 40
    in_channels = 3
    x = torch.randn(batch_size, in_channels, res, res, device=device)
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=device)
    weights = torch.ones(batch_size, dtype=torch.float32, device=device)

    model = EventCNN(in_channels=in_channels, n_filters=64, dense_hidden_dim=128).to(device)
    logits = model(x).squeeze(-1)

    assert logits.shape == (batch_size,), f"Expected logits shape {(batch_size,)}, got {logits.shape}"
    print("[PASS] Model Forward pass shape verified.")

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, labels)
    loss.backward()
    print("[PASS] Model Backward pass and gradient computation verified.")

    # 2. Test Pixelization function
    n_events = 5
    mock_data = {
        "part_pt": [np.random.uniform(5, 100, size=20) for _ in range(n_events)],
        "part_eta": [np.random.uniform(-4.5, 4.5, size=20) for _ in range(n_events)],
        "part_phi": [np.random.uniform(-np.pi, np.pi, size=20) for _ in range(n_events)],
        "part_type": [np.random.choice([0, 1, 2], size=20) for _ in range(n_events)],
    }
    df = pd.DataFrame(mock_data)
    images = pixelize_eflow_events(df, res=res)
    assert images.shape == (n_events, 3, res, res), f"Expected shape {(n_events, 3, res, res)}, got {images.shape}"
    assert np.all(images >= 0.0), "Energy/pT entries should be non-negative"
    print("[PASS] EFlow Pixelization logic verified.")
    print("==================================================")
    print("All Synthetic Tests Passed Successfully!")
    print("==================================================")


def test_real_pipeline(parquet_dir: str):
    print("==================================================")
    print(f"Running Real Parquet EventCNN Test on: {parquet_dir}")
    print("==================================================")

    from CNN.data_loader import create_fold_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test Device: {device}")

    # Use a tiny fraction (1% or 2%) to test data pipeline quickly
    train_loader, val_loader, test_loader, meta = create_fold_loaders(
        parquet_dir=parquet_dir,
        i_fold=0,
        task="EW_vs_Background",
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        res=40,
        dataset_fraction=0.01,
        dataset_seed=42,
    )

    print(f"Loaded Real Subset: Train={meta['n_train']}, Val={meta['n_val']}, Test={meta['n_test']}")
    assert meta["n_train"] > 0, "Train dataset should not be empty!"

    # Fetch one batch
    for images, labels, weights, evts in train_loader:
        print(f"Batch shape: Images={images.shape}, Labels={labels.shape}, Weights={weights.shape}")
        assert images.shape[1:] == (3, 40, 40), f"Unexpected image shape: {images.shape}"
        break

    # Run 1 epoch of training
    model = EventCNN(in_channels=3, n_filters=64, dense_hidden_dim=128)
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.0001,
        max_epochs=1,
        early_stopping_patience=1,
        checkpoint_dir="./CNN/test_checkpoints",
    )

    results = trainer.train()
    print("==================================================")
    print("Real Data Pipeline Smoke Test Passed Successfully!")
    print(f"Test AUC: {results['metrics']['test_roc_auc']:.4f}")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic smoke test")
    parser.add_argument("--real", action="store_true", help="Run smoke test on actual Parquet data")
    parser.add_argument("--parquet_dir", type=str, default="Sample/Parquet/batch_lowlevel_constituent_mg_sample")
    args = parser.parse_args()

    if not args.synthetic and not args.real:
        # Default to synthetic if nothing specified
        test_synthetic_pipeline()
    else:
        if args.synthetic:
            test_synthetic_pipeline()
        if args.real:
            test_real_pipeline(args.parquet_dir)


if __name__ == "__main__":
    main()
