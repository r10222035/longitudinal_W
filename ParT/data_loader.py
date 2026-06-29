"""Data loading module for WW polarization ParT training.

Handles Parquet file loading, deterministic 5-fold cross-validation,
event reconstruction into sequences of shape (5, 6), and PyTorch Dataset/DataLoader creation.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Add workspace and DNN directory to python path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DNN"))

from DNN.config import (
    get_process_label_and_weight,
    compute_sample_weight,
    balance_signal_background_weights,
    TASK_DEFINITIONS,
)
from DNN.data_loader import (
    get_all_parquet_files,
    load_and_merge_parquet,
)


class ParTFoldDataset(Dataset):
    """PyTorch Dataset for Parquet files with deterministic 5-fold splitting and

    reconstruction of event features into 5-particle sequence format.
    
    Reconstructs each event into a (5, 6) feature matrix:
    - Row 0: Lepton 1 [pt, eta, 0.0, 1.0, 0.0, 0.0]
    - Row 1: Lepton 2 [pt, eta, dphi_l2_l1, 1.0, 0.0, 0.0]
    - Row 2: Jet 1    [pt, eta, dphi_j1_l1, 0.0, 1.0, 0.0]
    - Row 3: Jet 2    [pt, eta, dphi_j2_l1, 0.0, 1.0, 0.0]
    - Row 4: MET      [pt, 0.0, dphi_met_l1, 0.0, 0.0, 1.0]
    """

    def __init__(
        self,
        parquet_file_paths: List[str],
        process_name: str,
        i_fold: int = 0,
        fold_type: str = "train",
        task: str = "EW_vs_Background",
        weight_strategy: str = "hybrid",
        pt_log_scale: bool = True,
    ):
        assert i_fold in range(5), f"i_fold must be 0-4, got {i_fold}"
        assert fold_type in ["train", "val", "test"], \
            f"fold_type must be 'train', 'val', or 'test', got {fold_type}"

        self.process_name = process_name
        self.i_fold = i_fold
        self.fold_type = fold_type
        self.task = task
        self.weight_strategy = weight_strategy
        self.pt_log_scale = pt_log_scale

        # Load and merge Parquet files
        print(f"Loading Parquet files for process '{process_name}'...")
        df = load_and_merge_parquet(parquet_file_paths)
        print(f"  Loaded {len(df)} events")

        # Auto-assign EventNumber based on row index
        event_numbers = np.arange(len(df))

        # Fold splitting logic
        test_mask = (event_numbers - i_fold) % 5 == 0
        val_mask = (event_numbers - i_fold + 1) % 5 == 0
        train_mask = ~(test_mask | val_mask)

        if fold_type == "train":
            mask = train_mask
        elif fold_type == "val":
            mask = val_mask
        else:  # test
            mask = test_mask

        # Apply fold filter
        df = df[mask].reset_index(drop=True)
        self.event_numbers = event_numbers[mask]

        print(f"  Fold {i_fold} ({fold_type}): {len(df)} events")

        # Reconstruct events into shape (N, 5, 6)
        self.features = self.reconstruct_sequence(df, pt_log_scale)

        # Assign labels and weights
        label, process_weight = get_process_label_and_weight(process_name, task)
        self.labels = np.full(len(df), label, dtype=np.int64)
        sample_weight = compute_sample_weight(
            process_weight=process_weight,
            n_events=len(df),
            strategy=weight_strategy,
        )
        self.weights = np.full(len(df), sample_weight, dtype=np.float32)

    def reconstruct_sequence(self, df: pd.DataFrame, pt_log_scale: bool) -> np.ndarray:
        """Reconstruct event features from pandas DataFrame into (N, 5, 6) NumPy array."""
        N = len(df)
        features = np.zeros((N, 5, 6), dtype=np.float32)

        # Read base variables
        l1_pt = df["l1_pt"].values.astype(np.float32)
        l1_eta = df["l1_eta"].values.astype(np.float32)

        l2_pt = df["l2_pt"].values.astype(np.float32)
        l2_eta = df["l2_eta"].values.astype(np.float32)
        dphi_l2_l1 = df["dphi_l2_l1"].values.astype(np.float32)

        j1_pt = df["j1_pt"].values.astype(np.float32)
        j1_eta = df["j1_eta"].values.astype(np.float32)
        dphi_j1_l1 = df["dphi_j1_l1"].values.astype(np.float32)

        j2_pt = df["j2_pt"].values.astype(np.float32)
        j2_eta = df["j2_eta"].values.astype(np.float32)
        dphi_j2_l1 = df["dphi_j2_l1"].values.astype(np.float32)

        met_et = df["met_et"].values.astype(np.float32)
        dphi_met_l1 = df["dphi_met_l1"].values.astype(np.float32)

        if pt_log_scale:
            l1_pt = np.log(np.maximum(l1_pt, 1e-3))
            l2_pt = np.log(np.maximum(l2_pt, 1e-3))
            j1_pt = np.log(np.maximum(j1_pt, 1e-3))
            j2_pt = np.log(np.maximum(j2_pt, 1e-3))
            met_et = np.log(np.maximum(met_et, 1e-3))

        # Fill Row 0: Lepton 1 (Type: [1, 0, 0])
        features[:, 0, 0] = l1_pt
        features[:, 0, 1] = l1_eta
        features[:, 0, 2] = 0.0
        features[:, 0, 3] = 1.0

        # Fill Row 1: Lepton 2 (Type: [1, 0, 0])
        features[:, 1, 0] = l2_pt
        features[:, 1, 1] = l2_eta
        features[:, 1, 2] = dphi_l2_l1
        features[:, 1, 3] = 1.0

        # Fill Row 2: Jet 1 (Type: [0, 1, 0])
        features[:, 2, 0] = j1_pt
        features[:, 2, 1] = j1_eta
        features[:, 2, 2] = dphi_j1_l1
        features[:, 2, 4] = 1.0

        # Fill Row 3: Jet 2 (Type: [0, 1, 0])
        features[:, 3, 0] = j2_pt
        features[:, 3, 1] = j2_eta
        features[:, 3, 2] = dphi_j2_l1
        features[:, 3, 4] = 1.0

        # Fill Row 4: MET (Type: [0, 0, 1])
        features[:, 4, 0] = met_et
        features[:, 4, 1] = 0.0
        features[:, 4, 2] = dphi_met_l1
        features[:, 4, 5] = 1.0

        return features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, int]:
        """Returns:

            features: torch.Tensor of shape (5, 6)
            label: 0 or 1
            weight: process weight for this event
            event_number: row index
        """
        return (
            torch.from_numpy(self.features[idx]),
            self.labels[idx],
            self.weights[idx],
            self.event_numbers[idx],
        )


def create_fold_datasets(
    parquet_dir: str,
    i_fold: int = 0,
    task: str = "EW_vs_Background",
    weight_strategy: str = "hybrid",
    pt_log_scale: bool = True,
    balance_weights: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train, validation, and test datasets for a given fold."""
    # Get all Parquet files by process
    files_by_process = get_all_parquet_files(parquet_dir)

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for process_name, file_paths in sorted(files_by_process.items()):
        print(f"\nProcessing '{process_name}'...")
        if process_name not in TASK_DEFINITIONS[task]["signal_processes"] and \
           process_name not in TASK_DEFINITIONS[task]["background_processes"]:
            warnings.warn(
                f"Process '{process_name}' not in signal or background for task '{task}'; skipping"
            )
            continue

        train_ds = ParTFoldDataset(
            file_paths, process_name, i_fold, "train", task, weight_strategy, pt_log_scale
        )
        val_ds = ParTFoldDataset(
            file_paths, process_name, i_fold, "val", task, weight_strategy, pt_log_scale
        )
        test_ds = ParTFoldDataset(
            file_paths, process_name, i_fold, "test", task, weight_strategy, pt_log_scale
        )

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)

    # Rebalance binary classes globally
    if balance_weights:
        for datasets in [train_datasets, val_datasets, test_datasets]:
            if len(datasets) > 0:
                all_labels = np.concatenate([ds.labels for ds in datasets])
                all_weights = np.concatenate([ds.weights for ds in datasets])
                
                balanced_weights = balance_signal_background_weights(all_labels, all_weights)
                
                start_idx = 0
                for ds in datasets:
                    end_idx = start_idx + len(ds)
                    ds.weights = balanced_weights[start_idx:end_idx]
                    start_idx = end_idx

    from torch.utils.data import ConcatDataset
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), ConcatDataset(test_datasets)


def create_fold_loaders(
    parquet_dir: str,
    i_fold: int = 0,
    task: str = "EW_vs_Background",
    weight_strategy: str = "hybrid",
    pt_log_scale: bool = True,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    balance_weights: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test splits."""
    train_ds, val_ds, test_ds = create_fold_datasets(
        parquet_dir, i_fold, task, weight_strategy, pt_log_scale, balance_weights
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
