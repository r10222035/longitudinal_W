"""Data loading and pixelization module for WW polarization EventCNN training.

Handles Parquet loading, 3-channel EFlow event image pixelization, deterministic
5-fold cross-validation splitting, and PyTorch Dataset/DataLoader creation.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
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


_GLOBAL_IMAGE_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def pixelize_eflow_events(
    df: pd.DataFrame,
    res: int = 40,
    eta_range: Tuple[float, float] = (-5.0, 5.0),
    phi_range: Tuple[float, float] = (-np.pi, np.pi),
    normalize_per_event: bool = False,
) -> np.ndarray:
    """Convert EFlow constituent particles into 3-channel (C, H, W) event images.

    Channels:
      - Channel 0: EFlowTrack (part_type == 0, pt)
      - Channel 1: EFlowPhoton (part_type == 1, pt/et)
      - Channel 2: EFlowNeutralHadron (part_type == 2, pt/et)

    Args:
        df: Pandas DataFrame containing 'part_pt', 'part_eta', 'part_phi', 'part_type'
        res: Grid resolution (res x res)
        eta_range: Pseudorapidity boundaries (min, max)
        phi_range: Azimuthal angle boundaries (min, max)
        normalize_per_event: Whether to apply per-event (C, H, W) standardization

    Returns:
        Numpy array of shape (N, 3, res, res), dtype float32
    """
    N = len(df)
    images = np.zeros((N, 3, res, res), dtype=np.float32)

    if N == 0:
        return images

    raw_pts = df["part_pt"].values
    raw_etas = df["part_eta"].values
    raw_phis = df["part_phi"].values
    raw_types = df["part_type"].values

    bins_eta = np.linspace(eta_range[0], eta_range[1], res + 1)
    bins_phi = np.linspace(phi_range[0], phi_range[1], res + 1)

    for i in range(N):
        pts = raw_pts[i]
        n_p = len(pts)
        if n_p == 0:
            continue

        p_pt = np.asarray(pts, dtype=np.float32)
        p_eta = np.asarray(raw_etas[i], dtype=np.float32)
        p_phi = np.asarray(raw_phis[i], dtype=np.float32)
        p_type = np.asarray(raw_types[i], dtype=np.int32)

        # Clip and digitize into 0-indexed bins
        idx_eta = np.digitize(p_eta, bins_eta) - 1
        idx_phi = np.digitize(p_phi, bins_phi) - 1

        # Keep only particles within physical detector bounds
        valid_mask = (idx_eta >= 0) & (idx_eta < res) & (idx_phi >= 0) & (idx_phi < res)
        if not np.any(valid_mask):
            continue

        v_pt = p_pt[valid_mask]
        v_eta = idx_eta[valid_mask]
        v_phi = idx_phi[valid_mask]
        v_type = p_type[valid_mask]

        # Accumulate pT / ET for each of the 3 channels
        for ch in range(3):
            ch_mask = (v_type == ch)
            if np.any(ch_mask):
                np.add.at(
                    images[i, ch],
                    (v_eta[ch_mask], v_phi[ch_mask]),
                    v_pt[ch_mask],
                )

        if normalize_per_event:
            img = images[i]
            mean = np.mean(img, axis=(1, 2), keepdims=True)
            std = np.std(img, axis=(1, 2), keepdims=True)
            std = np.where(std < 1e-8, 1e-8, std)
            images[i] = (img - mean) / std

    return images


class CNNFoldDataset(Dataset):
    """PyTorch Dataset for event images with deterministic 5-fold cross validation."""

    def __init__(
        self,
        parquet_file_paths: List[str],
        process_name: str,
        i_fold: int = 0,
        fold_type: str = "train",
        task: str = "EW_vs_Background",
        weight_strategy: str = "hybrid",
        res: int = 40,
        normalize_per_event: bool = False,
        clean_duplicates: bool = True,
        dataset_fraction: float = 1.0,
        dataset_seed: int = 42,
    ):
        assert i_fold in range(5), f"i_fold must be 0-4, got {i_fold}"
        assert fold_type in ["train", "val", "test"], \
            f"fold_type must be 'train', 'val', or 'test', got {fold_type}"

        self.process_name = process_name
        self.i_fold = i_fold
        self.fold_type = fold_type
        self.task = task
        self.weight_strategy = weight_strategy
        self.res = res
        self.normalize_per_event = normalize_per_event
        self.clean_duplicates = clean_duplicates
        self.dataset_fraction = dataset_fraction
        self.dataset_seed = dataset_seed

        cache_key = (process_name, res, normalize_per_event, clean_duplicates, dataset_fraction, dataset_seed)

        if cache_key not in _GLOBAL_IMAGE_CACHE:
            print(f"Loading and pixelizing {process_name} (fraction={dataset_fraction})...")
            df = load_and_merge_parquet(parquet_file_paths)

            if clean_duplicates and "EventNumber" in df.columns:
                n_before = len(df)
                df = df.drop_duplicates(subset=["EventNumber"]).reset_index(drop=True)
                n_after = len(df)
                if n_before != n_after:
                    print(f"  Removed {n_before - n_after} duplicate events in {process_name}")

            # Subsample if dataset_fraction < 1.0
            if dataset_fraction < 1.0:
                n_total = len(df)
                n_sample = max(1, int(n_total * dataset_fraction))
                rng = np.random.RandomState(dataset_seed)
                sampled_indices = rng.choice(n_total, size=n_sample, replace=False)
                df = df.iloc[sampled_indices].reset_index(drop=True)
                print(f"  Subsampled {process_name} from {n_total} to {n_sample} events")

            if "EventNumber" in df.columns:
                event_numbers = df["EventNumber"].values
            else:
                event_numbers = np.arange(len(df))

            images = pixelize_eflow_events(
                df=df,
                res=res,
                normalize_per_event=normalize_per_event,
            )

            _GLOBAL_IMAGE_CACHE[cache_key] = (images, event_numbers)

        all_images, all_event_numbers = _GLOBAL_IMAGE_CACHE[cache_key]
        n_events = len(all_images)

        # 5-fold cross-validation masks
        test_mask = (all_event_numbers - i_fold) % 5 == 0
        val_mask = (all_event_numbers - i_fold + 1) % 5 == 0
        train_mask = ~(test_mask | val_mask)

        if fold_type == "train":
            mask = train_mask
        elif fold_type == "val":
            mask = val_mask
        else:
            mask = test_mask

        self.features = all_images[mask]
        self.event_numbers = all_event_numbers[mask]
        n_filtered = len(self.features)

        print(f"  Fold {i_fold} ({fold_type}): {n_filtered} events (from cached {n_events} events)")

        raw_label, process_weight = get_process_label_and_weight(process_name, task)
        self.labels = np.full(n_filtered, raw_label, dtype=np.int64)

        sample_weight = compute_sample_weight(
            process_weight=process_weight,
            n_events=n_filtered,
            strategy=weight_strategy,
        )
        self.weights = np.full(n_filtered, sample_weight, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
            torch.tensor(self.event_numbers[idx], dtype=torch.int64),
        )


def create_fold_loaders(
    parquet_dir: str,
    i_fold: int,
    task: str = "EW_vs_Background",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    weight_strategy: str = "hybrid",
    balance_weights: bool = True,
    res: int = 40,
    normalize_per_event: bool = False,
    clean_duplicates: bool = True,
    dataset_fraction: float = 1.0,
    dataset_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create train, validation, and test DataLoaders for a given fold."""
    files_by_process = get_all_parquet_files(parquet_dir)
    task_def = TASK_DEFINITIONS.get(task)
    if not task_def:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_DEFINITIONS.keys())}")

    relevant_processes = set(task_def["signal_processes"] + task_def["background_processes"])
    active_files = {p: f for p, f in files_by_process.items() if p in relevant_processes}

    if not active_files:
        raise ValueError(f"No parquet files found for task '{task}' in {parquet_dir}")

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for process_name, paths in active_files.items():
        for ftype, dlist in [("train", train_datasets), ("val", val_datasets), ("test", test_datasets)]:
            ds = CNNFoldDataset(
                parquet_file_paths=paths,
                process_name=process_name,
                i_fold=i_fold,
                fold_type=ftype,
                task=task,
                weight_strategy=weight_strategy,
                res=res,
                normalize_per_event=normalize_per_event,
                clean_duplicates=clean_duplicates,
                dataset_fraction=dataset_fraction,
                dataset_seed=dataset_seed,
            )
            dlist.append(ds)

    if balance_weights:
        for dlist in [train_datasets, val_datasets, test_datasets]:
            if len(dlist) > 0:
                all_labels = np.concatenate([ds.labels for ds in dlist])
                all_weights = np.concatenate([ds.weights for ds in dlist])
                eval_labels = all_labels[:, 0] if all_labels.ndim == 2 else all_labels
                balanced_weights = balance_signal_background_weights(eval_labels, all_weights)

                start_idx = 0
                for ds in dlist:
                    end_idx = start_idx + len(ds)
                    ds.weights = balanced_weights[start_idx:end_idx]
                    start_idx = end_idx

    train_concat = torch.utils.data.ConcatDataset(train_datasets)
    val_concat = torch.utils.data.ConcatDataset(val_datasets)
    test_concat = torch.utils.data.ConcatDataset(test_datasets)

    train_loader = DataLoader(
        train_concat,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_concat,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_concat,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    metadata = {
        "task": task,
        "i_fold": i_fold,
        "n_train": len(train_concat),
        "n_val": len(val_concat),
        "n_test": len(test_concat),
        "res": res,
    }

    return train_loader, val_loader, test_loader, metadata
