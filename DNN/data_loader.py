"""
Data loading module for WW polarization DNN training.

Handles Parquet file loading, deterministic 5-fold cross-validation,
label/weight assignment, and PyTorch Dataset/DataLoader creation.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Dict
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import get_process_label_and_weight, compute_sample_weight
from config import TASK_DEFINITIONS

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_process_name_from_filename(filename: str) -> str:
    """
    Extract process name from Parquet filename.
    
    Examples:
        "WWjj_EW-LL-WW_cmf_run_01_sr.parquet" -> "WWjj_EW_LL_WW_cmf"
        "WWjj_QCD_run_01_sr.parquet" -> "WWjj_QCD"
    
    Args:
        filename: Parquet filename (without directory)
    
    Returns:
        Normalized process name
    """
    # Remove extension
    name = filename.replace(".parquet", "")
    # Remove run info and _sr suffix
    name = name.split("_run_")[0]
    # Replace hyphens with underscores for consistency
    name = name.replace("-", "_")
    return name


def get_all_parquet_files(parquet_dir: str) -> Dict[str, List[str]]:
    """
    Scan all Parquet files in batch directories.
    
    Args:
        parquet_dir: Path to parent batch directory (e.g., "Sample/Parquet")
    
    Returns:
        Dict mapping process_name -> list of file paths
    """
    files_by_process = {}
    parquet_path = Path(parquet_dir)
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    
    # Scan all subdirectories for .parquet files
    for parquet_file in parquet_path.rglob("*.parquet"):
        process_name = parse_process_name_from_filename(parquet_file.name)
        if process_name not in files_by_process:
            files_by_process[process_name] = []
        files_by_process[process_name].append(str(parquet_file))
    
    if not files_by_process:
        raise ValueError(f"No Parquet files found in {parquet_dir}")
    
    return files_by_process


def load_and_merge_parquet(file_paths: List[str]) -> pd.DataFrame:
    """
    Load and merge multiple Parquet files.
    
    Args:
        file_paths: List of Parquet file paths to merge
    
    Returns:
        Merged DataFrame
    """
    dfs = []
    for fpath in file_paths:
        df = pd.read_parquet(fpath)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# NORMALIZATION LAYER
# ============================================================================

class PreScaler:
    """
    Deterministic per-feature pre-scaler.

    This class applies element-wise, deterministic transformations to
    selected features (e.g. np.sqrt, np.log1p). It does NOT implement
    any fitting or statistics-based normalization.

    `scale_fn` should be either None (identity) or a dict mapping
    feature_name -> callable(column_array) -> column_array.
    """

    def __init__(
        self,
        scale_fn: Optional[Dict[str, Callable]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        if scale_fn is not None and not isinstance(scale_fn, dict):
            raise TypeError("scale_fn must be a dict mapping feature_name->callable or None")
        self.scale_fn = scale_fn
        self.feature_names = feature_names

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Apply configured per-feature pre-scaling to a 2D numpy array.

        X is expected to be shape (n_samples, n_features). When `scale_fn`
        is None this returns X unchanged. If a feature in `scale_fn` is not
        present in `feature_names` it will be ignored with a warning.
        """
        if self.scale_fn is None:
            return X

        if self.feature_names is None:
            raise ValueError("feature_names required when using scale_fn dict")

        Xs = X.copy()
        for fname, fn in self.scale_fn.items():
            if fname not in self.feature_names:
                warnings.warn(f"scale_fn dict contains unknown feature '{fname}'; ignoring")
                continue
            idx = self.feature_names.index(fname)
            col = Xs[:, idx]
            if not callable(fn):
                raise ValueError(f"scale_fn for feature '{fname}' must be callable")
            # apply callable to the whole column
            res = fn(col)
            Xs[:, idx] = np.asarray(res)
        return Xs


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class ParquetFoldDataset(Dataset):
    """
    PyTorch Dataset for Parquet files with deterministic 5-fold splitting.
    
    Features:
    - Loads features from Parquet (32 physics variables)
    - Auto-assigns EventNumber based on row index
    - Deterministic fold splitting: test=(EventNumber-i_fold)%5==0,
      val=(EventNumber-i_fold+1)%5==0, train=rest
    - Assigns label and weight based on process name and task definition
    - Applies deterministic pre-scaling to features (no fitting; e.g. sqrt/log)
    """
    
    FEATURE_COLUMNS = [
        # Leptons
        "l1_pt", "l1_eta", "l1_flavor_code",
        "l2_pt", "l2_eta", "l2_flavor_code",
        # Jets
        "j1_pt", "j1_eta", "j2_pt", "j2_eta",
        # MET
        "met_et", "met_phi",
        # Angular
        "dphi_l2_l1", "dphi_j1_l1", "dphi_j2_l1", "dphi_met_l1", 
        "dphi_jj", "dr_ll", "dr_jj",
        # Kinematics
        "m_ll", "m_jj", "pt_ll", "deta_ll", "dy_jj",
        # TransvMass
        "mt_l1_met", "mt_l2_met", "mt_ll_met", "mt0_ll_met",
        # Physics
        "zstar_l1", "zstar_l2", "ptprod_ll_over_jj",
        # Geometry
        "min_dr_lj",
    ]
    
    def __init__(
        self,
        parquet_file_paths: List[str],
        process_name: str,
        i_fold: int = 0,
        fold_type: str = "train",
        task: str = "EW_vs_Background",
        weight_strategy: str = "process",
        pre_scaler: Optional[PreScaler] = None,
    ):
        """
        Args:
            parquet_file_paths: List of Parquet files for this process
            process_name: Physics process name (e.g., "WWjj_EW_LL_WW_cmf")
            i_fold: Fold index (0-4)
            fold_type: "train", "val", or "test"
            task: Task definition ("EW_vs_Background", etc.)
            scale_fn: Optional pre-scaling function for features
        """
        assert i_fold in range(5), f"i_fold must be 0-4, got {i_fold}"
        assert fold_type in ["train", "val", "test"], \
            f"fold_type must be 'train', 'val', or 'test', got {fold_type}"
        
        self.process_name = process_name
        self.i_fold = i_fold
        self.fold_type = fold_type
        self.task = task
        self.weight_strategy = weight_strategy
        self.pre_scaler = pre_scaler
        
        # Load and merge Parquet files
        print(f"Loading Parquet files for process '{process_name}'...")
        df = load_and_merge_parquet(parquet_file_paths)
        print(f"  Loaded {len(df)} events")
        
        # Auto-assign EventNumber based on row index
        event_numbers = np.arange(len(df))
        
        # Fold assignment logic
        # test: (EventNumber - i_fold) % 5 == 0
        # val:  (EventNumber - i_fold + 1) % 5 == 0
        # train: rest
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
        
        # Extract features
        self.features = df[self.FEATURE_COLUMNS].values.astype(np.float32)

        # Apply deterministic pre-scaling (e.g. sqrt/log) if requested.
        if self.pre_scaler is not None:
            self.features = self.pre_scaler.apply(self.features)
        
        # Assign labels and weights
        label, process_weight = get_process_label_and_weight(process_name, task)
        self.labels = np.full(len(df), label, dtype=np.int64)
        sample_weight = compute_sample_weight(
            process_weight=process_weight,
            n_events=len(df),
            strategy=weight_strategy,
        )
        self.weights = np.full(len(df), sample_weight, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, int]:
        """
        Returns:
            features: torch.Tensor of shape (32,)
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


# ============================================================================
# DATA LOADER FACTORY
# ============================================================================

def create_fold_datasets(
    parquet_dir: str,
    i_fold: int = 0,
    task: str = "EW_vs_Background",
    weight_strategy: str = "process",
    scale_fn: Optional[Dict[str, Callable]] = None,
) -> Tuple[ParquetFoldDataset, ParquetFoldDataset, ParquetFoldDataset]:
    """
    Create train, validation, and test datasets for a given fold.
    
    Important: Only deterministic pre-scaling is applied (e.g. sqrt/log). No fitting
    or statistics-based normalization is performed by this factory.
    
    Args:
        parquet_dir: Path to parent batch directory
        i_fold: Fold index (0-4)
        task: Task definition
        scale_fn: Optional pre-scaling function
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Get all Parquet files by process
    files_by_process = get_all_parquet_files(parquet_dir)
    # Build a PreScaler from the provided scale_fn (deterministic element-wise)
    pre_scaler = PreScaler(scale_fn, feature_names=ParquetFoldDataset.FEATURE_COLUMNS) if scale_fn is not None else None
    
    # Create datasets for each split (train, val, test)
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
        
        # Create train, val, test datasets for this process (apply same pre_scaler)
        train_ds = ParquetFoldDataset(
            file_paths, process_name, i_fold, "train", task, weight_strategy, pre_scaler
        )
        val_ds = ParquetFoldDataset(
            file_paths, process_name, i_fold, "val", task, weight_strategy, pre_scaler
        )
        test_ds = ParquetFoldDataset(
            file_paths, process_name, i_fold, "test", task, weight_strategy, pre_scaler
        )
        
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)

    # Concatenate datasets from all processes
    from torch.utils.data import ConcatDataset

    train_concat = ConcatDataset(train_datasets)
    val_concat = ConcatDataset(val_datasets)
    test_concat = ConcatDataset(test_datasets)
    
    # Return concatenated datasets (features already preprocessed)
    return train_concat, val_concat, test_concat


def create_fold_loaders(
    parquet_dir: str,
    i_fold: int = 0,
    task: str = "EW_vs_Background",
    weight_strategy: str = "process",
    scale_fn: Optional[Dict[str, Callable]] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        parquet_dir: Path to batch directory
        i_fold: Fold index (0-4)
        task: Task definition
        scale_fn: Optional pre-scaling function
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = create_fold_datasets(
        parquet_dir, i_fold, task, weight_strategy, scale_fn
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
