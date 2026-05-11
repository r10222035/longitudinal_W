"""
Data loading module for WW polarization DNN training.

Handles Parquet file loading, deterministic 5-fold cross-validation,
label/weight assignment, and PyTorch Dataset/DataLoader creation.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Dict, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from config import get_process_label_and_weight


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

class FeatureNormalizer:
    """
    Feature normalization layer that supports optional pre-scaling.

    Supports three `scale_fn` forms:
      - None / "none": identity
      - "log_pt": apply log(abs(x) + eps) to all features
      - callable: function(X) -> X_scaled applied to whole matrix
      - dict: mapping feature_name -> callable(column) for per-feature preproc

    After pre-scaling, a `StandardScaler` is fit per-column (z-score).
    """

    def __init__(
        self,
        scale_fn: Optional[Union[str, Callable, Dict[str, Callable]]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.scale_fn = scale_fn
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.fitted = False
        self.n_features = None

    def _apply_scale_fn(self, X: np.ndarray) -> np.ndarray:
        """Apply pre-scaling function if specified.

        For dict `scale_fn`, requires `feature_names` to map names -> column index.
        """
        if self.scale_fn is None or self.scale_fn == "none":
            return X

        if isinstance(self.scale_fn, dict):
            if self.feature_names is None:
                raise ValueError("feature_names required when using dict scale_fn")
            Xs = X.copy()
            for fname, fn in self.scale_fn.items():
                if fname not in self.feature_names:
                    warnings.warn(f"scale_fn dict contains unknown feature '{fname}'; ignoring")
                    continue
                idx = self.feature_names.index(fname)
                col = Xs[:, idx]
                if not callable(fn):
                    raise ValueError(f"scale_fn for feature '{fname}' must be callable")
                res = fn(col)
                Xs[:, idx] = np.asarray(res)
            return Xs

        if self.scale_fn == "log_pt":
            return np.log(np.abs(X) + 1e-6)

        if callable(self.scale_fn):
            return self.scale_fn(X)

        raise ValueError(f"Unknown scale_fn: {self.scale_fn}")

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        X_scaled = self._apply_scale_fn(X)
        self.scaler.fit(X_scaled)
        self.n_features = X.shape[1]
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        X_scaled = self._apply_scale_fn(X)
        return self.scaler.transform(X_scaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    @property
    def mean(self) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted.")
        return self.scaler.mean_

    @property
    def scale(self) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted.")
        return self.scaler.scale_


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
    - Normalizes features (fitted only on training set)
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
        scale_fn: Optional[str] = None,
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
        self.scale_fn = scale_fn
        
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
        
        # Assign labels and weights
        label, process_weight = get_process_label_and_weight(process_name, task)
        self.labels = np.full(len(df), label, dtype=np.int64)
        self.weights = np.full(len(df), process_weight, dtype=np.float32)
        
        # Initialize normalizer (will be fitted externally for train set)
        self.normalizer = FeatureNormalizer(scale_fn=scale_fn, feature_names=self.FEATURE_COLUMNS)
        self._normalized = False
    
    def fit_normalizer(self):
        """Fit feature normalizer on this dataset."""
        self.normalizer.fit(self.features)
        self._normalized = True
    
    def apply_normalizer(self):
        """Apply normalization to features."""
        if not self.normalizer.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit_normalizer() on train set first.")
        self.features = self.normalizer.transform(self.features).astype(np.float32)
    
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
    scale_fn: Optional[str] = None,
) -> Tuple[ParquetFoldDataset, ParquetFoldDataset, ParquetFoldDataset]:
    """
    Create train, validation, and test datasets for a given fold.
    
    Important: Normalizer is fitted on train set only, then applied to all splits.
    
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
    
    # Create datasets for each split (train, val, test)
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for process_name, file_paths in sorted(files_by_process.items()):
        print(f"\nProcessing '{process_name}'...")
        
        # Create train, val, test datasets for this process
        train_ds = ParquetFoldDataset(
            file_paths, process_name, i_fold, "train", task, scale_fn
        )
        val_ds = ParquetFoldDataset(
            file_paths, process_name, i_fold, "val", task, scale_fn
        )
        test_ds = ParquetFoldDataset(
            file_paths, process_name, i_fold, "test", task, scale_fn
        )
        
        # Fit normalizer on train set
        train_ds.fit_normalizer()
        
        # Apply same normalizer to all splits
        train_ds.apply_normalizer()
        val_ds.normalizer = train_ds.normalizer
        val_ds.apply_normalizer()
        test_ds.normalizer = train_ds.normalizer
        test_ds.apply_normalizer()
        
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)
    
    # Concatenate datasets from all processes
    from torch.utils.data import ConcatDataset
    
    train_concat = ConcatDataset(train_datasets)
    val_concat = ConcatDataset(val_datasets)
    test_concat = ConcatDataset(test_datasets)
    
    return train_concat, val_concat, test_concat


def create_fold_loaders(
    parquet_dir: str,
    i_fold: int = 0,
    task: str = "EW_vs_Background",
    scale_fn: Optional[str] = None,
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
        parquet_dir, i_fold, task, scale_fn
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
