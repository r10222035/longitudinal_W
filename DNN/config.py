"""
Configuration module for WW polarization state DNN training.

Defines task configurations, process-to-label mappings, and hyperparameters.
"""

from pathlib import Path
from typing import Dict, List, Any
import math

import numpy as np
import yaml

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

PROCESS_WEIGHTS = {
    # EW WW polarization states (signal for EW_vs_Background task)
    "WWjj_EW_LL_WW_cmf": 0.138912,
    "WWjj_EW_LT_WW_cmf": 0.440544,
    "WWjj_EW_TT_WW_cmf": 0.877367,

    # Simplified EW process names (for wwzz_mix batch)
    "WWjj_EW": 1.436625,  # Sum of all EW WWjj states
    
    # Backgrounds
    "WWjj_QCD": 0.110415,
    "WZjj_EW": 0.200249,
    "WZjj_QCD": 0.443649,
}

WEIGHT_STRATEGIES = {
    "process",
    "inverse_event_count",
    "hybrid",
}

# ============================================================================
# PROCESS ALIASES
# ============================================================================

PROCESS_ALIASES = {
    # Map simplified names (from wwzz_mix batch) to detailed names
    "WWjj_EW": "WWjj_EW_LL_WW_cmf",  # Maps simplified to first detailed variant
}


def resolve_process_name(process_name: str) -> str:
    """
    Resolve a process name using aliases if necessary.
    
    Args:
        process_name: Original process name
    
    Returns:
        Resolved process name
    """
    return PROCESS_ALIASES.get(process_name, process_name)


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

TASK_DEFINITIONS = {
    "EW_vs_Background": {
        "name": "EW WWjj vs All Backgrounds",
        "type": "binary",
        "signal_processes": [
            "WWjj_EW",  # Simplified name for wwzz_mix batch
        ],
        "background_processes": [
            "WWjj_QCD",
            "WWjj_Int",
            "WZjj_EW",
            "WZjj_QCD",
            "top",
            "ZZ",
            "charge_flip",
        ],
        "signal_label": 1,
        "background_label": 0,
    },
    "PolState_LL_vs_LT_TT": {
        "name": "WW Polarization: LL vs (LT+TT)",
        "type": "binary",
        "signal_processes": [
            "WWjj_EW_LL_WW_cmf",
        ],
        "background_processes": [
            "WWjj_EW_LT_WW_cmf",
            "WWjj_EW_TT_WW_cmf",
        ],
        "signal_label": 1,
        "background_label": 0,
    },
    "PolState_LL_LT_vs_TT": {
        "name": "WW Polarization: (LL+LT) vs TT",
        "type": "binary",
        "signal_processes": [
            "WWjj_EW_LL_WW_cmf",
            "WWjj_EW_LT_WW_cmf",
        ],
        "background_processes": [
            "WWjj_EW_TT_WW_cmf",
        ],
        "signal_label": 1,
        "background_label": 0,
    }, 
}


def get_process_label_and_weight(
    process_name: str, task: str = "EW_vs_Background"
) -> tuple:
    """
    Map a process name to its label and weight for a given task.
    
    Args:
        process_name: Name of the physics process (e.g., "WWjj_EW_LL_WW_cmf")
        task: Task definition to use (default: "EW_vs_Background")
    
    Returns:
        (label, weight) tuple where label is 0 or 1, weight is physics cross-section (fb)
    
    Raises:
        ValueError: If process_name not found in PROCESS_WEIGHTS or task not found
    """
    orig_name = process_name
    process_name = resolve_process_name(process_name)

    if task not in TASK_DEFINITIONS:
        raise ValueError(f"Task '{task}' not found. Available: {list(TASK_DEFINITIONS.keys())}")
    
    if process_name not in PROCESS_WEIGHTS:
        raise ValueError(f"Process '{process_name}' not found in PROCESS_WEIGHTS")
    
    task_def = TASK_DEFINITIONS[task]
    weight = PROCESS_WEIGHTS[process_name]
    
    if orig_name in task_def["signal_processes"] or process_name in task_def["signal_processes"]:
        label = task_def["signal_label"]
    elif orig_name in task_def["background_processes"] or process_name in task_def["background_processes"]:
        label = task_def["background_label"]
    else:
        raise ValueError(
            f"Process '{process_name}' (original: '{orig_name}') not in signal or background for task '{task}'"
        )
    
    return label, weight


def compute_sample_weight(
    process_weight: float,
    n_events: int,
    strategy: str = "process",
) -> float:
    """
    Compute the final sample weight for a process under a given strategy.

    Args:
        process_weight: Physics cross-section weight for the process
        n_events: Number of events in the current split for that process
        strategy: One of 'process', 'inverse_event_count', or 'hybrid'

    Returns:
        Final per-event sample weight
    """
    if strategy not in WEIGHT_STRATEGIES:
        raise ValueError(
            f"Unknown weight strategy '{strategy}'. Available: {sorted(WEIGHT_STRATEGIES)}"
        )

    if strategy == "process":
        return float(process_weight)

    if n_events <= 0:
        return 0.0

    inverse_event_count = 1.0 / float(n_events)

    if strategy == "inverse_event_count":
        return inverse_event_count

    return float(process_weight) * inverse_event_count


def balance_signal_background_weights(
    labels: np.ndarray,
    weights: np.ndarray,
    signal_label: int = 1,
    background_label: int = 0,
) -> np.ndarray:
    """
    Rebalance binary-class sample weights so signal and background have equal total weight.

    The combined total weight is preserved, while the signal and background totals are
    both set to half of the original combined total. Relative weights within each class
    are preserved.
    """
    labels = np.asarray(labels)
    balanced_weights = np.asarray(weights, dtype=np.float64).copy()

    signal_mask = labels == signal_label
    background_mask = labels == background_label

    if not signal_mask.any() or not background_mask.any():
        return balanced_weights.astype(np.float32)

    signal_total = float(balanced_weights[signal_mask].sum())
    background_total = float(balanced_weights[background_mask].sum())

    if signal_total <= 0.0 or background_total <= 0.0:
        return balanced_weights.astype(np.float32)

    target_total = 0.5 * (signal_total + background_total)
    balanced_weights[signal_mask] *= target_total / signal_total
    balanced_weights[background_mask] *= target_total / background_total

    return balanced_weights.astype(np.float32)


# ============================================================================
# FEATURE PRE-SCALING CONFIGURATION
# ============================================================================

# Deterministic per-feature pre-scaling (e.g. sqrt, log).
# Dict maps feature_name -> callable(column_array) -> scaled_column_array.
# Set to None for identity (no scaling).
# Modify this dict to experiment with different feature transformations.
SCALE_FN = {
    'l1_pt': np.log,
    'l1_eta': lambda x: x,
    'l1_flavor_code': lambda x: x,
    'l2_pt': np.log,
    'l2_eta': lambda x: x,
    'dphi_l2_l1': lambda x: x,
    'l2_flavor_code': lambda x: x,
    'j1_pt': np.log,
    'j1_eta': lambda x: x,
    'dphi_j1_l1': lambda x: x,
    'j2_pt': np.log,
    'j2_eta': lambda x: x,
    'dphi_j2_l1': lambda x: x,
    'met_et': np.log,
    # 'met_phi': lambda x: x,
    'dphi_met_l1': lambda x: x,
    'zstar_l1': np.sqrt,
    'zstar_l2': np.sqrt,
    'mt_l1_met': np.sqrt,
    'mt_l2_met': np.sqrt,
    'dr_ll': lambda x: x,
    'deta_ll': np.sqrt,
    'm_ll': np.log,
    'pt_ll': np.sqrt,
    'mt_ll_met': np.sqrt,
    'mt0_ll_met': np.sqrt,
    'dr_jj': lambda x: x,
    'dy_jj': lambda x: x,
    'm_jj': np.log,
    'dphi_jj': lambda x: x,
    'ptprod_ll_over_jj': lambda x: np.log(x + 0.02),
    'min_dr_lj': lambda x: x,
}


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "default_config.yaml"


def _load_yaml_mapping(config_path: Path) -> Dict[str, Any]:
    """Load a YAML mapping from disk."""
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return loaded


def load_default_config_data() -> Dict[str, Any]:
    """Load the canonical default config data from YAML when available."""
    if DEFAULT_CONFIG_PATH.exists():
        return _load_yaml_mapping(DEFAULT_CONFIG_PATH)
    else:
        raise FileNotFoundError(f"Default config not found at {DEFAULT_CONFIG_PATH}")


def load_training_config(
    config_path: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> "TrainingConfig":
    """Load a training config from YAML and optional runtime overrides."""
    if config_path is None:
        config_data = load_default_config_data()
    else:
        config_data = _load_yaml_mapping(Path(config_path))

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config_data[key] = value

    return TrainingConfig(**config_data)


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class TrainingConfig:
    """Hyperparameter configuration for DNN training."""

    def __init__(self, **kwargs: Any):
        defaults = load_default_config_data()

        weight_strategy = kwargs.get(
            "weight_strategy", defaults.get("weight_strategy", "process")
        )
        if weight_strategy not in WEIGHT_STRATEGIES:
            raise ValueError(
                f"Unknown weight_strategy '{weight_strategy}'. Available: {sorted(WEIGHT_STRATEGIES)}"
            )

        for key, value in defaults.items():
            setattr(self, key, kwargs.get(key, value))

        self.weight_strategy = weight_strategy

        unknown_keys = set(kwargs) - set(defaults)
        if unknown_keys:
            raise ValueError(f"Unknown TrainingConfig fields: {sorted(unknown_keys)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Construct a config from a dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"TrainingConfig({self.to_dict()})"


# Default configuration instance
DEFAULT_CONFIG = TrainingConfig()
