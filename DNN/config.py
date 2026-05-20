"""
Configuration module for WW polarization state DNN training.

Defines task configurations, process-to-label mappings, and hyperparameters.
"""

from typing import Dict, List, Any
import math

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

PROCESS_WEIGHTS = {
    # EW WW polarization states (signal for EW_vs_Background task)
    "WWjj_EW_LL_WW_cmf": 18.29,
    "WWjj_EW_LT_WW_cmf": 58.88,
    "WWjj_EW_TT_WW_cmf": 124.50,
    "WWjj_EW_LL_pp_cmf": 11.49,
    "WWjj_EW_LT_pp_cmf": 67.84,
    "WWjj_EW_TT_pp_cmf": 123.07,
    "WWjj_EW_MGH7": 206.52,
    
    # Simplified EW process names (for wwzz_mix batch)
    "WWjj_EW": 206.52,  # Sum of all EW WWjj states
    
    # Backgrounds
    "WWjj_QCD": 24.05,
    "WWjj_Int": 7.57,
    "WZjj_EW": 14.95,
    "WZjj_QCD": 28.50,
    "top": 5.02,
    "ZZ": 2.51,
    "charge_flip": 10.10,
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
    if task not in TASK_DEFINITIONS:
        raise ValueError(f"Task '{task}' not found. Available: {list(TASK_DEFINITIONS.keys())}")
    
    if process_name not in PROCESS_WEIGHTS:
        raise ValueError(f"Process '{process_name}' not found in PROCESS_WEIGHTS")
    
    task_def = TASK_DEFINITIONS[task]
    weight = PROCESS_WEIGHTS[process_name]
    
    if process_name in task_def["signal_processes"]:
        label = task_def["signal_label"]
    elif process_name in task_def["background_processes"]:
        label = task_def["background_label"]
    else:
        raise ValueError(
            f"Process '{process_name}' not in signal or background for task '{task}'"
        )
    
    return label, weight


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


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class TrainingConfig:
    """Hyperparameter configuration for DNN training."""
    
    def __init__(self):
        # Data
        self.batch_size: int = 256
        self.num_workers: int = 4
        self.pin_memory: bool = True
        
        # Task selection
        self.task: str = "EW_vs_Background"  # Classification task
        
        # Model architecture
        self.n_features: int = 32  # Physics features from Parquet
        self.hidden_width: int = 128
        self.n_hidden_layers: int = 4
        self.dropout_rate: float = 0.3
        
        # Initialization (Ref. [96])
        self.init_var_scale: float = 2.952  # for truncated normal weights
        self.init_bias_std: float = 0.2  # for bias normal distribution
        
        # Optimizer
        self.learning_rate: float = 0.001
        self.adam_eps: float = 1e-8
        self.weight_decay: float = 0.0
        
        # Training
        self.max_epochs: int = 200
        self.early_stopping_patience: int = 20
        self.early_stopping_threshold: float = 1e-4  # min improvement
        
        # Data paths
        self.parquet_dir: str = "Sample/Parquet/batch_sr_parquet_ew_polar"
        self.output_dir: str = "./DNN/results"
        
        # Random seed for reproducibility
        self.seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self) -> str:
        return f"TrainingConfig({self.to_dict()})"


# Default configuration instance
DEFAULT_CONFIG = TrainingConfig()
