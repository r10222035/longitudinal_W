"""Selection package for reconstructed-level analysis."""

from __future__ import annotations

# ===== Lazy ROOT initialization =====
_root_initialized = False
_root_init_error = None


def _ensure_root_initialized():
    """Ensure ROOT and Delphes are initialized. Safe to call multiple times."""
    global _root_initialized, _root_init_error
    
    if _root_initialized:
        return
    
    if _root_init_error is not None:
        raise RuntimeError(f"ROOT initialization previously failed: {_root_init_error}")
    
    try:
        from .core.root_init import initialize_root_delphes
        initialize_root_delphes()
        _root_initialized = True
    except Exception as e:
        _root_init_error = str(e)
        raise RuntimeError(
            f"Failed to initialize ROOT and Delphes: {e}\n"
            f"Ensure conda environment is activated with ROOT and Delphes installed."
        ) from e


# ===== Public API imports =====
from .core.cuts import (
    CUT_STAGE,
    pass_SR_cuts,
    pass_low_mjj_cr_cuts,
    pass_WZ_CR_cuts,
)
from .core.parallel import selection_cut_parallel, selection_cut_with_region
from .yields.mg5 import compute_expected_events_mg_by_region
from .yields.reporting import print_expected_event_counts_table
from .yields.sherpa import compute_expected_events_sherpa_by_region

__all__ = [
    "CUT_STAGE",
    "pass_SR_cuts",
    "pass_low_mjj_cr_cuts",
    "pass_WZ_CR_cuts",
    "selection_cut_parallel",
    "selection_cut_with_region",
    "compute_expected_events_mg_by_region",
    "print_expected_event_counts_table",
    "compute_expected_events_sherpa_by_region",
    "_ensure_root_initialized",
]
