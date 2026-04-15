"""ROOT and Delphes initialization utilities."""

from __future__ import annotations

import os
import sys


def _resolve_prefix():
    """Resolve active environment prefix for ROOT/Delphes assets."""
    return os.environ.get("CONDA_PREFIX") or sys.prefix


def prepare_cppyy_api_path():
    """Set CPPYY_API_PATH early so importing ROOT won't emit CPyCppyy warnings."""
    prefix = _resolve_prefix()
    if not prefix:
        return

    candidate = f"{prefix}/include/CPyCppyy"
    if os.path.isdir(candidate):
        os.environ["CPPYY_API_PATH"] = candidate


def initialize_root_delphes():
    """Initialize ROOT and load Delphes library with proper C++ dictionary support.
    
    This function ensures that:
    1. CPyCppyy API path is correctly set (required for C++ object serialization)
    2. ROOT module is imported
    3. libDelphes.so is loaded (provides run-time type information for Delphes objects)
    
    Call this function once per process before accessing ROOT objects, especially
    when reading Delphes trees from ROOT files. Safe to call multiple times.
    
    Raises:
        RuntimeError: If ROOT/Delphes initialization fails.
    """
    # Determine prefix: use CONDA_PREFIX if set, otherwise fall back to sys.prefix
    # (important for subprocess contexts where CONDA_PREFIX may not be inherited)
    prefix = _resolve_prefix()
    
    if not prefix:
        raise RuntimeError(
            "Cannot determine environment prefix. "
            "Ensure conda environment is activated."
        )
    
    # Set CPyCppyy API path before importing ROOT.
    prepare_cppyy_api_path()
    
    # Import ROOT module
    import ROOT
    
    # Load Delphes shared library to provide runtime type information
    delphes_lib = f"{prefix}/lib/libDelphes.so"
    if not os.path.isfile(delphes_lib):
        raise RuntimeError(
            f"libDelphes.so not found at {delphes_lib}. "
            "Ensure ROOT and Delphes are properly installed in the environment."
        )
    
    ROOT.gSystem.Load(delphes_lib)
