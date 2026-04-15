"""MG5 expected-yield utilities."""

from __future__ import annotations

import re
from pathlib import Path

from selection.core.parallel import selection_cut_parallel


def _get_cross_section_from_mg5_banner(banner_path):
    """Return generated cross-section in fb from an MG5 banner file."""
    banner_path = Path(banner_path)
    with open(banner_path, "r") as f:
        text = f.read()

    match = re.search(r"#\s+Integrated weight \(pb\)\s*:\s*([0-9.eE+-]+)", text)
    if not match:
        raise ValueError(f"Cannot find Integrated weight (pb) in {banner_path}")

    return float(match.group(1)) * 1000.0


def compute_expected_events_mg_by_region(
    sample_inputs,
    luminosity_fb_inv=139.0,
    num_workers=None,
    chunks_per_file=1,
):
    """Parallel expected-yield analysis using the shared selection core."""
    from selection import _ensure_root_initialized
    _ensure_root_initialized()
    
    results = {}
    sample_groups = {sample_name: [paths["root"]] for sample_name, paths in sample_inputs.items()}

    print("[Yields] Processing all samples with file + chunk parallelism...\n")
    sr_results_parallel = selection_cut_parallel(
        sample_groups=sample_groups,
        cut_func_name="pass_SR_cuts",
        num_workers=num_workers,
        chunks_per_file=chunks_per_file,
    )
    low_mjj_results_parallel = selection_cut_parallel(
        sample_groups=sample_groups,
        cut_func_name="pass_low_mjj_cr_cuts",
        num_workers=num_workers,
        chunks_per_file=chunks_per_file,
    )
    wz_results_parallel = selection_cut_parallel(
        sample_groups=sample_groups,
        cut_func_name="pass_WZ_CR_cuts",
        num_workers=num_workers,
        chunks_per_file=chunks_per_file,
    )

    for sample_name, paths in sample_inputs.items():
        sigma_gen_fb = _get_cross_section_from_mg5_banner(paths["banner"])
        sr = sr_results_parallel[sample_name]
        low_mjj = low_mjj_results_parallel[sample_name]
        wz = wz_results_parallel[sample_name]

        n_total = sr["cutflow"]["Total"]
        n_sr_passed = sr["cutflow"]["Jet cut"]
        acc_sr = n_sr_passed / n_total if n_total > 0 else 0.0
        acc_low_mjj = low_mjj["cutflow"]["Jet cut"] / n_total if n_total > 0 else 0.0
        acc_wz = wz["cutflow"]["Jet cut"] / n_total if n_total > 0 else 0.0

        sigma_sr_fb = sigma_gen_fb * acc_sr
        sigma_low_mjj_fb = sigma_gen_fb * acc_low_mjj
        sigma_wz_fb = sigma_gen_fb * acc_wz

        results[sample_name] = {
            "sigma_gen_fb": sigma_gen_fb,
            "sigma_sr_fb": sigma_sr_fb,
            "sigma_low_mjj_fb": sigma_low_mjj_fb,
            "sigma_wz_fb": sigma_wz_fb,
            "n_total_events": n_total,
            "n_passed_events": n_sr_passed,
            "n_expected_sr": sigma_sr_fb * luminosity_fb_inv,
            "n_expected_low_mjj": sigma_low_mjj_fb * luminosity_fb_inv,
            "n_expected_wz": sigma_wz_fb * luminosity_fb_inv,
        }

    return results