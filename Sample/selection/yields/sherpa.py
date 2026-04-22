"""Sherpa expected-yield utilities."""

from __future__ import annotations

import os
import re

import numpy as np

from selection.core.parallel import selection_cut_parallel


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m|\[[0-9;]*m")
_SHERPA_PROCESS_XSEC_RE = re.compile(
    r"^\s*(?P<process>[^:]+?)\s*:\s*"
    r"(?P<xsec>[0-9.eE+-]+)\s*(?P<xsec_unit>pb|fb)\s*"
    r"\+\-\s*\(\s*(?P<err>[0-9.eE+-]+)\s*(?P<err_unit>pb|fb)\s*=\s*"
    r"(?P<rel_err>[0-9.eE+-]+)\s*%\s*\)"
    r"(?:\s*exp\.\s*eff:\s*(?P<exp_eff>[0-9.eE+-]+)\s*%)?\s*$"
)

_BR_W_TO_E = 0.108201
_BR_W_TO_MU = 0.108201
_BR_Z_TO_EE = 0.0336633
_BR_Z_TO_MUMU = 0.0336633
_BR_W_LEPTONIC = _BR_W_TO_E + _BR_W_TO_MU
_BR_Z_LEPTONIC = _BR_Z_TO_EE + _BR_Z_TO_MUMU


def _strip_ansi_codes(text):
    """Strip ANSI escape sequences and Sherpa color markers from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


def _to_pb(value, unit):
    """Convert cross-section value from pb/fb to pb."""
    if unit == "pb":
        return value
    if unit == "fb":
        return value / 1000.0
    raise ValueError(f"Unsupported cross-section unit: {unit}")


def _infer_sherpa_decay_branching_ratio(process_name):
    """Infer the leptonic branching-ratio factor from a Sherpa process name."""
    bosons = re.findall(r"W[+-]|Z", process_name)
    n_w = sum(1 for boson in bosons if boson.startswith("W"))
    n_z = sum(1 for boson in bosons if boson == "Z")
    return (_BR_W_LEPTONIC ** n_w) * (_BR_Z_LEPTONIC ** n_z)


def get_cross_section_from_sherpa_integration_log(
    integration_log_path,
    apply_decay_branching_ratios=True,
):
    """Read Sherpa integration log and return total cross-section in pb.

    The final summary lines are matched by the pattern:
    <process> : <xsec> <unit> +- ( <err> <unit> = <rel_err> % ) [exp. eff: ...]
    If the same process appears multiple times, the last occurrence is used.
    When apply_decay_branching_ratios is true, multiply each process cross-section
    by the inferred leptonic branching fraction for its on-shell W/Z content.
    """
    if not os.path.isfile(integration_log_path):
        raise FileNotFoundError(f"Sherpa integration log not found: {integration_log_path}")

    process_summary = {}
    with open(integration_log_path, "r", encoding="utf-8", errors="replace") as f_log:
        for raw_line in f_log:
            clean_line = _strip_ansi_codes(raw_line.rstrip("\n"))
            match = _SHERPA_PROCESS_XSEC_RE.match(clean_line)
            if not match:
                continue

            process = match.group("process").strip()
            xsec_value = float(match.group("xsec"))
            xsec_unit = match.group("xsec_unit")
            err_value = float(match.group("err"))
            err_unit = match.group("err_unit")
            rel_err_percent = float(match.group("rel_err"))
            exp_eff_text = match.group("exp_eff")
            exp_eff_percent = float(exp_eff_text) if exp_eff_text is not None else None
            decay_branching_ratio = _infer_sherpa_decay_branching_ratio(process)

            raw_xsec_pb = _to_pb(xsec_value, xsec_unit)
            raw_err_pb = _to_pb(err_value, err_unit)
            if apply_decay_branching_ratios:
                xsec_pb = raw_xsec_pb * decay_branching_ratio
                err_pb = raw_err_pb * decay_branching_ratio
            else:
                xsec_pb = raw_xsec_pb
                err_pb = raw_err_pb

            if (not np.isfinite(xsec_pb)) or (not np.isfinite(err_pb)):
                continue

            process_summary[process] = {
                "raw_xsec_pb": raw_xsec_pb,
                "raw_abs_err_pb": raw_err_pb,
                "xsec_pb": xsec_pb,
                "abs_err_pb": err_pb,
                "rel_err_percent": rel_err_percent,
                "exp_eff_percent": exp_eff_percent,
                "decay_branching_ratio": decay_branching_ratio,
            }

    if not process_summary:
        raise RuntimeError(
            "No valid process cross-section summary line found in "
            f"Sherpa integration log: {integration_log_path}"
        )

    process_items = [
        {
            "process": process,
            "raw_xsec_pb": info["raw_xsec_pb"],
            "raw_abs_err_pb": info["raw_abs_err_pb"],
            "xsec_pb": info["xsec_pb"],
            "abs_err_pb": info["abs_err_pb"],
            "rel_err_percent": info["rel_err_percent"],
            "exp_eff_percent": info["exp_eff_percent"],
            "decay_branching_ratio": info["decay_branching_ratio"],
        }
        for process, info in process_summary.items()
    ]
    xsecs_pb = [item["xsec_pb"] for item in process_items]
    abs_errs_pb = [item["abs_err_pb"] for item in process_items]
    raw_xsecs_pb = [item["raw_xsec_pb"] for item in process_items]
    raw_abs_errs_pb = [item["raw_abs_err_pb"] for item in process_items]

    total_xsec_pb = float(np.sum(xsecs_pb))
    total_abs_err_pb = float(np.sqrt(np.sum(np.square(abs_errs_pb))))
    total_rel_err_percent = (total_abs_err_pb / total_xsec_pb * 100.0) if total_xsec_pb > 0 else np.nan
    raw_total_xsec_pb = float(np.sum(raw_xsecs_pb))
    raw_total_abs_err_pb = float(np.sqrt(np.sum(np.square(raw_abs_errs_pb))))

    return {
        "mean_xsec_pb": total_xsec_pb,
        "min_xsec_pb": float(np.min(xsecs_pb)),
        "max_xsec_pb": float(np.max(xsecs_pb)),
        "raw_mean_xsec_pb": raw_total_xsec_pb,
        "raw_min_xsec_pb": float(np.min(raw_xsecs_pb)),
        "raw_max_xsec_pb": float(np.max(raw_xsecs_pb)),
        "raw_total_abs_err_pb": raw_total_abs_err_pb,
        "n_events_total": 0,
        "n_valid": len(process_items),
        "n_missing": 0,
        "n_invalid": 0,
        "num_workers": 1,
        "source": "integration_log",
        "apply_decay_branching_ratios": apply_decay_branching_ratios,
        "integration_log_path": integration_log_path,
        "n_processes": len(process_items),
        "total_abs_err_pb": total_abs_err_pb,
        "total_rel_err_percent": total_rel_err_percent,
        "process_breakdown": process_items,
    }


def _normalize_sherpa_root_inputs(
    root_dir=None,
    root_paths=None,
    max_root_files=None,
):
    """Return validated ROOT input list from explicit paths or directory scan."""
    if root_paths is not None:
        source = "root_paths"
        candidates = list(root_paths)
        n_total = len(candidates)
    else:
        if root_dir is None:
            raise ValueError("Provide one of root_paths or root_dir.")
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"ROOT directory not found: {root_dir}")

        source = "root_dir"
        candidates = sorted(
            [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".root")],
            key=lambda p: os.path.basename(p),
        )
        n_total = len(candidates)

    if max_root_files is not None:
        candidates = candidates[: int(max_root_files)]

    checked_root_paths = []
    missing_roots = []
    for root_path in candidates:
        if not os.path.isfile(root_path):
            missing_roots.append(root_path)
            continue
        checked_root_paths.append(root_path)

    return {
        "source": source,
        "root_paths": checked_root_paths,
        "n_root_total": n_total,
        "n_root_selected": len(candidates),
        "n_root_valid": len(checked_root_paths),
        "missing_root_paths": missing_roots,
    }


def compute_expected_events_sherpa_by_region(
    integration_log_path,
    root_dir=None,
    root_paths=None,
    sample_name="EW_WWjj_Sherpa",
    luminosity_fb_inv=139.0,
    num_workers=None,
    chunks_per_file=1,
    max_root_files=None,
    apply_decay_branching_ratios=True,
):
    """Compute expected yields for Sherpa using integration-log cross-sections only.

    Event selection is run on ROOT files provided by root_paths or root_dir.
    Cross-sections are always extracted from integration log.
    """
    root_input_summary = _normalize_sherpa_root_inputs(
        root_dir=root_dir,
        root_paths=root_paths,
        max_root_files=max_root_files,
    )
    selected_root_paths = root_input_summary["root_paths"]
    if len(selected_root_paths) == 0:
        raise RuntimeError("No valid ROOT files found for Sherpa selection.")

    xsec_stats = get_cross_section_from_sherpa_integration_log(
        integration_log_path,
        apply_decay_branching_ratios=apply_decay_branching_ratios,
    )
    sigma_gen_fb = xsec_stats["mean_xsec_pb"] * 1000.0

    sample_groups = {sample_name: selected_root_paths}

    print(f"[Sherpa] root source: {root_input_summary['source']}")
    print(
        f"[Sherpa] root files (valid/selected/total): "
        f"{root_input_summary['n_root_valid']}/{root_input_summary['n_root_selected']}/{root_input_summary['n_root_total']}"
    )
    print(f"[Sherpa] cross-section source: integration_log ({xsec_stats['n_processes']} process summaries)")
    print(f"[Sherpa] decay BR applied: {xsec_stats['apply_decay_branching_ratios']}")
    print(f"[Sherpa] raw xsec [pb]: {xsec_stats['raw_mean_xsec_pb']:.6e}")
    print(f"[Sherpa] effective xsec [pb]: {xsec_stats['mean_xsec_pb']:.6e}")
    print(f"[Sherpa] integration log: {integration_log_path}")
    if root_input_summary["missing_root_paths"]:
        print(f"[Sherpa] missing root paths (first 5): {root_input_summary['missing_root_paths'][:5]}")

    sr_results = selection_cut_parallel(
        sample_groups=sample_groups,
        cut_func_name="pass_SR_cuts",
        num_workers=num_workers,
        chunks_per_file=chunks_per_file,
    )
    low_mjj_results = selection_cut_parallel(
        sample_groups=sample_groups,
        cut_func_name="pass_low_mjj_cr_cuts",
        num_workers=num_workers,
        chunks_per_file=chunks_per_file,
    )
    wz_results = selection_cut_parallel(
        sample_groups=sample_groups,
        cut_func_name="pass_WZ_CR_cuts",
        num_workers=num_workers,
        chunks_per_file=chunks_per_file,
    )

    sr = sr_results[sample_name]
    low_mjj = low_mjj_results[sample_name]
    wz = wz_results[sample_name]

    n_total = sr["cutflow"]["Total"]
    n_sr_passed = sr["cutflow"]["Jet cut"]
    acc_sr = n_sr_passed / n_total if n_total > 0 else 0.0
    acc_low_mjj = low_mjj["cutflow"]["Jet cut"] / n_total if n_total > 0 else 0.0
    acc_wz = wz["cutflow"]["Jet cut"] / n_total if n_total > 0 else 0.0

    sigma_sr_fb = sigma_gen_fb * acc_sr
    sigma_low_mjj_fb = sigma_gen_fb * acc_low_mjj
    sigma_wz_fb = sigma_gen_fb * acc_wz

    results = {
        sample_name: {
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
    }

    diagnostics = {
        "root_input_summary": root_input_summary,
        "xsec_stats": xsec_stats,
        "n_total_events_in_root": n_total,
        "luminosity_fb_inv": luminosity_fb_inv,
    }

    return results, diagnostics
