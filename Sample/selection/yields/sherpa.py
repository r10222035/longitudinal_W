"""Sherpa expected-yield utilities."""

from __future__ import annotations

import concurrent.futures
import os
import re

import numpy as np
import pyhepmc

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


def match_sherpa_hepmc_root_pairs(hepmc_dir, root_dir, max_pairs=None):
    """Match sample_seedN.hepmc with sample_seedN.root."""
    hepmc_files = sorted(
        [os.path.join(hepmc_dir, f) for f in os.listdir(hepmc_dir) if f.endswith(".hepmc")],
        key=lambda p: os.path.basename(p),
    )
    if max_pairs is not None:
        hepmc_files = hepmc_files[: int(max_pairs)]

    root_files = sorted(
        [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".root")],
        key=lambda p: os.path.basename(p),
    )
    root_map = {os.path.splitext(os.path.basename(p))[0]: p for p in root_files}

    pairs = []
    missing_root = []
    for hepmc_path in hepmc_files:
        stem = os.path.splitext(os.path.basename(hepmc_path))[0]
        root_path = root_map.get(stem)
        if root_path is None:
            missing_root.append(stem)
            continue
        pairs.append((hepmc_path, root_path))

    hepmc_stems = {os.path.splitext(os.path.basename(p))[0] for p in hepmc_files}
    extra_root = sorted([stem for stem in root_map.keys() if stem not in hepmc_stems])

    return {
        "pairs": pairs,
        "n_hepmc": len(hepmc_files),
        "n_root": len(root_files),
        "n_pairs": len(pairs),
        "missing_root": missing_root,
        "extra_root": extra_root,
    }


def _summarize_hepmc_xsec_file(hepmc_path, max_events_per_file=None):
    """Summarize event-level cross-sections from one HepMC file."""
    n_events_total = 0
    n_valid = 0
    n_missing = 0
    n_invalid = 0
    xsec_sum_pb = 0.0
    xsec_min_pb = None
    xsec_max_pb = None

    with pyhepmc.open(hepmc_path) as f_hepmc:
        for event in f_hepmc:
            if max_events_per_file is not None and n_events_total >= int(max_events_per_file):
                break

            n_events_total += 1
            if (not event) or (event.cross_section is None):
                n_missing += 1
                continue

            try:
                xsec_pb = float(event.cross_section.xsec(0))
            except Exception:
                n_invalid += 1
                continue

            if not np.isfinite(xsec_pb):
                n_invalid += 1
                continue

            n_valid += 1
            xsec_sum_pb += xsec_pb
            xsec_min_pb = xsec_pb if xsec_min_pb is None else min(xsec_min_pb, xsec_pb)
            xsec_max_pb = xsec_pb if xsec_max_pb is None else max(xsec_max_pb, xsec_pb)

    return {
        "n_events_total": n_events_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "n_invalid": n_invalid,
        "xsec_sum_pb": xsec_sum_pb,
        "xsec_min_pb": xsec_min_pb,
        "xsec_max_pb": xsec_max_pb,
    }


def get_cross_section_from_sherpa_hepmc_files(hepmc_paths, num_workers=None, max_events_per_file=None):
    """Read event-level cross-sections from HepMC files and return average in pb."""
    hepmc_paths = list(hepmc_paths)
    if len(hepmc_paths) == 0:
        raise RuntimeError("No HepMC files are provided for cross-section extraction.")

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) // 2)
    num_workers = max(1, min(int(num_workers), len(hepmc_paths)))

    file_summaries = []
    if num_workers == 1:
        for hepmc_path in hepmc_paths:
            file_summaries.append(
                _summarize_hepmc_xsec_file(
                    hepmc_path=hepmc_path,
                    max_events_per_file=max_events_per_file,
                )
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _summarize_hepmc_xsec_file,
                    hepmc_path,
                    max_events_per_file,
                ): hepmc_path
                for hepmc_path in hepmc_paths
            }
            for future in concurrent.futures.as_completed(futures):
                file_path = futures[future]
                try:
                    file_summaries.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"Failed to parse HepMC file: {file_path} ({exc})") from exc

    n_events_total = sum(item["n_events_total"] for item in file_summaries)
    n_valid = sum(item["n_valid"] for item in file_summaries)
    n_missing = sum(item["n_missing"] for item in file_summaries)
    n_invalid = sum(item["n_invalid"] for item in file_summaries)
    xsec_sum_pb = sum(item["xsec_sum_pb"] for item in file_summaries)

    valid_mins = [item["xsec_min_pb"] for item in file_summaries if item["xsec_min_pb"] is not None]
    valid_maxs = [item["xsec_max_pb"] for item in file_summaries if item["xsec_max_pb"] is not None]

    if n_valid == 0:
        raise RuntimeError("No valid cross_section found in the provided HepMC files.")

    mean_xsec_pb = xsec_sum_pb / n_valid
    return {
        "mean_xsec_pb": mean_xsec_pb,
        "min_xsec_pb": min(valid_mins) if valid_mins else None,
        "max_xsec_pb": max(valid_maxs) if valid_maxs else None,
        "n_events_total": n_events_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "n_invalid": n_invalid,
        "num_workers": num_workers,
    }


def _normalize_sherpa_pairs(hepmc_dir=None, root_dir=None, forced_pairs=None, max_pairs=None):
    """Return validated (hepmc_path, root_path) pairs from either dirs or explicit input."""
    if forced_pairs is not None:
        checked_pairs = []
        for idx, pair in enumerate(forced_pairs, 1):
            if len(pair) != 2:
                raise ValueError(f"Invalid pair at index {idx}: must be (hepmc_path, root_path).")

            hepmc_path, root_path = pair
            if not os.path.isfile(hepmc_path):
                raise FileNotFoundError(f"HepMC file not found: {hepmc_path}")
            if not os.path.isfile(root_path):
                raise FileNotFoundError(f"ROOT file not found: {root_path}")

            checked_pairs.append((hepmc_path, root_path))

        return {
            "pairs": checked_pairs,
            "n_hepmc": len(checked_pairs),
            "n_root": len(checked_pairs),
            "n_pairs": len(checked_pairs),
            "missing_root": [],
            "extra_root": [],
            "source": "forced_pairs",
        }

    if hepmc_dir is None or root_dir is None:
        raise ValueError("Either forced_pairs or both hepmc_dir/root_dir must be provided.")

    pairing = match_sherpa_hepmc_root_pairs(hepmc_dir=hepmc_dir, root_dir=root_dir, max_pairs=max_pairs)
    pairing["source"] = "matched_dirs"
    return pairing


def compute_expected_events_sherpa_by_region(
    hepmc_dir=None,
    root_dir=None,
    sample_name="EW_WWjj_Sherpa",
    luminosity_fb_inv=139.0,
    num_workers=None,
    chunks_per_file=1,
    max_pairs=None,
    xsec_num_workers=None,
    xsec_max_events_per_file=None,
    forced_pairs=None,
):
    """Compute expected yields for Sherpa using either matched dirs or explicit pairs."""
    pairing = _normalize_sherpa_pairs(
        hepmc_dir=hepmc_dir,
        root_dir=root_dir,
        forced_pairs=forced_pairs,
        max_pairs=max_pairs,
    )
    pairs = pairing["pairs"]
    if len(pairs) == 0:
        raise RuntimeError("No matched HepMC/ROOT pairs found.")

    hepmc_paths = [h for h, _ in pairs]
    root_paths = [r for _, r in pairs]

    xsec_stats = get_cross_section_from_sherpa_hepmc_files(
        hepmc_paths,
        num_workers=xsec_num_workers,
        max_events_per_file=xsec_max_events_per_file,
    )
    sigma_gen_fb = xsec_stats["mean_xsec_pb"] * 1000.0

    sample_groups = {sample_name: root_paths}

    print(f"[Sherpa] source={pairing['source']}")
    print(f"[Sherpa] matched pairs: {pairing['n_pairs']} / hepmc={pairing['n_hepmc']} (root total={pairing['n_root']})")
    print(f"[Sherpa] cross-section workers: {xsec_stats['num_workers']}")
    if pairing["missing_root"]:
        print(f"[Sherpa] missing root for stems: {pairing['missing_root'][:5]} ...")

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
        "pairing": pairing,
        "xsec_stats": xsec_stats,
        "n_total_events_in_root": n_total,
        "luminosity_fb_inv": luminosity_fb_inv,
    }

    return results, diagnostics


def compute_expected_events_sherpa_by_region_from_log(
    integration_log_path,
    hepmc_dir=None,
    root_dir=None,
    sample_name="EW_WWjj_Sherpa",
    luminosity_fb_inv=139.0,
    num_workers=None,
    chunks_per_file=1,
    max_pairs=None,
    forced_pairs=None,
):
    """Compute expected yields for Sherpa using integration-log cross-sections.

    ROOT event selection still comes from matched dirs or explicit pairs, while
    generated cross-section is extracted from Sherpa integration log summaries.
    """
    pairing = _normalize_sherpa_pairs(
        hepmc_dir=hepmc_dir,
        root_dir=root_dir,
        forced_pairs=forced_pairs,
        max_pairs=max_pairs,
    )
    pairs = pairing["pairs"]
    if len(pairs) == 0:
        raise RuntimeError("No matched HepMC/ROOT pairs found.")

    root_paths = [r for _, r in pairs]

    xsec_stats = get_cross_section_from_sherpa_integration_log(integration_log_path)
    sigma_gen_fb = xsec_stats["mean_xsec_pb"] * 1000.0

    sample_groups = {sample_name: root_paths}

    print(f"[Sherpa] source={pairing['source']}")
    print(f"[Sherpa] matched pairs: {pairing['n_pairs']} / hepmc={pairing['n_hepmc']} (root total={pairing['n_root']})")
    print(f"[Sherpa] cross-section source: integration_log ({xsec_stats['n_processes']} process summaries)")
    print(f"[Sherpa] decay BR applied: {xsec_stats['apply_decay_branching_ratios']}")
    print(f"[Sherpa] raw xsec [pb]: {xsec_stats['raw_mean_xsec_pb']:.6e}")
    print(f"[Sherpa] effective xsec [pb]: {xsec_stats['mean_xsec_pb']:.6e}")
    print(f"[Sherpa] integration log: {integration_log_path}")
    if pairing["missing_root"]:
        print(f"[Sherpa] missing root for stems: {pairing['missing_root'][:5]} ...")

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
        "pairing": pairing,
        "xsec_stats": xsec_stats,
        "n_total_events_in_root": n_total,
        "luminosity_fb_inv": luminosity_fb_inv,
    }

    return results, diagnostics