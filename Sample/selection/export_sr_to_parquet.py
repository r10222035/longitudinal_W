#!/usr/bin/env python3
"""Export SR-passed Delphes events to a flat Parquet table for DNN training.

This script reuses the canonical reconstructed-level SR definition from
`selection.core.cuts.pass_SR_cuts` and computes both basic and derived physics
features required by the DNN input specification.
"""

from __future__ import annotations
import concurrent.futures

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure `selection` package import works whether invoked from repo root or Sample.
_THIS_FILE = Path(__file__).resolve()
_SAMPLE_DIR = _THIS_FILE.parent.parent
if str(_SAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_SAMPLE_DIR))

from selection import _ensure_root_initialized
from selection.core.cuts import CUT_STAGE, pass_SR_cuts

_ensure_root_initialized()
import ROOT

EPS = 1.0e-9
FLAVOR_TO_PDG = {"e": 11, "mu": 13}


def _abs_delta_phi(phi1: float, phi2: float) -> float:
    """Return absolute wrapped delta-phi in [0, pi]."""
    dphi = phi1 - phi2
    return abs(math.atan2(math.sin(dphi), math.cos(dphi)))


def _standard_mt(obj_p4, met_et: float, met_phi: float) -> float:
    """Standard transverse mass using ET(obj)=sqrt(m^2+pt^2)."""
    et_x_sq = obj_p4.M() * obj_p4.M() + obj_p4.Pt() * obj_p4.Pt()
    et_x = math.sqrt(max(et_x_sq, 0.0))
    met_px = met_et * math.cos(met_phi)
    met_py = met_et * math.sin(met_phi)

    px_tot = obj_p4.Px() + met_px
    py_tot = obj_p4.Py() + met_py

    mt_sq = (et_x + met_et) * (et_x + met_et) - (px_tot * px_tot + py_tot * py_tot)
    return math.sqrt(max(mt_sq, 0.0))


def _projected_mt0(l1_p4, l2_p4, met_et: float, met_phi: float) -> float:
    """Projected massless transverse mass mT0 for ll+MET system."""
    met_px = met_et * math.cos(met_phi)
    met_py = met_et * math.sin(met_phi)

    scalar_sum_pt = l1_p4.Pt() + l2_p4.Pt() + met_et
    px_tot = l1_p4.Px() + l2_p4.Px() + met_px
    py_tot = l1_p4.Py() + l2_p4.Py() + met_py

    mt0_sq = scalar_sum_pt * scalar_sum_pt - (px_tot * px_tot + py_tot * py_tot)
    return math.sqrt(max(mt0_sq, 0.0))


def _zstar(eta_l: float, eta_j1: float, eta_j2: float) -> float:
    denom = eta_j1 - eta_j2
    if abs(denom) < EPS:
        return float("nan")
    center = 0.5 * (eta_j1 + eta_j2)
    return abs((eta_l - center) / denom)


def _init_feature_store() -> dict[str, list]:
    return {
        # Basic lepton features
        "l1_pt": [],
        "l1_eta": [],
        "l1_flavor": [],
        "l1_flavor_code": [],
        "l2_pt": [],
        "l2_eta": [],
        "l2_flavor": [],
        "l2_flavor_code": [],
        # Basic jet features
        "j1_pt": [],
        "j1_eta": [],
        "j2_pt": [],
        "j2_eta": [],
        # Missing transverse momentum
        "met_et": [],
        "met_phi": [],
        # Delta-phi with leading lepton
        "dphi_l2_l1": [],
        "dphi_j1_l1": [],
        "dphi_j2_l1": [],
        "dphi_met_l1": [],
        # Zeppenfeld variables
        "zstar_l1": [],
        "zstar_l2": [],
        # Standard transverse masses
        "mt_l1_met": [],
        "mt_l2_met": [],
        "mt_ll_met": [],
        # Projected massless transverse mass
        "mt0_ll_met": [],
        # Di-lepton observables
        "dr_ll": [],
        "deta_ll": [],
        "m_ll": [],
        "pt_ll": [],
        # Di-jet observables
        "dr_jj": [],
        "dy_jj": [],
        "m_jj": [],
        "dphi_jj": [],
        # Scalar pT product ratio
        "ptprod_ll_over_jj": [],
        # Closest lepton-jet distance
        "min_dr_lj": [],
    }


def _iter_entry_ranges(start_index: int, end_index: int, chunk_size: int):
    start_index = max(0, int(start_index))
    end_index = max(start_index, int(end_index))
    chunk_size = max(1, int(chunk_size))

    for chunk_start in range(start_index, end_index, chunk_size):
        chunk_end = min(end_index, chunk_start + chunk_size)
        yield chunk_start, chunk_end


def _extract_features(
    tree,
    max_events: int | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
    start_index: int = 0,
    end_index: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    n_entries = int(tree.GetEntries())
    i_start = max(0, int(start_index))
    i_stop = n_entries if end_index is None else min(n_entries, int(end_index))
    if max_events is not None:
        i_stop = min(i_stop, i_start + max_events)
    n_loop = max(0, i_stop - i_start)

    features = _init_feature_store()
    cutflow = {
        CUT_STAGE["total"]: 0,
        CUT_STAGE["lepton"]: 0,
        CUT_STAGE["met"]: 0,
        CUT_STAGE["jet"]: 0,
    }
    diagnostics = {
        "zstar_nan_count": 0,
        "pt_ratio_nan_count": 0,
    }

    event_iter = tqdm(
        range(n_loop),
        desc=progress_desc or "Event loop",
        unit="evt",
        disable=not show_progress,
    )

    for i in event_iter:
        tree.GetEntry(i_start + i)
        cutflow[CUT_STAGE["total"]] += 1

        stage, objs = pass_SR_cuts(tree, return_objects=True)
        if stage >= 1:
            cutflow[CUT_STAGE["lepton"]] += 1
        if stage >= 2:
            cutflow[CUT_STAGE["met"]] += 1
        if stage >= 3:
            cutflow[CUT_STAGE["jet"]] += 1
        if stage < 3:
            continue

        leptons = objs["leptons"]
        ll_p4 = objs["ll_p4"]
        jets = objs["jets"]
        met = objs["met"]

        l1 = leptons[0]
        l2 = leptons[1]
        l1_p4 = l1["p4"]
        l2_p4 = l2["p4"]

        j1 = jets[0]
        j2 = jets[1]

        met_et = float(met.MET)
        met_phi = float(met.Phi)

        l1_flavor = l1["flavor"]
        l2_flavor = l2["flavor"]

        eta_j1 = float(j1.Eta())
        eta_j2 = float(j2.Eta())

        zstar_l1 = _zstar(float(l1_p4.Eta()), eta_j1, eta_j2)
        zstar_l2 = _zstar(float(l2_p4.Eta()), eta_j1, eta_j2)
        if math.isnan(zstar_l1) or math.isnan(zstar_l2):
            diagnostics["zstar_nan_count"] += 1

        mt_l1 = _standard_mt(l1_p4, met_et, met_phi)
        mt_l2 = _standard_mt(l2_p4, met_et, met_phi)
        mt_ll = _standard_mt(ll_p4, met_et, met_phi)
        mt0_ll = _projected_mt0(l1_p4, l2_p4, met_et, met_phi)

        ptprod_jj = j1.Pt() * j2.Pt()
        if abs(ptprod_jj) < EPS:
            pt_ratio = float("nan")
            diagnostics["pt_ratio_nan_count"] += 1
        else:
            pt_ratio = (l1_p4.Pt() * l2_p4.Pt()) / ptprod_jj

        min_dr_lj = min(
            l1_p4.DeltaR(j1),
            l1_p4.DeltaR(j2),
            l2_p4.DeltaR(j1),
            l2_p4.DeltaR(j2),
        )

        features["l1_pt"].append(float(l1_p4.Pt()))
        features["l1_eta"].append(float(l1_p4.Eta()))
        features["l1_flavor"].append(l1_flavor)
        # dict.get(..., 0): keep unknown flavor as 0 instead of raising KeyError.
        features["l1_flavor_code"].append(FLAVOR_TO_PDG.get(l1_flavor, 0))
        features["l2_pt"].append(float(l2_p4.Pt()))
        features["l2_eta"].append(float(l2_p4.Eta()))
        features["l2_flavor"].append(l2_flavor)
        features["l2_flavor_code"].append(FLAVOR_TO_PDG.get(l2_flavor, 0))

        features["j1_pt"].append(float(j1.Pt()))
        features["j1_eta"].append(eta_j1)
        features["j2_pt"].append(float(j2.Pt()))
        features["j2_eta"].append(eta_j2)

        features["met_et"].append(met_et)
        features["met_phi"].append(met_phi)

        features["dphi_l2_l1"].append(abs(float(l1_p4.DeltaPhi(l2_p4))))
        features["dphi_j1_l1"].append(abs(float(j1.DeltaPhi(l1_p4))))
        features["dphi_j2_l1"].append(abs(float(j2.DeltaPhi(l1_p4))))
        features["dphi_met_l1"].append(_abs_delta_phi(met_phi, float(l1_p4.Phi())))

        features["zstar_l1"].append(float(zstar_l1))
        features["zstar_l2"].append(float(zstar_l2))

        features["mt_l1_met"].append(float(mt_l1))
        features["mt_l2_met"].append(float(mt_l2))
        features["mt_ll_met"].append(float(mt_ll))
        features["mt0_ll_met"].append(float(mt0_ll))

        features["dr_ll"].append(float(l1_p4.DeltaR(l2_p4)))
        features["deta_ll"].append(abs(float(l1_p4.Eta() - l2_p4.Eta())))
        features["m_ll"].append(float(ll_p4.M()))
        features["pt_ll"].append(float(ll_p4.Pt()))

        features["dr_jj"].append(float(j1.DeltaR(j2)))
        features["dy_jj"].append(abs(float(j1.Rapidity() - j2.Rapidity())))
        features["m_jj"].append(float((j1 + j2).M()))
        features["dphi_jj"].append(abs(float(j1.DeltaPhi(j2))))

        features["ptprod_ll_over_jj"].append(float(pt_ratio))
        features["min_dr_lj"].append(float(min_dr_lj))

    out_arrays: dict[str, np.ndarray] = {}
    for key, values in features.items():
        if key in {"l1_flavor", "l2_flavor"}:
            out_arrays[key] = np.asarray(values, dtype=object)
        elif key in {"l1_flavor_code", "l2_flavor_code"}:
            out_arrays[key] = np.asarray(values, dtype=np.int16)
        else:
            out_arrays[key] = np.asarray(values, dtype=np.float32)

    return out_arrays, cutflow, diagnostics


def _features_to_table(features: dict[str, np.ndarray]) -> pa.Table:
    table_dict: dict[str, pa.Array] = {}
    for key, arr in features.items():
        if key in {"l1_flavor", "l2_flavor"}:
            table_dict[key] = pa.array(arr.tolist(), type=pa.string())
        elif key in {"l1_flavor_code", "l2_flavor_code"}:
            table_dict[key] = pa.array(arr.tolist(), type=pa.int16())
        else:
            table_dict[key] = pa.array(arr, type=pa.float32())

    return pa.Table.from_pydict(table_dict)


def _write_parquet(features: dict[str, np.ndarray], output_path: Path, compression: str, row_group_size: int):
    table = _features_to_table(features)
    pq.write_table(table, str(output_path), compression=compression, row_group_size=row_group_size)


def _chunk_worker(task: tuple[str, str, int, int]) -> dict:
    input_root, tree_name, start_index, end_index = task

    from selection.core.root_init import initialize_root_delphes

    initialize_root_delphes()
    import ROOT

    f = ROOT.TFile.Open(str(input_root))
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {input_root}")

    tree = f.Get(tree_name)
    if tree is None:
        f.Close()
        raise RuntimeError(f"Cannot find TTree '{tree_name}' in {input_root}")

    features, cutflow, diagnostics = _extract_features(
        tree,
        max_events=None,
        show_progress=False,
        progress_desc=None,
        start_index=start_index,
        end_index=end_index,
    )
    f.Close()

    return {
        "start_index": int(start_index),
        "end_index": int(end_index),
        "features": features,
        "cutflow": cutflow,
        "diagnostics": diagnostics,
    }


def _export_parquet_parallel(
    input_path: Path,
    tree_name: str,
    output_path: Path,
    compression: str,
    row_group_size: int,
    max_events: int | None,
    workers: int,
    chunk_size: int,
    show_progress: bool,
) -> tuple[dict[str, int], dict[str, int], int]:
    f = ROOT.TFile.Open(str(input_path))
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {input_path}")

    tree = f.Get(tree_name)
    if tree is None:
        f.Close()
        raise RuntimeError(f"Cannot find TTree '{tree_name}' in {input_path}")

    n_entries = int(tree.GetEntries())
    i_stop = n_entries if max_events is None else min(n_entries, max_events)
    f.Close()

    if i_stop <= 0:
        raise RuntimeError(f"No entries to export in {input_path}")

    tasks = list(_iter_entry_ranges(0, i_stop, chunk_size))
    if not tasks:
        raise RuntimeError(f"Failed to create chunk tasks for {input_path}")

    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    cutflow = {
        CUT_STAGE["total"]: 0,
        CUT_STAGE["lepton"]: 0,
        CUT_STAGE["met"]: 0,
        CUT_STAGE["jet"]: 0,
    }
    diagnostics = {
        "zstar_nan_count": 0,
        "pt_ratio_nan_count": 0,
    }
    n_passed = 0
    parquet_writer = None

    ctx = mp.get_context("spawn")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            future_to_task = {
                executor.submit(_chunk_worker, (str(input_path), tree_name, start, end)): (start, end)
                for start, end in tasks
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_task),
                total=len(future_to_task),
                desc=f"chunks: {input_path.name}",
                unit="chunk",
                disable=not show_progress,
            ):
                start, end = future_to_task[future]
                result = future.result()
                features = result["features"]
                chunk_cutflow = result["cutflow"]
                chunk_diagnostics = result["diagnostics"]

                for key in cutflow:
                    cutflow[key] += int(chunk_cutflow[key])
                for key in diagnostics:
                    diagnostics[key] += int(chunk_diagnostics[key])

                chunk_passed = int(chunk_cutflow[CUT_STAGE["jet"]])
                n_passed += chunk_passed
                _validate_features(features, n_passed=chunk_passed)
                table = _features_to_table(features)

                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(
                        str(partial_path),
                        table.schema,
                        compression=compression,
                    )

                parquet_writer.write_table(table, row_group_size=row_group_size)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if partial_path.exists():
        partial_path.replace(output_path)
    else:
        raise RuntimeError(f"Parallel export did not produce parquet output: {partial_path}")

    return cutflow, diagnostics, n_passed


def _validate_features(features: dict[str, np.ndarray], n_passed: int):
    # Fail fast for data integrity: required columns exist and row counts agree.
    required_columns = {
        "l1_pt", "l1_eta", "l1_flavor", "l1_flavor_code", "l2_pt", "l2_eta", "l2_flavor", "l2_flavor_code",
        "j1_pt", "j1_eta", "j2_pt", "j2_eta",
        "met_et", "met_phi",
        "dphi_l2_l1", "dphi_j1_l1", "dphi_j2_l1", "dphi_met_l1",
        "zstar_l1", "zstar_l2",
        "mt_l1_met", "mt_l2_met", "mt_ll_met", "mt0_ll_met",
        "dr_ll", "deta_ll", "m_ll", "pt_ll",
        "dr_jj", "dy_jj", "m_jj", "dphi_jj",
        "ptprod_ll_over_jj", "min_dr_lj",
    }

    missing = sorted(required_columns - set(features.keys()))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    for key, arr in features.items():
        if arr.shape[0] != n_passed:
            raise ValueError(
                f"Column length mismatch for '{key}': {arr.shape[0]} != {n_passed}"
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SR-passed Delphes events into a flat DNN-ready Parquet file.",
    )
    parser.add_argument("--input-root", required=True, help="Path to input ROOT file.")
    parser.add_argument("--tree", default="Delphes", help="TTree name (default: Delphes).")
    parser.add_argument("--output-parquet", required=True, help="Path to output Parquet file.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional max number of events to scan.")
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=50000,
        help="Parquet row-group size for write_table.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker processes for chunked export. 0 means auto.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Entry count per chunk when using parallel export. 0 means auto.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable event-level progress bar.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    # Resolve user paths once so behavior is independent of current working directory.
    input_path = Path(args.input_root).resolve()
    output_path = Path(args.output_parquet).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f = ROOT.TFile.Open(str(input_path))
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {input_path}")

    tree = f.Get(args.tree)
    if tree is None:
        f.Close()
        raise RuntimeError(f"Cannot find TTree '{args.tree}' in {input_path}")

    progress_desc = f"events: {input_path.name}"
    n_entries = int(tree.GetEntries())

    compression = None if args.compression == "none" else args.compression
    workers = args.workers

    effective_entries = n_entries if args.max_events is None else min(n_entries, args.max_events)
    chunk_size = args.chunk_size
    if chunk_size <= 0:
        if workers <= 1 or effective_entries <= 0:
            chunk_size = max(1, effective_entries)
        else:
            chunk_size = max(1, math.ceil(effective_entries / (workers * 4)))
    chunk_size = max(1, int(chunk_size))

    use_parallel = workers > 1 and effective_entries > chunk_size
    if use_parallel:
        f.Close()
        cutflow, diagnostics, n_passed = _export_parquet_parallel(
            input_path=input_path,
            tree_name=args.tree,
            output_path=output_path,
            compression=compression,
            row_group_size=args.row_group_size,
            max_events=args.max_events,
            workers=workers,
            chunk_size=chunk_size,
            show_progress=not args.no_progress,
        )
        feature_columns = sorted(_init_feature_store().keys())
    else:
        try:
            features, cutflow, diagnostics = _extract_features(
                tree,
                max_events=args.max_events,
                show_progress=not args.no_progress,
                progress_desc=progress_desc,
            )
            n_passed = int(cutflow[CUT_STAGE["jet"]])
            _validate_features(features, n_passed=n_passed)
            _write_parquet(
                features,
                output_path=output_path,
                compression=compression,
                row_group_size=args.row_group_size,
            )
            feature_columns = sorted(features.keys())
        finally:
            f.Close()

    # Sidecar metadata supports reproducibility/debugging without opening Parquet.
    sidecar = {
        "input_root": str(input_path),
        "tree": args.tree,
        "output_parquet": str(output_path),
        "max_events": args.max_events,
        "parallel": {
            "enabled": use_parallel,
            "workers": workers,
            "chunk_size": chunk_size,
            "effective_entries": effective_entries,
        },
        "cutflow": cutflow,
        "diagnostics": diagnostics,
        "n_output_rows": n_passed,
        "columns": feature_columns,
    }
    sidecar_path = output_path.with_suffix(output_path.suffix + ".cutflow.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"[done] wrote parquet: {output_path}")
    print(f"[done] wrote cutflow: {sidecar_path}")
    print(f"[summary] total={cutflow[CUT_STAGE['total']]} pass_SR={n_passed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
