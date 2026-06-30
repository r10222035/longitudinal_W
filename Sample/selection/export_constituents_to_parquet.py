#!/usr/bin/env python3
"""Export SR-passed Delphes events to a Parquet table with Track & Tower constituents.

This script reuses the canonical reconstructed-level SR definition from
`selection.core.cuts.pass_SR_cuts` and extracts all constituent features from
`Track` and `Tower` branches for downstream network training (CNN, Particle Net, ParT).
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


def _init_feature_store() -> dict[str, list]:
    return {
        # Track constituent features
        "track_pt": [],
        "track_eta": [],
        "track_phi": [],
        "track_charge": [],
        "track_x": [],
        "track_y": [],
        "track_z": [],
        "track_t": [],
        "track_mass": [],
        "track_pid": [],
        "n_tracks": [],

        # Tower constituent features
        "tower_et": [],
        "tower_eta": [],
        "tower_phi": [],
        "tower_energy": [],
        "tower_t": [],
        "tower_eem": [],
        "tower_ehad": [],
        "n_towers": [],
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
) -> tuple[dict[str, list], dict[str, int], dict[str, int]]:
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
    diagnostics = {}

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

        # Extract Track info
        track_pts = []
        track_etas = []
        track_phis = []
        track_charges = []
        track_xs = []
        track_ys = []
        track_zs = []
        track_ts = []
        track_masses = []
        track_pids = []

        n_tr = int(tree.Track.GetEntries())
        for tr_idx in range(n_tr):
            tr = tree.Track.At(tr_idx)
            track_pts.append(float(tr.PT))
            track_etas.append(float(tr.Eta))
            track_phis.append(float(tr.Phi))
            track_charges.append(int(tr.Charge))
            track_xs.append(float(tr.X))
            track_ys.append(float(tr.Y))
            track_zs.append(float(tr.Z))
            track_ts.append(float(tr.T))
            track_masses.append(float(tr.Mass))
            track_pids.append(int(tr.PID))

        features["track_pt"].append(track_pts)
        features["track_eta"].append(track_etas)
        features["track_phi"].append(track_phis)
        features["track_charge"].append(track_charges)
        features["track_x"].append(track_xs)
        features["track_y"].append(track_ys)
        features["track_z"].append(track_zs)
        features["track_t"].append(track_ts)
        features["track_mass"].append(track_masses)
        features["track_pid"].append(track_pids)
        features["n_tracks"].append(n_tr)

        # Extract Tower info
        tower_ets = []
        tower_etas = []
        tower_phis = []
        tower_energies = []
        tower_ts = []
        tower_eems = []
        tower_ehads = []

        n_tow = int(tree.Tower.GetEntries())
        for tow_idx in range(n_tow):
            tow = tree.Tower.At(tow_idx)
            tower_ets.append(float(tow.ET))
            tower_etas.append(float(tow.Eta))
            tower_phis.append(float(tow.Phi))
            tower_energies.append(float(tow.E))
            tower_ts.append(float(tow.T))
            tower_eems.append(float(tow.Eem))
            tower_ehads.append(float(tow.Ehad))

        features["tower_et"].append(tower_ets)
        features["tower_eta"].append(tower_etas)
        features["tower_phi"].append(tower_phis)
        features["tower_energy"].append(tower_energies)
        features["tower_t"].append(tower_ts)
        features["tower_eem"].append(tower_eems)
        features["tower_ehad"].append(tower_ehads)
        features["n_towers"].append(n_tow)

    return features, cutflow, diagnostics


def _features_to_table(features: dict[str, list]) -> pa.Table:
    table_dict: dict[str, pa.Array] = {}
    for key, val in features.items():
        if key in {"n_tracks", "n_towers"}:
            table_dict[key] = pa.array(val, type=pa.int32())
        elif key in {"track_charge", "track_pid"}:
            table_dict[key] = pa.array(val, type=pa.list_(pa.int32()))
        else:
            table_dict[key] = pa.array(val, type=pa.list_(pa.float32()))

    return pa.Table.from_pydict(table_dict)


def _write_parquet(features: dict[str, list], output_path: Path, compression: str, row_group_size: int):
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
    diagnostics = {}
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


def _validate_features(features: dict[str, list], n_passed: int):
    # Fail fast for data integrity: required columns exist and row counts agree.
    required_columns = {
        "track_pt", "track_eta", "track_phi", "track_charge", "track_x", "track_y", "track_z", "track_t", "track_mass", "track_pid", "n_tracks",
        "tower_et", "tower_eta", "tower_phi", "tower_energy", "tower_t", "tower_eem", "tower_ehad", "n_towers",
    }

    missing = sorted(required_columns - set(features.keys()))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    for key, val in features.items():
        if len(val) != n_passed:
            raise ValueError(
                f"Column length mismatch for '{key}': {len(val)} != {n_passed}"
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SR-passed Delphes events Track & Tower constituents to a Parquet file.",
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
