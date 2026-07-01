#!/usr/bin/env python3
"""Export SR-passed Delphes events to a Parquet table with Refined Constituents.

Classifies all particles (leptons, jet constituents, and remaining particles) 
and tags their physical origin (part_tag) using exact spatial matching.
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
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure `selection` package import works.
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
        # Event level & reconstructed objects metadata
        "event_number": [],
        "l1_pt": [], "l1_eta": [], "l1_phi": [], "l1_charge": [], "l1_flavor": [],
        "l2_pt": [], "l2_eta": [], "l2_phi": [], "l2_charge": [], "l2_flavor": [],
        "j1_pt": [], "j1_eta": [], "j1_phi": [], "j1_mass": [],
        "j2_pt": [], "j2_eta": [], "j2_phi": [], "j2_mass": [],
        "met_et": [], "met_phi": [],
        
        # Flattened constituent particles list
        "part_pt": [],
        "part_eta": [],
        "part_phi": [],
        "part_charge": [],
        "part_mass": [],
        "part_type": [],  # 0: Track, 1: Tower, 2: Electron, 3: Muon
        "part_tag": [],   # 0: Lepton 1, 1: Lepton 2, 2: Jet 1 constituent, 3: Jet 2 constituent, 4: Unassociated
    }


def _iter_entry_ranges(start_index: int, end_index: int, chunk_size: int):
    start_index = max(0, int(start_index))
    end_index = max(start_index, int(end_index))
    chunk_size = max(1, int(chunk_size))

    for chunk_start in range(start_index, end_index, chunk_size):
        chunk_end = min(end_index, chunk_start + chunk_size)
        yield chunk_start, chunk_end


def _is_in_electron_crack(eta):
    aeta = abs(eta)
    return 1.37 < aeta < 1.52


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

        # Extract reconstructed level objects
        leptons = objs["leptons"]
        l1_p4 = leptons[0]["p4"]
        l2_p4 = leptons[1]["p4"]
        l1_flavor = 0 if leptons[0]["flavor"] == "e" else 1 # 0: e, 1: mu
        l2_flavor = 0 if leptons[1]["flavor"] == "e" else 1
        
        j1_p4 = objs["jets"][0]
        j2_p4 = objs["jets"][1]
        
        met = objs["met"]

        # ----------------------------------------------------
        # Extract native Delphes Jets to fetch constituents
        # ----------------------------------------------------
        sr_jets = []
        for j in tree.Jet:
            if abs(j.Eta) < 4.5 and j.PT > 25.0:
                sr_jets.append(j)
        sr_jets.sort(key=lambda x: x.PT, reverse=True)
        
        jet1 = sr_jets[0]
        jet2 = sr_jets[1]

        # Build coordinate key sets for constituents (precision up to 5 decimals to avoid float variations)
        jet1_keys = set()
        n_const1 = jet1.Constituents.GetEntries()
        for c_idx in range(n_const1):
            c_obj = jet1.Constituents.At(c_idx)
            if c_obj:
                jet1_keys.add((round(c_obj.Eta, 5), round(c_obj.Phi, 5)))
                
        jet2_keys = set()
        n_const2 = jet2.Constituents.GetEntries()
        for c_idx in range(n_const2):
            c_obj = jet2.Constituents.At(c_idx)
            if c_obj:
                jet2_keys.add((round(c_obj.Eta, 5), round(c_obj.Phi, 5)))

        # ----------------------------------------------------
        # Collect and Tag all particles
        # ----------------------------------------------------
        event_particles = []

        # Part 1: Reconstruction Leptons themselves
        # Lepton 1 (tag 0)
        l1_type = 2 if leptons[0]["flavor"] == "e" else 3
        event_particles.append({
            "pt": float(l1_p4.Pt()),
            "eta": float(l1_p4.Eta()),
            "phi": float(l1_p4.Phi()),
            "charge": float(leptons[0]["charge"]),
            "mass": float(l1_p4.M()),
            "type": l1_type,
            "tag": 0,
        })
        # Lepton 2 (tag 1)
        l2_type = 2 if leptons[1]["flavor"] == "e" else 3
        event_particles.append({
            "pt": float(l2_p4.Pt()),
            "eta": float(l2_p4.Eta()),
            "phi": float(l2_p4.Phi()),
            "charge": float(leptons[1]["charge"]),
            "mass": float(l2_p4.M()),
            "type": l2_type,
            "tag": 1,
        })

        # Part 2: Tracks (tag 2 if Jet 1, 3 if Jet 2, else 4)
        n_tracks = tree.Track.GetEntries()
        for t_idx in range(n_tracks):
            tr = tree.Track.At(t_idx)
            key = (round(tr.Eta, 5), round(tr.Phi, 5))
            
            if key in jet1_keys:
                tag = 2
            elif key in jet2_keys:
                tag = 3
            else:
                tag = 4
                
            event_particles.append({
                "pt": float(tr.PT),
                "eta": float(tr.Eta),
                "phi": float(tr.Phi),
                "charge": float(tr.Charge),
                "mass": float(tr.Mass),
                "type": 0,  # Track
                "tag": tag,
            })

        # Part 3: Towers (tag 2 if Jet 1, 3 if Jet 2, else 4)
        n_towers = tree.Tower.GetEntries()
        for tow_idx in range(n_towers):
            tow = tree.Tower.At(tow_idx)
            key = (round(tow.Eta, 5), round(tow.Phi, 5))
            
            if key in jet1_keys:
                tag = 2
            elif key in jet2_keys:
                tag = 3
            else:
                tag = 4
                
            event_particles.append({
                "pt": float(tow.ET),
                "eta": float(tow.Eta),
                "phi": float(tow.Phi),
                "charge": 0.0,
                "mass": 0.0,
                "type": 1,  # Tower
                "tag": tag,
            })

        # Sort all particles by pt descending
        event_particles.sort(key=lambda x: x["pt"], reverse=True)

        # Store to features dict
        features["event_number"].append(int(i_start + i))
        
        features["l1_pt"].append(float(l1_p4.Pt()))
        features["l1_eta"].append(float(l1_p4.Eta()))
        features["l1_phi"].append(float(l1_p4.Phi()))
        features["l1_charge"].append(int(leptons[0]["charge"]))
        features["l1_flavor"].append(int(l1_flavor))
        
        features["l2_pt"].append(float(l2_p4.Pt()))
        features["l2_eta"].append(float(l2_p4.Eta()))
        features["l2_phi"].append(float(l2_p4.Phi()))
        features["l2_charge"].append(int(leptons[1]["charge"]))
        features["l2_flavor"].append(int(l2_flavor))

        features["j1_pt"].append(float(j1_p4.Pt()))
        features["j1_eta"].append(float(j1_p4.Eta()))
        features["j1_phi"].append(float(j1_p4.Phi()))
        features["j1_mass"].append(float(j1_p4.M()))

        features["j2_pt"].append(float(j2_p4.Pt()))
        features["j2_eta"].append(float(j2_p4.Eta()))
        features["j2_phi"].append(float(j2_p4.Phi()))
        features["j2_mass"].append(float(j2_p4.M()))

        features["met_et"].append(float(met.MET))
        features["met_phi"].append(float(met.Phi))

        features["part_pt"].append([p["pt"] for p in event_particles])
        features["part_eta"].append([p["eta"] for p in event_particles])
        features["part_phi"].append([p["phi"] for p in event_particles])
        features["part_charge"].append([p["charge"] for p in event_particles])
        features["part_mass"].append([p["mass"] for p in event_particles])
        features["part_type"].append([p["type"] for p in event_particles])
        features["part_tag"].append([p["tag"] for p in event_particles])

    return features, cutflow, diagnostics


def _features_to_table(features: dict[str, list]) -> pa.Table:
    table_dict: dict[str, pa.Array] = {}
    scalar_ints = {"event_number", "l1_charge", "l1_flavor", "l2_charge", "l2_flavor"}
    scalar_floats = {
        "l1_pt", "l1_eta", "l1_phi", 
        "l2_pt", "l2_eta", "l2_phi", 
        "j1_pt", "j1_eta", "j1_phi", "j1_mass",
        "j2_pt", "j2_eta", "j2_phi", "j2_mass",
        "met_et", "met_phi"
    }
    list_ints = {"part_type", "part_tag"}
    list_floats = {"part_pt", "part_eta", "part_phi", "part_charge", "part_mass"}

    for key, val in features.items():
        if key in scalar_ints:
            table_dict[key] = pa.array(val, type=pa.int32())
        elif key in scalar_floats:
            table_dict[key] = pa.array(val, type=pa.float32())
        elif key in list_ints:
            table_dict[key] = pa.array(val, type=pa.list_(pa.int32()))
        elif key in list_floats:
            table_dict[key] = pa.array(val, type=pa.list_(pa.float32()))
        else:
            table_dict[key] = pa.array(val)

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
    required_columns = {
        "event_number",
        "l1_pt", "l1_eta", "l1_phi", "l1_charge", "l1_flavor",
        "l2_pt", "l2_eta", "l2_phi", "l2_charge", "l2_flavor",
        "j1_pt", "j1_eta", "j1_phi", "j1_mass",
        "j2_pt", "j2_eta", "j2_phi", "j2_mass",
        "met_et", "met_phi",
        "part_pt", "part_eta", "part_phi", "part_charge", "part_mass", "part_type", "part_tag",
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

    # Sidecar metadata supports reproducibility/debugging
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
