"""Parallelized event selection utilities."""

from __future__ import annotations

import concurrent.futures
import os

import numpy as np
import ROOT
from tqdm import tqdm

from .cuts import pass_SR_cuts, pass_WZ_CR_cuts, pass_low_mjj_cr_cuts


def _get_cut_function(cut_func_name):
    """Get cut function by name to enable pickling across processes."""
    cut_functions = {
        "pass_SR_cuts": pass_SR_cuts,
        "pass_low_mjj_cr_cuts": pass_low_mjj_cr_cuts,
        "pass_WZ_CR_cuts": pass_WZ_CR_cuts,
    }
    if cut_func_name not in cut_functions:
        raise ValueError(f"Unknown cut function: {cut_func_name}")
    return cut_functions[cut_func_name]


def _selection_cut_on_entry_range(tree, start_index, end_index, cut_func):
    """Core selection loop over a half-open entry range [start_index, end_index)."""
    cutflow = {
        "Total": 0,
        "lepton cut": 0,
        "MET cut": 0,
        "Jet cut": 0,
    }

    deta_ll, dphi_jj, mT = [], [], []
    pt_j2, m_jj, deltaR_jj = [], [], []

    for entry_index in range(start_index, end_index):
        tree.GetEntry(entry_index)
        cutflow["Total"] += 1

        stage, region_objects = cut_func(tree, return_objects=True)
        if stage >= 1:
            cutflow["lepton cut"] += 1
        if stage >= 2:
            cutflow["MET cut"] += 1
        if stage >= 3:
            cutflow["Jet cut"] += 1

        if stage < 3:
            continue

        leptons = region_objects["leptons"]
        l1, l2 = leptons[0], leptons[1]
        ll_p4 = region_objects["ll_p4"]
        met = region_objects["met"]
        jets = region_objects["jets"]
        j1, j2 = jets[0], jets[1]

        deta_ll.append(abs(l1["p4"].Eta() - l2["p4"].Eta()))
        dphi_jj.append(abs(j1.DeltaPhi(j2)))

        met_px = met.MET * np.cos(met.Phi)
        met_py = met.MET * np.sin(met.Phi)
        Et_ll = np.sqrt(ll_p4.Pt() ** 2 + ll_p4.M() ** 2)
        px_tot = ll_p4.Px() + met_px
        py_tot = ll_p4.Py() + met_py
        mT_sq = (Et_ll + met.MET) ** 2 - (px_tot ** 2 + py_tot ** 2)
        mT.append(np.sqrt(mT_sq) if mT_sq > 0 else 0.0)

        pt_j2.append(j2.Pt())
        m_jj.append((j1 + j2).M())
        deltaR_jj.append(j1.DeltaR(j2))

    return {
        "cutflow": cutflow,
        "deta_ll": np.array(deta_ll),
        "dphi_jj": np.array(dphi_jj),
        "mT": np.array(mT),
        "pt_j2": np.array(pt_j2),
        "m_jj": np.array(m_jj),
        "deltaR_jj": np.array(deltaR_jj),
    }


def selection_cut_with_region(tree, cut_func):
    n_entries = int(tree.GetEntries())
    return _selection_cut_on_entry_range(tree, 0, n_entries, cut_func)


def _selection_cut_chunk(root_path, start, end, cut_func_name):
    cut_func = _get_cut_function(cut_func_name)

    f = ROOT.TFile(root_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {root_path}")

    tree = f.Get("Delphes")
    if tree is None:
        f.Close()
        raise RuntimeError(f"Cannot find Delphes tree in {root_path}")

    n_entries = int(tree.GetEntries())
    i0 = max(0, int(start))
    i1 = min(n_entries, int(end))
    results = _selection_cut_on_entry_range(tree, i0, i1, cut_func)
    f.Close()
    return results


def _merge_selection_results(results_list):
    if not results_list:
        return {
            "cutflow": {"Total": 0, "lepton cut": 0, "MET cut": 0, "Jet cut": 0},
            "deta_ll": np.array([]),
            "dphi_jj": np.array([]),
            "mT": np.array([]),
            "pt_j2": np.array([]),
            "m_jj": np.array([]),
            "deltaR_jj": np.array([]),
        }

    merged_cutflow = {
        "Total": 0,
        "lepton cut": 0,
        "MET cut": 0,
        "Jet cut": 0,
    }

    arrays_to_merge = {
        "deta_ll": [],
        "dphi_jj": [],
        "mT": [],
        "pt_j2": [],
        "m_jj": [],
        "deltaR_jj": [],
    }

    for result in results_list:
        for key in merged_cutflow:
            merged_cutflow[key] += result["cutflow"][key]
        for key in arrays_to_merge:
            arr = result.get(key, np.array([]))
            if arr.size > 0:
                arrays_to_merge[key].append(arr)

    return {
        "cutflow": merged_cutflow,
        "deta_ll": np.concatenate(arrays_to_merge["deta_ll"]) if arrays_to_merge["deta_ll"] else np.array([]),
        "dphi_jj": np.concatenate(arrays_to_merge["dphi_jj"]) if arrays_to_merge["dphi_jj"] else np.array([]),
        "mT": np.concatenate(arrays_to_merge["mT"]) if arrays_to_merge["mT"] else np.array([]),
        "pt_j2": np.concatenate(arrays_to_merge["pt_j2"]) if arrays_to_merge["pt_j2"] else np.array([]),
        "m_jj": np.concatenate(arrays_to_merge["m_jj"]) if arrays_to_merge["m_jj"] else np.array([]),
        "deltaR_jj": np.concatenate(arrays_to_merge["deltaR_jj"]) if arrays_to_merge["deltaR_jj"] else np.array([]),
    }


def selection_cut_parallel(sample_groups, cut_func_name, num_workers=None, chunks_per_file=1):
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) // 2)
    num_workers = max(1, int(num_workers))
    chunks_per_file = max(1, int(chunks_per_file))

    grouped_results = {}
    group_items = list(sample_groups.items())

    for group_idx, (group_name, root_files) in enumerate(group_items, 1):
        print(
            f"\n[{group_idx}/{len(group_items)}] {group_name}: "
            f"Processing {len(root_files)} file(s) with {num_workers} worker(s), "
            f"{chunks_per_file} chunk(s) per file..."
        )

        tasks = []
        for root_path in root_files:
            f = ROOT.TFile(root_path)
            if f and not f.IsZombie():
                tree = f.Get("Delphes")
                if tree:
                    n_entries = int(tree.GetEntries())
                    chunk_size = max(1, n_entries // chunks_per_file)
                    for chunk_idx in range(chunks_per_file):
                        start = chunk_idx * chunk_size
                        end = (chunk_idx + 1) * chunk_size if chunk_idx < chunks_per_file - 1 else n_entries
                        tasks.append((root_path, start, end))
                f.Close()

        chunk_results = []
        if tasks:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_selection_cut_chunk, root_path, start, end, cut_func_name): (root_path, start, end)
                    for root_path, start, end in tasks
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"    {group_name}",
                    leave=True,
                ):
                    try:
                        chunk_results.append(future.result())
                    except Exception as exc:
                        root_path, start, end = futures[future]
                        print(f"  [Error] chunk task failed for {root_path} [{start}, {end}): {exc}")

        merged = _merge_selection_results(chunk_results)
        grouped_results[group_name] = merged
        print(f"  ✓ [{group_name}] done -> Total={merged['cutflow']['Total']} events, passed={merged['cutflow']['Jet cut']}")

    return grouped_results
