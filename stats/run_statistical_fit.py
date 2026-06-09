#!/usr/bin/env python3
"""
DNN Statistical Inference with pyhf

This script:
1. Loads/evaluates DNN scores for two-stage classifiers (EW_vs_Background, PolState_LL_vs_LT_TT).
2. Performs nested 2D binning and unrolls the results into a 1D histogram.
3. Builds a single-channel pyhf workspace.
4. Performs statistical fitting to compute expected discovery significance and 95% CL upper limits.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add repo root and DNN directory to sys.path to resolve imports correctly
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "DNN"))
sys.path.insert(0, str(repo_root))

import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

# Import modules from the DNN package
from DNN.config import (
    TASK_DEFINITIONS,
    SCALE_FN,
    PROCESS_WEIGHTS,
)
from DNN.data_loader import get_all_parquet_files, ParquetFoldDataset, PreScaler
from DNN.model import create_model

# Import pyhf and set backend to numpy
import pyhf
pyhf.set_backend("numpy")


def get_total_nominal(file_paths):
    """
    Read the total nominal generated events for a process from all run cutflow files.
    """
    total_nominal = 0
    for fpath in file_paths:
        cutflow_path = fpath + ".cutflow.json"
        if os.path.exists(cutflow_path):
            try:
                with open(cutflow_path, "r") as f:
                    cutflow = json.load(f)
                    total_nominal += cutflow.get("cutflow", {}).get("Total", 0)
            except Exception as e:
                print(f"Error reading cutflow {cutflow_path}: {e}")
        else:
            print(f"Warning: cutflow file not found for {fpath}")
    return total_nominal


def evaluate_dnn_scores(parquet_dir, output_dir_ew, output_dir_pol, pol_task_name, device=None):
    """
    Evaluate EW_vs_Background and polarization models on test splits of the 5 folds.
    Returns scores and weights mapped by process.
    """
    parquet_path = repo_root / parquet_dir
    ew_path = repo_root / output_dir_ew
    pol_path = repo_root / output_dir_pol

    # Verify path existence
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_path}")
    if not ew_path.exists():
        raise FileNotFoundError(f"EW model directory not found: {ew_path}")
    if not pol_path.exists():
        raise FileNotFoundError(f"Pol model directory not found: {pol_path}")

    # Verify fold checkpoints existence
    for i in range(5):
        if not (ew_path / f"fold_{i}/checkpoints/best_model.pt").exists():
            raise FileNotFoundError(f"Missing EW model checkpoint for fold_{i} at {ew_path / f'fold_{i}/checkpoints/best_model.pt'}")
        if not (pol_path / f"fold_{i}/checkpoints/best_model.pt").exists():
            raise FileNotFoundError(f"Missing Pol model checkpoint for fold_{i} at {pol_path / f'fold_{i}/checkpoints/best_model.pt'}")

    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nEvaluating models on actual Parquet files using device: {device}")

        files_by_process = get_all_parquet_files(str(parquet_path))
        pre_scaler = PreScaler(SCALE_FN, feature_names=ParquetFoldDataset.FEATURE_COLUMNS)

        results = {}

        for process_name, file_paths in sorted(files_by_process.items()):
            # Skip the mixed process to avoid double counting
            if process_name == "WWjj_EW":
                continue

            # Ensure the process is active in either task definition
            is_valid = False
            for task_name in ["EW_vs_Background", pol_task_name]:
                task_def = TASK_DEFINITIONS[task_name]
                if (process_name in task_def["signal_processes"] or 
                    process_name in task_def["background_processes"]):
                    is_valid = True
                    break
            if not is_valid:
                continue
            # Calculate total passed events (in SR) for normalization
            import pyarrow.parquet as pq
            n_passed_total = 0
            for fpath in file_paths:
                try:
                    n_passed_total += pq.read_metadata(fpath).num_rows
                except Exception as e:
                    print(f"Error reading metadata for {fpath}: {e}")
            
            if n_passed_total <= 0:
                print(f"Warning: total passed events for {process_name} is {n_passed_total}; skipping.")
                continue

            xsec_fb = PROCESS_WEIGHTS.get(process_name, 0.0)
            lumi = 139.0
            event_weight = (xsec_fb * lumi) / n_passed_total
            results[process_name] = {
                "scores_ew": [],
                "scores_pol": [],
                "weights": []
            }

            for i_fold in range(5):
                # Load EW model for this fold
                model_ew = create_model(n_features=32, hidden_width=128, n_hidden_layers=4, dropout_rate=0.3, init_var_scale=2.952, device=device)
                model_ew_path = ew_path / f"fold_{i_fold}/checkpoints/best_model.pt"
                model_ew.load_state_dict(torch.load(model_ew_path, map_location=device, weights_only=True))
                model_ew.eval()

                # Load Pol model for this fold
                model_pol = create_model(n_features=32, hidden_width=128, n_hidden_layers=4, dropout_rate=0.3, init_var_scale=2.952, device=device)
                model_pol_path = pol_path / f"fold_{i_fold}/checkpoints/best_model.pt"
                model_pol.load_state_dict(torch.load(model_pol_path, map_location=device, weights_only=True))
                model_pol.eval()

                # Determine which task makes this process valid for loading
                task_to_load = "EW_vs_Background"
                if (process_name in TASK_DEFINITIONS[pol_task_name]["signal_processes"] or 
                    process_name in TASK_DEFINITIONS[pol_task_name]["background_processes"]):
                    task_to_load = pol_task_name

                # Load dataset for test split of this fold
                dataset = ParquetFoldDataset(
                    parquet_file_paths=file_paths,
                    process_name=process_name,
                    i_fold=i_fold,
                    fold_type="test",
                    task=task_to_load,
                    weight_strategy="process",
                    pre_scaler=pre_scaler
                )

                if len(dataset) == 0:
                    continue

                loader = DataLoader(dataset, batch_size=256, shuffle=False)

                with torch.no_grad():
                    for batch_features, _, _, _ in loader:
                        batch_features = batch_features.to(device)

                        # Evaluate EW model
                        logits_ew = model_ew(batch_features).cpu().numpy().squeeze()
                        if logits_ew.ndim == 0:
                            logits_ew = np.array([logits_ew])
                        probs_ew = 1.0 / (1.0 + np.exp(-logits_ew))

                        # Evaluate Pol model
                        logits_pol = model_pol(batch_features).cpu().numpy().squeeze()
                        if logits_pol.ndim == 0:
                            logits_pol = np.array([logits_pol])
                        probs_pol = 1.0 / (1.0 + np.exp(-logits_pol))

                        results[process_name]["scores_ew"].append(probs_ew)
                        results[process_name]["scores_pol"].append(probs_pol)
                        results[process_name]["weights"].append(np.full(len(probs_ew), event_weight))

            # Concatenate results across folds
            if len(results[process_name]["scores_ew"]) > 0:
                results[process_name]["scores_ew"] = np.concatenate(results[process_name]["scores_ew"])
                results[process_name]["scores_pol"] = np.concatenate(results[process_name]["scores_pol"])
                results[process_name]["weights"] = np.concatenate(results[process_name]["weights"])
                print(f"  Processed {process_name:25s}: N_events_test = {len(results[process_name]['scores_ew']):6d}, total yield = {np.sum(results[process_name]['weights']):.2f}")
            else:
                del results[process_name]

        return results

    except Exception as e:
        print(f"\n[Error] Error evaluating actual data: {e}")
        import traceback
        traceback.print_exc()
        raise e


def unroll_2d_to_1d(scores_ew, scores_pol, weights, ww_edges, pol_edges):
    """
    Perform nested 2D binning and unroll into a 1D yield array.
    """
    n_bins_total = sum(len(edges) - 1 for edges in pol_edges)
    hist = np.zeros(n_bins_total)

    # Bin index along the EW dimension
    x_bins = np.digitize(scores_ew, ww_edges) - 1
    x_bins = np.clip(x_bins, 0, len(ww_edges) - 2)

    bin_offset = 0
    for idx_x in range(len(ww_edges) - 1):
        edges_y = pol_edges[idx_x]
        n_bins_y = len(edges_y) - 1

        mask = (x_bins == idx_x)
        if np.any(mask):
            y_subset = scores_pol[mask]
            w_subset = weights[mask]

            y_bins = np.digitize(y_subset, edges_y) - 1
            y_bins = np.clip(y_bins, 0, n_bins_y - 1)

            for idx_y in range(n_bins_y):
                hist[bin_offset + idx_y] = np.sum(w_subset[y_bins == idx_y])

        bin_offset += n_bins_y

    return hist


# ============================================================================
# FIT CONFIGURATIONS
# ============================================================================

FIT_CONFIGS = {
    "LL": {
        "pol_task_name": "PolState_LL_vs_LT_TT",
        "default_output_dir_pol": "DNN/results/pol_ll_vs_lt_tt/hybrid",
        "signal_processes": ["WWjj_EW_LL_WW_cmf"],
        "norm_bkg_processes": ["WWjj_EW_LT_WW_cmf", "WWjj_EW_TT_WW_cmf"],
        "signal_name": "Signal_LL",
        "norm_bkg_name": "Background_TX",
        "poi_name": "mu_L",
        "norm_bkg_param": "mu_T",
        "ww_edges": [0.0, 0.2, 0.6, 1.0],
        "pol_edges": [
            [0.0, 0.4, 0.7, 1.0],               # Bin 1 of EW score
            [0.0, 0.5, 0.8, 1.0],               # Bin 2 of EW score
            [0.0, 0.4, 0.6, 0.7, 0.8, 1.0]      # Bin 3 of EW score
        ]
    },
    "LL_LT": {
        "pol_task_name": "PolState_LL_LT_vs_TT",
        "default_output_dir_pol": "DNN/results/pol_ll_lt_vs_tt/hybrid",
        "signal_processes": ["WWjj_EW_LL_WW_cmf", "WWjj_EW_LT_WW_cmf"],
        "norm_bkg_processes": ["WWjj_EW_TT_WW_cmf"],
        "signal_name": "Signal_LL_LT",
        "norm_bkg_name": "Background_TT",
        "poi_name": "mu_LL_LT",
        "norm_bkg_param": "mu_T",
        "ww_edges": [0.0, 0.3, 0.7, 1.0],
        "pol_edges": [
            [0.0, 0.3, 0.7, 1.0],               # Bin 1 of EW score
            [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0],               # Bin 2 of EW score
            [0.0, 0.15, 0.35, 0.45, 0.55, 0.7, 1.0]      # Bin 3 of EW score
        ]
    }
}


def build_pyhf_workspace(hist_sig, hist_bg_tx, hist_bg_other, sig_name="Signal_LL", bg_tx_name="Background_TX", poi_name="mu_L", norm_bkg_param="mu_T"):
    """
    Construct a pyhf workspace dictionary representing a single channel.
    """
    # Sum of all backgrounds is the nominal expectation
    nominal_bg = hist_bg_tx + hist_bg_other

    workspace_dict = {
        "channels": [
            {
                "name": "single_channel",
                "samples": [
                    {
                        "name": sig_name,
                        "data": list(hist_sig),
                        "modifiers": [
                            {
                                "name": poi_name,
                                "type": "normfactor",
                                "data": None
                            }
                        ]
                    },
                    {
                        "name": bg_tx_name,
                        "data": list(hist_bg_tx),
                        "modifiers": [
                            {
                                "name": norm_bkg_param,
                                "type": "normfactor",
                                "data": None
                            }
                        ]
                    },
                    {
                        "name": "Background_Other",
                        "data": list(hist_bg_other),
                        "modifiers": []
                    }
                ]
            }
        ],
        "measurements": [
            {
                "name": "measurement_name",
                "config": {
                    "poi": poi_name,
                    "parameters": [
                        {
                            "name": poi_name,
                            "bounds": [[0.0, 10.0]],
                            "inits": [1.0]
                        },
                        {
                            "name": norm_bkg_param,
                            "bounds": [[0.0, 5.0]],
                            "inits": [1.0]
                        }
                    ]
                }
            }
        ],
        "observations": [
            {
                "name": "single_channel",
                "data": list(nominal_bg)
            }
        ],
        "version": "1.0.0"
    }
    return workspace_dict


def perform_statistical_fit(workspace_dict, poi_name="mu_L"):
    """
    Perform statistical fit, generating Asimov data, and computing expected significance 
    and upper limits.
    """
    ws = pyhf.Workspace(workspace_dict)
    model = ws.model()
    obs_data = ws.data(model)

    print("\n--- Model Configuration ---")
    print(f"Parameters: {model.config.par_names}")
    print(f"Suggested bounds: {model.config.suggested_bounds()}")

    # 1. Fit under Background-Only hypothesis (POI = 0) to get Asimov data
    pars_bkg = pyhf.infer.mle.fixed_poi_fit(0.0, obs_data, model)
    print(f"Best-fit parameters under {poi_name}=0: {pars_bkg}")

    asimov_data_0 = model.expected_data(pars_bkg)
    print(f"Background-only Asimov data yields:\n  {asimov_data_0.tolist()}")

    # 2. Get Asimov data under Signal+Background hypothesis (POI = 1)
    poi_idx = model.config.par_order.index(poi_name)
    pars_sig = list(pars_bkg)
    pars_sig[poi_idx] = 1.0
    asimov_data_1 = model.expected_data(pars_sig)
    print(f"Signal+Background Asimov data ({poi_name}=1) yields:\n  {asimov_data_1.tolist()}")

    # 3. Expected Discovery Significance (Z_0) for POI = 1
    # We run hypotest with null hypothesis POI = 0 on signal Asimov data
    print(f"\nCalculating expected discovery significance for {poi_name}=1...")
    p_val_obs, p_val_exp = pyhf.infer.hypotest(
        0.0,
        asimov_data_1,
        model,
        test_stat="q0",
        return_expected=True
    )
    Z_0 = stats.norm.isf(p_val_obs)

    # 4. Expected 95% CL Upper Limit on POI
    # We run upper limit on background-only Asimov data
    print(f"Calculating expected 95% CL upper limits on {poi_name}...")
    obs_limit, exp_limits = pyhf.infer.intervals.upper_limits.upper_limit(
        asimov_data_0,
        model
    )

    return Z_0, obs_limit, exp_limits


def main():
    parser = argparse.ArgumentParser(description="Run statistical fit using pyhf.")
    parser.add_argument(
        "--signal-mode",
        choices=["LL", "LL_LT"],
        default="LL",
        help="Signal mode for fitting: 'LL' (Signal=LL, Bkg_TX=LT+TT) or 'LL_LT' (Signal=LL+LT, Bkg_TX=TT)."
    )
    parser.add_argument(
        "--output-dir-pol",
        default=None,
        help="Custom path to polarization model directory (default depends on signal mode)."
    )
    parser.add_argument(
        "--ww-edges",
        type=float,
        nargs="+",
        default=None,
        help="Custom binning edges along EW score dimension."
    )
    parser.add_argument(
        "--pol-edges",
        type=str,
        default=None,
        help="Custom binning edges along polarization score dimension as a JSON string."
    )
    args = parser.parse_args()

    cfg = FIT_CONFIGS[args.signal_mode]
    pol_task_name = cfg["pol_task_name"]

    parquet_dir = "Sample/Parquet/batch_sr_parquet_mg_sample"
    output_dir_ew = "DNN/results/ew_vs_bg/process"
    output_dir_pol = args.output_dir_pol if args.output_dir_pol is not None else cfg["default_output_dir_pol"]

    print(f"Running fit in mode: {args.signal_mode}")
    print(f"  Signal processes: {cfg['signal_processes']}")
    print(f"  Normalized background processes: {cfg['norm_bkg_processes']}")
    print(f"  Polarization task name: {pol_task_name}")
    print(f"  Polarization model path: {output_dir_pol}")

    scores_by_process = evaluate_dnn_scores(
        parquet_dir=parquet_dir,
        output_dir_ew=output_dir_ew,
        output_dir_pol=output_dir_pol,
        pol_task_name=pol_task_name
    )

    # Separate events into the three samples based on config
    sig_ew, sig_pol, sig_w = [], [], []
    tx_ew, tx_pol, tx_w = [], [], []
    other_ew, other_pol, other_w = [], [], []

    for process_name, data in scores_by_process.items():
        if process_name in cfg["signal_processes"]:
            sig_ew.append(data["scores_ew"])
            sig_pol.append(data["scores_pol"])
            sig_w.append(data["weights"])
        elif process_name in cfg["norm_bkg_processes"]:
            tx_ew.append(data["scores_ew"])
            tx_pol.append(data["scores_pol"])
            tx_w.append(data["weights"])
        else:
            other_ew.append(data["scores_ew"])
            other_pol.append(data["scores_pol"])
            other_w.append(data["weights"])

    # Concatenate the lists
    sig_ew = np.concatenate(sig_ew) if sig_ew else np.array([])
    sig_pol = np.concatenate(sig_pol) if sig_pol else np.array([])
    sig_w = np.concatenate(sig_w) if sig_w else np.array([])

    tx_ew = np.concatenate(tx_ew) if tx_ew else np.array([])
    tx_pol = np.concatenate(tx_pol) if tx_pol else np.array([])
    tx_w = np.concatenate(tx_w) if tx_w else np.array([])

    other_ew = np.concatenate(other_ew) if other_ew else np.array([])
    other_pol = np.concatenate(other_pol) if other_pol else np.array([])
    other_w = np.concatenate(other_w) if other_w else np.array([])

    print(f"\n--- Process Categorization Results (Yields) ---")
    print(f"  {cfg['signal_name']} Yield        : {np.sum(sig_w):.4f}")
    print(f"  {cfg['norm_bkg_name']} Yield    : {np.sum(tx_w):.4f}")
    print(f"  Background_Other Yield : {np.sum(other_w):.4f}")

    # Step 2: Binning and Unrolling
    ww_edges = args.ww_edges if args.ww_edges is not None else cfg["ww_edges"]
    if args.pol_edges is not None:
        try:
            pol_edges = json.loads(args.pol_edges)
        except Exception as e:
            raise ValueError(f"Failed to parse --pol-edges as JSON: {e}")
    else:
        pol_edges = cfg["pol_edges"]

    print(f"\nApplying binning:")
    print(f"  EW score edges: {ww_edges}")
    print(f"  Pol score edges: {pol_edges}")

    hist_sig = unroll_2d_to_1d(sig_ew, sig_pol, sig_w, ww_edges, pol_edges)
    hist_bg_tx = unroll_2d_to_1d(tx_ew, tx_pol, tx_w, ww_edges, pol_edges)
    hist_bg_other = unroll_2d_to_1d(other_ew, other_pol, other_w, ww_edges, pol_edges)

    print(f"\n--- Unrolled 1D Histogram Bin Yields ({len(hist_sig)} bins) ---")
    print(f"  {cfg['signal_name']}        : {hist_sig.round(4).tolist()}")
    print(f"  {cfg['norm_bkg_name']}    : {hist_bg_tx.round(4).tolist()}")
    print(f"  Background_Other : {hist_bg_other.round(4).tolist()}")

    # Step 3: Build pyhf workspace
    workspace_dict = build_pyhf_workspace(
        hist_sig, hist_bg_tx, hist_bg_other,
        sig_name=cfg["signal_name"],
        bg_tx_name=cfg["norm_bkg_name"],
        poi_name=cfg["poi_name"],
        norm_bkg_param=cfg["norm_bkg_param"]
    )

    # Step 4: Perform Statistical Fit
    Z_0, obs_limit, exp_limits = perform_statistical_fit(workspace_dict, poi_name=cfg["poi_name"])

    # Step 5: Print final results
    print("\n================ STATISTICAL INFERENCE RESULTS ================")
    print(f"Expected Discovery Significance Z_0 for {cfg['poi_name']}=1: {Z_0:.4f} sigma")
    print(f"Median Expected 95% CL Upper Limit on {cfg['poi_name']}   : {exp_limits[2]:.4f}")
    print("Expected 95% CL Upper Limit Bands:")
    print(f"  -2 sigma: {exp_limits[0]:.4f}")
    print(f"  -1 sigma: {exp_limits[1]:.4f}")
    print(f"  +1 sigma: {exp_limits[3]:.4f}")
    print(f"  +2 sigma: {exp_limits[4]:.4f}")
    print("===============================================================")


if __name__ == "__main__":
    main()
