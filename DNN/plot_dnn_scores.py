import sys
import os
from pathlib import Path

# Add repo root and DNN directory to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "DNN"))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from DNN.config import (
    load_training_config,
    TASK_DEFINITIONS,
    SCALE_FN,
    WEIGHT_STRATEGIES,
)
from DNN.data_loader import get_all_parquet_files, ParquetFoldDataset, PreScaler
from DNN.model import create_model

def evaluate_task(task_name, output_dir, parquet_dir, weight_strategy="process"):
    print(f"\nEvaluating task: {task_name}...")
    
    # Load config with overrides
    config_path = repo_root / "DNN" / "configs" / "default_config.yaml"
    config = load_training_config(
        config_path=config_path,
        overrides={
            "task": task_name,
            "output_dir": output_dir,
            "parquet_dir": parquet_dir,
            "weight_strategy": weight_strategy,
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Weight strategy: {config.weight_strategy}")
    
    # Get all parquet files
    files_by_process = get_all_parquet_files(str(repo_root / config.parquet_dir))
    
    pre_scaler = PreScaler(SCALE_FN, feature_names=ParquetFoldDataset.FEATURE_COLUMNS)
    task_def = TASK_DEFINITIONS[task_name]
    
    # Results dictionary
    results = {}
    
    for i_fold in range(5):
        print(f"  Processing Fold {i_fold}...")
        # Create and load model
        model = create_model(
            n_features=config.n_features,
            hidden_width=config.hidden_width,
            n_hidden_layers=config.n_hidden_layers,
            dropout_rate=config.dropout_rate,
            init_var_scale=config.init_var_scale,
            device=device,
        )
        
        model_path = repo_root / config.output_dir / config.weight_strategy / f"fold_{i_fold}" / "checkpoints" / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        for process_name, file_paths in sorted(files_by_process.items()):
            if (process_name not in task_def["signal_processes"] and 
                process_name not in task_def["background_processes"]):
                continue
                
            if process_name not in results:
                results[process_name] = {
                    "logits": [],
                    "probs": [],
                    "weights": [],
                    "labels": [],
                    "event_numbers": []
                }
                
            # Create fold test dataset
            dataset = ParquetFoldDataset(
                parquet_file_paths=file_paths,
                process_name=process_name,
                i_fold=i_fold,
                fold_type="test",
                task=task_name,
                pre_scaler=pre_scaler
            )
            
            if len(dataset) == 0:
                continue
                
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            
            with torch.no_grad():
                for batch_features, batch_labels, batch_weights, batch_event_ids in loader:
                    batch_features = batch_features.to(device)
                    logits = model(batch_features).cpu().numpy().squeeze()
                    
                    if logits.ndim == 0:
                        logits = np.array([logits])
                        
                    probs = 1.0 / (1.0 + np.exp(-logits))
                    
                    results[process_name]["logits"].append(logits)
                    results[process_name]["probs"].append(probs)
                    results[process_name]["weights"].append(batch_weights.numpy())
                    results[process_name]["labels"].append(batch_labels.numpy())
                    results[process_name]["event_numbers"].append(batch_event_ids.numpy())

    # Concatenate results
    for p in results:
        results[p]["logits"] = np.concatenate(results[p]["logits"])
        results[p]["probs"] = np.concatenate(results[p]["probs"])
        results[p]["weights"] = np.concatenate(results[p]["weights"])
        results[p]["labels"] = np.concatenate(results[p]["labels"])
        results[p]["event_numbers"] = np.concatenate(results[p]["event_numbers"])
        print(f"    Process {p}: evaluated {len(results[p]['probs'])} events")
        
    return results


def strategy_suffix(weight_strategy):
    return f"_{weight_strategy}"


def strategy_marker(weight_strategy):
    markers = {
        "process": "Baseline: process weight",
        "inverse_event_count": "Weighting: inverse event count",
        "hybrid": "Weighting: process weight x inverse event count",
    }
    return markers.get(weight_strategy, f"Weighting: {weight_strategy}")

def plot_binary_unstacked(results, task_name, output_pdf, weight_strategy="process"):
    print(f"Plotting binary unstacked scores for task {task_name}...")
    
    if task_name == "EW_vs_Background":
        xlabel = r"DNN$_{W^{\pm}W^{\pm}}$ Score"
        sig_label = r"$W^{\pm}W^{\pm}\text{-EW}$"
        bg_label = r"Backgrounds"
    elif task_name == "PolState_LL_vs_LT_TT":
        xlabel = r"DNN$_{\mathrm{pol}}$ Score"
        sig_label = r"$W_{\mathrm{L}}^{\pm}W_{\mathrm{L}}^{\pm}\text{-EW}$"
        bg_label = r"$W_{\mathrm{L}}^{\pm}W_{\mathrm{T}}^{\pm}\text{-EW} + W_{\mathrm{T}}^{\pm}W_{\mathrm{T}}^{\pm}\text{-EW}$"
    else:
        raise ValueError(f"Unknown task name: {task_name}")
        
    all_probs = []
    all_labels = []
    
    for p in results:
        probs = results[p]["probs"]
        if len(probs) == 0:
            continue
        all_probs.append(probs)
        all_labels.append(results[p]["labels"])
        
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    probs_bg = all_probs[all_labels == 0]
    probs_sg = all_probs[all_labels == 1]
    
    bins = np.linspace(0.0, 1.0, 21)
    
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    
    ax.hist(
        probs_sg,
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2.0,
        color="#E63928",
        linestyle="solid",
        label=sig_label
    )
    
    ax.hist(
        probs_bg,
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2.0,
        color="#3378FF",
        linestyle="dashed",
        label=bg_label
    )
        
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized events")
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=1.2, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.0, top=True, right=True)
    ax.minorticks_on()
    ax.legend(frameon=False)

    fig.savefig(output_pdf, format="pdf")
    print(f"Saved plot to {output_pdf}")
    plt.close(fig)

def main():
    # Make sure output figures directory exists
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Task 1: EW_vs_Background
    for weight_strategy in ["process", "inverse_event_count", "hybrid"]:
        try:
            results_ew = evaluate_task(
            task_name="EW_vs_Background",
            output_dir="DNN/results/ew_vs_bg",
            parquet_dir="Sample/Parquet/batch_sr_parquet_mg_sample",
            weight_strategy=weight_strategy,
        )
            suffix = strategy_suffix(weight_strategy)
            plot_binary_unstacked(
                results=results_ew,
                task_name="EW_vs_Background",
                output_pdf=str(figures_dir / f"dnn_score_dist_ew_vs_bg_binary_unstacked{suffix}.pdf"),
                weight_strategy=weight_strategy,
            )
        except Exception as e:
            print(f"Error processing EW_vs_Background with strategy {weight_strategy}: {e}")
            import traceback
            traceback.print_exc()

    # Task 2: PolState_LL_vs_LT_TT
    for weight_strategy in ["process", "inverse_event_count", "hybrid"]:
        try:
            results_pol = evaluate_task(
            task_name="PolState_LL_vs_LT_TT",
            output_dir="DNN/results/pol_ll_vs_lt_tt",
            parquet_dir="Sample/Parquet/batch_sr_parquet_mg_sample",
            weight_strategy=weight_strategy,
        )
            suffix = strategy_suffix(weight_strategy)
            plot_binary_unstacked(
                results=results_pol,
                task_name="PolState_LL_vs_LT_TT",
                output_pdf=str(figures_dir / f"dnn_score_dist_pol_ll_vs_lttt_binary_unstacked{suffix}.pdf"),
                weight_strategy=weight_strategy,
            )
        except Exception as e:
            print(f"Error processing PolState_LL_vs_LT_TT with strategy {weight_strategy}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
