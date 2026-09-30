"""Event Image visualization tool for high-statistics (50k events) analysis.

Plots:
1. Multi-event average energy/pT deposition heatmaps (EW vs QCD background, 50k events).
2. Polarization 3-state comparison heatmaps (LL vs LT vs TT, 50k events).
3. Difference map (LL - TT) to inspect subtle channel discrepancies.
4. 1D eta-projection profiles for quantitative channel-by-channel comparison.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add workspace directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DNN"))

from CNN.data_loader import pixelize_eflow_events
from DNN.data_loader import get_all_parquet_files, load_and_merge_parquet


def plot_average_heatmaps(
    avg_dict: dict,
    output_path: str,
    title_suffix: str = "",
    res: int = 40,
):
    """Plot multi-event average heatmaps comparing processes."""
    n_procs = len(avg_dict)
    fig, axes = plt.subplots(n_procs, 4, figsize=(18, 3.8 * n_procs))

    if n_procs == 1:
        axes = np.expand_dims(axes, 0)

    channel_titles = [
        "Avg Track pT (GeV)",
        "Avg Photon ET (GeV)",
        "Avg Neutral Hadron ET (GeV)",
        "Total Energy (Ch0+1+2)",
    ]
    cmaps = ["magma", "viridis", "cividis", "inferno"]
    extent = [-5.0, 5.0, -np.pi, np.pi]

    for row_idx, (pname, avg_img) in enumerate(avg_dict.items()):
        total_e = np.sum(avg_img, axis=0)

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            data = avg_img[col_idx] if col_idx < 3 else total_e
            vmax = max(np.percentile(data[data > 0], 99.5), 1e-2) if np.any(data > 0) else 1.0

            im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmaps[col_idx], vmin=0, vmax=vmax, aspect="auto")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("GeV / bin", fontsize=9)

            ax.set_title(f"{pname} ({title_suffix})\n{channel_titles[col_idx]}", fontsize=11)
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$\phi$")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved average heatmaps plot to {output_path}")
    plt.close()


def plot_difference_heatmaps(
    img_a: np.ndarray,
    img_b: np.ndarray,
    name_a: str,
    name_b: str,
    output_path: str,
    title_suffix: str = "",
    res: int = 40,
):
    """Plot difference maps (A - B) to highlight subtle discrepancies."""
    diff_img = img_a - img_b
    total_a = np.sum(img_a, axis=0)
    total_b = np.sum(img_b, axis=0)
    diff_total = total_a - total_b

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    channel_titles = [
        f"Diff Track pT: {name_a} - {name_b}",
        f"Diff Photon ET: {name_a} - {name_b}",
        f"Diff Neutral Hadron ET: {name_a} - {name_b}",
        f"Diff Total Energy: {name_a} - {name_b}",
    ]
    extent = [-5.0, 5.0, -np.pi, np.pi]

    for col_idx in range(4):
        ax = axes[col_idx]
        data = diff_img[col_idx] if col_idx < 3 else diff_total
        abs_max = np.percentile(np.abs(data), 99.5)
        abs_max = max(abs_max, 1e-3)

        im = ax.imshow(
            data.T,
            origin="lower",
            extent=extent,
            cmap="bwr",
            vmin=-abs_max,
            vmax=abs_max,
            aspect="auto",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Delta$ GeV / bin", fontsize=9)

        ax.set_title(f"{channel_titles[col_idx]}\n({title_suffix})", fontsize=10)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\phi$")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved difference heatmaps plot to {output_path}")
    plt.close()


def plot_eta_projection_profiles(
    avg_dict: dict,
    output_path: str,
    res: int = 40,
):
    """Integrate along phi to compare 1D eta-distribution profiles directly."""
    bins_eta = np.linspace(-5.0, 5.0, res)
    channel_names = ["Track pT (Ch 0)", "Photon ET (Ch 1)", "Neutral Hadron ET (Ch 2)", "Total Energy (Ch 0+1+2)"]
    colors = {"Pol LL (W_L W_L)": "red", "Pol LT (W_L W_T)": "green", "Pol TT (W_T W_T)": "blue"}
    linestyles = {"Pol LL (W_L W_L)": "-", "Pol LT (W_L W_T)": "--", "Pol TT (W_T W_T)": ":"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    for col_idx in range(4):
        ax = axes[col_idx]
        for pname, avg_img in avg_dict.items():
            if col_idx < 3:
                # Sum over phi (axis 2)
                eta_profile = np.sum(avg_img[col_idx], axis=1)
            else:
                total_e = np.sum(avg_img, axis=0)
                eta_profile = np.sum(total_e, axis=1)

            c = colors.get(pname, "black")
            ls = linestyles.get(pname, "-")
            ax.plot(bins_eta, eta_profile, label=pname, color=c, linestyle=ls, linewidth=2)

        ax.set_title(f"1D Profile: {channel_names[col_idx]}", fontsize=11)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel("Total Deposited Energy (GeV)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved 1D eta-profiles plot to {output_path}")
    plt.close()


def main():
    parquet_dir = "Sample/Parquet/batch_lowlevel_constituent_mg_sample"
    out_dir = "figures/event_images"
    res = 40
    n_avg = 50000  # 50,000 events

    print("Scanning Parquet files...")
    all_files = get_all_parquet_files(parquet_dir)

    target_processes = {
        "EW_Signal": "WWjj_EW",
        "QCD_Background": "WWjj_QCD",
        "Pol_LL": "WWjj_EW_LL_WW_cmf",
        "Pol_LT": "WWjj_EW_LT_WW_cmf",
        "Pol_TT": "WWjj_EW_TT_WW_cmf",
    }

    avg_by_proc = {}

    for key, pname in target_processes.items():
        if pname not in all_files:
            matched = [k for k in all_files.keys() if pname in k or k in pname]
            if matched:
                pname = matched[0]
            else:
                print(f"Warning: {pname} not found in files. Available: {list(all_files.keys())}")
                continue

        # Use up to 35 files to guarantee at least 50,000 events
        files = all_files[pname][:35]
        print(f"Loading {key} ({pname}) from {len(files)} files to gather {n_avg} events...")
        df = load_and_merge_parquet(files)
        if len(df) > n_avg:
            df = df.iloc[:n_avg]
        print(f"  Pixelizing and averaging {len(df)} events...")
        imgs = pixelize_eflow_events(df, res=res)
        avg_by_proc[key] = np.mean(imgs, axis=0)
        del df, imgs

    # 1. Plot Average Heatmaps: EW vs QCD Background (50k Events)
    avg_ew_vs_qcd = {
        "EW WWjj (Signal)": avg_by_proc["EW_Signal"],
        "QCD WWjj (Background)": avg_by_proc["QCD_Background"],
    }
    plot_average_heatmaps(
        avg_ew_vs_qcd,
        output_path=f"{out_dir}/average_ew_vs_qcd_50k.png",
        title_suffix="Avg of 50k Events",
        res=res,
    )

    # 2. Plot Average Heatmaps: Polarization LL vs LT vs TT (50k Events)
    avg_pol_3state = {
        "Pol LL (W_L W_L)": avg_by_proc["Pol_LL"],
        "Pol LT (W_L W_T)": avg_by_proc["Pol_LT"],
        "Pol TT (W_T W_T)": avg_by_proc["Pol_TT"],
    }
    plot_average_heatmaps(
        avg_pol_3state,
        output_path=f"{out_dir}/average_polarization_ll_lt_tt_50k.png",
        title_suffix="Avg of 50k Events",
        res=res,
    )

    # 3. Plot Difference Heatmap: LL - TT (50k Events)
    plot_difference_heatmaps(
        img_a=avg_pol_3state["Pol LL (W_L W_L)"],
        img_b=avg_pol_3state["Pol TT (W_T W_T)"],
        name_a="LL",
        name_b="TT",
        output_path=f"{out_dir}/diff_polarization_ll_minus_tt_50k.png",
        title_suffix="50k Events Difference",
        res=res,
    )

    # 4. Plot 1D eta-projection profiles
    plot_eta_projection_profiles(
        avg_pol_3state,
        output_path=f"{out_dir}/polarization_eta_profiles_50k.png",
        res=res,
    )

    print("All 50k high-statistics visualization plots generated successfully!")


if __name__ == "__main__":
    main()
