"""High-statistics event image plotting tool.

Generates standalone vector PDF figures for LaTeX inclusion via \subfloat:
1. Single-event individual channel images (Track, Photon, Hadron, Composite).
2. EW vs. QCD background average heatmaps (140,000 events, 4 channels each).
3. Polarization states (LL, LT, TT) average heatmaps (200,000 events, 4 channels each).
4. Difference heatmaps (LL - TT, 4 channels).
5. 1D eta-projection profiles (4 channels, comparing LL, LT, TT).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add workspace directory and DNN directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DNN"))

from CNN.data_loader import pixelize_eflow_events
from DNN.data_loader import get_all_parquet_files, load_and_merge_parquet


def save_single_heatmap_pdf(
    data: np.ndarray,
    output_path: str,
    cmap: str,
    cbar_label: str,
    vmin: float = 0.0,
    vmax: float = None,
    extent: list = [-5.0, 5.0, -np.pi, np.pi],
):
    """Save an individual 2D heatmap as a standalone PDF for LaTeX subfloat."""
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    if vmax is None:
        vmax = max(np.percentile(data[data > 0], 99.5), 1e-2) if np.any(data > 0) else 1.0

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_xlabel(r"$\eta$", fontsize=11)
    ax.set_ylabel(r"$\phi$", fontsize=11)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_rgb_composite_pdf(
    img_3ch: np.ndarray,
    output_path: str,
    extent: list = [-5.0, 5.0, -np.pi, np.pi],
):
    """Save an individual RGB composite image as a standalone PDF."""
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    rgb = np.zeros((40, 40, 3))
    for c in range(3):
        ch_data = img_3ch[c]
        vmax = np.percentile(ch_data[ch_data > 0], 99) if np.any(ch_data > 0) else 1.0
        rgb[:, :, c] = np.clip(ch_data / max(vmax, 1e-3), 0, 1)

    rgb = np.transpose(rgb, (1, 0, 2))
    ax.imshow(rgb, origin="lower", extent=extent, aspect="auto")
    ax.set_xlabel(r"$\eta$", fontsize=11)
    ax.set_ylabel(r"$\phi$", fontsize=11)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_single_profile_pdf(
    bins_eta: np.ndarray,
    profiles_dict: dict,
    output_path: str,
    ylabel: str = r"Deposited Energy [GeV]",
):
    """Save an individual 1D eta-profile as a standalone PDF."""
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    colors = {"Pol LL ($W_L W_L$)": "#d7191c", "Pol LT ($W_L W_T$)": "#2ca25f", "Pol TT ($W_T W_T$)": "#2b83ba"}
    linestyles = {"Pol LL ($W_L W_T$)": "--", "Pol LL ($W_L W_L$)": "-", "Pol TT ($W_T W_T$)": ":"}

    for name, prof in profiles_dict.items():
        c = colors.get(name, "black")
        ls = linestyles.get(name, "-")
        ax.plot(bins_eta, prof, label=name, color=c, linestyle=ls, linewidth=2)

    ax.set_xlabel(r"$\eta$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="Force recomputation from parquet")
    args = parser.parse_args()

    parquet_dir = "Sample/Parquet/batch_lowlevel_constituent_mg_sample"
    pdf_dir = "figures/event_images/pdf"
    cache_file = Path("figures/event_images/cached_event_images.npz")
    res = 40
    bins_eta = np.linspace(-5.0, 5.0, res)

    cmaps = ["magma", "viridis", "cividis", "inferno"]
    cbar_labels = [
        r"Avg Track $p_{\mathrm{T}}$ [GeV]",
        r"Avg Photon $E_{\mathrm{T}}$ [GeV]",
        r"Avg Neutral Hadron $E_{\mathrm{T}}$ [GeV]",
        r"Avg Total Energy [GeV]",
    ]

    target_configs = {
        "EW_Signal": {"key": "WWjj_EW", "n_events": 140000},
        "QCD_Background": {"key": "WWjj_QCD", "n_events": 140000},
        "Pol_LL": {"key": "WWjj_EW_LL_WW_cmf", "n_events": 200000},
        "Pol_LT": {"key": "WWjj_EW_LT_WW_cmf", "n_events": 200000},
        "Pol_TT": {"key": "WWjj_EW_TT_WW_cmf", "n_events": 200000},
    }

    avg_images = {}

    if cache_file.exists() and not args.recompute:
        print(f"Loading cached average images from {cache_file}...")
        data = np.load(cache_file)
        bins_eta = data["bins_eta"]
        for k in target_configs.keys():
            avg_images[k] = data[k]
    else:
        print("Scanning Parquet files...")
        all_files = get_all_parquet_files(parquet_dir)

        # Process each target process
        for proc_name, cfg in target_configs.items():
            target_key = cfg["key"]
            n_target = cfg["n_events"]

            if target_key in all_files:
                files = all_files[target_key]
            else:
                matched = [k for k in all_files.keys() if k == target_key]
                if not matched:
                    raise KeyError(f"Exact key '{target_key}' not found in all_files: {list(all_files.keys())}")
                files = all_files[matched[0]]

            print(f"\nProcessing {proc_name} (key: {target_key}, target: {n_target} events)...")
            loaded_dfs = []
            total_evts = 0
            for f in files:
                df_part = pd.read_parquet(f)
                loaded_dfs.append(df_part)
                total_evts += len(df_part)
                if total_evts >= n_target:
                    break

            df = pd.concat(loaded_dfs, ignore_index=True)
            if len(df) > n_target:
                df = df.iloc[:n_target]
            print(f"  Pixelizing {len(df)} events...")
            imgs = pixelize_eflow_events(df, res=res)
            avg_img = np.mean(imgs, axis=0)  # shape (3, 40, 40)
            total_energy = np.sum(avg_img, axis=0, keepdims=True)  # shape (1, 40, 40)
            full_avg = np.concatenate([avg_img, total_energy], axis=0)  # shape (4, 40, 40)
            avg_images[proc_name] = full_avg

            # Generate single-event gallery from first event of EW_Signal
            if proc_name == "EW_Signal":
                print("  Generating standalone single-event PDFs...")
                single_evt = imgs[0]  # shape (3, 40, 40)
                save_single_heatmap_pdf(single_evt[0], f"{pdf_dir}/single_event_track.pdf", "magma", r"Track $p_{\mathrm{T}}$ [GeV]")
                save_single_heatmap_pdf(single_evt[1], f"{pdf_dir}/single_event_photon.pdf", "viridis", r"Photon $E_{\mathrm{T}}$ [GeV]")
                save_single_heatmap_pdf(single_evt[2], f"{pdf_dir}/single_event_hadron.pdf", "cividis", r"Neutral Hadron $E_{\mathrm{T}}$ [GeV]")

            del df, imgs, loaded_dfs

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, bins_eta=bins_eta, **avg_images)
        print(f"\nSaved image cache to {cache_file}")

    # 1. Output EW vs QCD Standalone PDFs (8 PDFs)
    print("\nGenerating EW vs. QCD standalone PDFs (140,000 events)...")
    proc_prefixes = {
        "EW_Signal": "ew_140k",
        "QCD_Background": "qcd_140k",
    }
    for proc_name, prefix in proc_prefixes.items():
        img_4ch = avg_images[proc_name]
        ch_names = ["track", "photon", "hadron", "total"]
        for c in range(4):
            save_single_heatmap_pdf(
                img_4ch[c],
                f"{pdf_dir}/{prefix}_{ch_names[c]}.pdf",
                cmaps[c],
                cbar_labels[c],
            )

    # 2. Output Polarization (LL, LT, TT) Standalone PDFs (12 PDFs)
    print("\nGenerating Polarization standalone PDFs (200,000 events)...")
    pol_prefixes = {
        "Pol_LL": "pol_ll_200k",
        "Pol_LT": "pol_lt_200k",
        "Pol_TT": "pol_tt_200k",
    }
    for proc_name, prefix in pol_prefixes.items():
        img_4ch = avg_images[proc_name]
        ch_names = ["track", "photon", "hadron", "total"]
        for c in range(4):
            save_single_heatmap_pdf(
                img_4ch[c],
                f"{pdf_dir}/{prefix}_{ch_names[c]}.pdf",
                cmaps[c],
                cbar_labels[c],
            )

    # 3. Output Difference Heatmaps (LL - TT, 4 PDFs)
    print("\nGenerating Difference Heatmap PDFs (LL - TT, 200,000 events)...")
    diff_img_4ch = avg_images["Pol_LL"] - avg_images["Pol_TT"]
    ch_names = ["track", "photon", "hadron", "total"]
    diff_labels = [
        r"$\Delta$ Track $p_{\mathrm{T}}$ [GeV]",
        r"$\Delta$ Photon $E_{\mathrm{T}}$ [GeV]",
        r"$\Delta$ Neutral Hadron $E_{\mathrm{T}}$ [GeV]",
        r"$\Delta$ Total Energy [GeV]",
    ]
    for c in range(4):
        data = diff_img_4ch[c]
        abs_max = max(np.percentile(np.abs(data), 99.5), 1e-3)
        save_single_heatmap_pdf(
            data,
            f"{pdf_dir}/diff_ll_tt_200k_{ch_names[c]}.pdf",
            cmap="bwr",
            cbar_label=diff_labels[c],
            vmin=-abs_max,
            vmax=abs_max,
        )

    # 4. Output 1D eta-projection profiles (4 PDFs)
    print("\nGenerating 1D eta-profile PDFs (200,000 events)...")
    pol_display_names = {
        "Pol_LL": "Pol LL ($W_L W_L$)",
        "Pol_LT": "Pol LT ($W_L W_T$)",
        "Pol_TT": "Pol TT ($W_T W_T$)",
    }
    for c in range(4):
        profiles = {}
        for proc_name, disp_name in pol_display_names.items():
            # Integrate over phi (axis 1)
            prof_1d = np.sum(avg_images[proc_name][c], axis=1)
            profiles[disp_name] = prof_1d

        save_single_profile_pdf(
            bins_eta,
            profiles,
            f"{pdf_dir}/profile_200k_{ch_names[c]}.pdf",
            ylabel=f"Total Deposited {cbar_labels[c].replace('Avg ', '')}",
        )

    print("\nAll 32 standalone vector PDF figures generated successfully in:", pdf_dir)


if __name__ == "__main__":
    main()

