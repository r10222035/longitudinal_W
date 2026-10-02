"""Generate crisp, white-background standalone vector PDFs for single-event image representation."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DNN"))

from CNN.data_loader import pixelize_eflow_events


def make_white_cmap(color_hex_mid: str, color_hex_dark: str, name: str):
    """Create a colormap where 0 is pure white, transitioning to color."""
    return LinearSegmentedColormap.from_list(name, ["#ffffff", color_hex_mid, color_hex_dark])


def save_single_white_pdf(
    data: np.ndarray,
    output_path: str,
    cmap,
    cbar_label: str,
    extent: list = [-5.0, 5.0, -np.pi, np.pi],
):
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    vmax = max(np.percentile(data[data > 0], 99.5), 1e-2) if np.any(data > 0) else 1.0

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
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
    print(f"Saved {output_path}")


def main():
    import glob
    parquet_files = sorted(glob.glob("Sample/Parquet/batch_lowlevel_constituent_mg_sample/*WWjj_EW*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No parquet files found")

    df = pd.read_parquet(parquet_files[0])
    imgs = pixelize_eflow_events(df.iloc[:20], res=40)

    # Pick an event with good energy deposition in all channels
    scores = [np.count_nonzero(im[0]) + np.count_nonzero(im[1]) + np.count_nonzero(im[2]) for im in imgs]
    best_idx = int(np.argmax(scores))
    print(f"Selected event index: {best_idx} with {scores[best_idx]} active pixels")

    evt = imgs[best_idx]
    tot = np.sum(evt, axis=0)

    pdf_dir = "figures/event_images/pdf"

    cmap_trk = make_white_cmap("#2b83ba", "#08306b", "white_blue")
    cmap_pho = make_white_cmap("#4dac26", "#00441b", "white_green")
    cmap_had = make_white_cmap("#fdae61", "#a63603", "white_orange")
    cmap_tot = make_white_cmap("#9970ab", "#40004b", "white_purple")

    save_single_white_pdf(evt[0], f"{pdf_dir}/single_event_track.pdf", cmap_trk, r"Track $p_{\mathrm{T}}$ [GeV]")
    save_single_white_pdf(evt[1], f"{pdf_dir}/single_event_photon.pdf", cmap_pho, r"Photon $E_{\mathrm{T}}$ [GeV]")
    save_single_white_pdf(evt[2], f"{pdf_dir}/single_event_hadron.pdf", cmap_had, r"Neutral Hadron $E_{\mathrm{T}}$ [GeV]")
    save_single_white_pdf(tot, f"{pdf_dir}/single_event_total.pdf", cmap_tot, r"Total Energy [GeV]")

    # Remove composite if exists
    comp_path = Path(f"{pdf_dir}/single_event_composite.pdf")
    if comp_path.exists():
        comp_path.unlink()
        print(f"Removed old composite image: {comp_path}")


if __name__ == "__main__":
    main()
