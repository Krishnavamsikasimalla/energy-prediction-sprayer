#!/usr/bin/env python3
"""
plot_track.py — Plot GNSS routes from training_dataset.csv or raw MF4 files.
Colours the track by energy consumption, speed, or spray status.

Usage:
  python scripts/plot_track.py                          # from processed CSV
  python scripts/plot_track.py --source data/raw/       # decode from MF4
  python scripts/plot_track.py --color speed            # colour by speed
  python scripts/plot_track.py --color spray            # show spray zones
  python scripts/plot_track.py --file can0_20260302_1215.mf4  # single file
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("data/processed/training_dataset.csv")
OUTPUT_DIR   = Path("outputs/plots")


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two GPS points."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi        = np.radians(lat2 - lat1)
    dlam        = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def load_from_csv(path: Path, source_file: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Filter to a specific source file if requested
    if source_file and "Source_File" in df.columns:
        df = df[df["Source_File"] == source_file]

    # Drop rows without GNSS
    for col in ["Start_Lat", "Start_Lon", "End_Lat", "End_Lon"]:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found. Rebuild dataset with updated physics.py.")
            sys.exit(1)

    df = df.dropna(subset=["Start_Lat", "Start_Lon"])
    df = df[(df["Start_Lat"] != 0) & (df["Start_Lon"] != 0)]

    if df.empty:
        logger.error("No valid GNSS data in dataset. Check that GNSS signals are decoded.")
        sys.exit(1)

    return df


def plot_track(df: pd.DataFrame, color_by: str = "energy", title: str = "Mission Track"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lats = df["Start_Lat"].values
    lons = df["Start_Lon"].values

    # ── Colour mapping ────────────────────────────────────────────────────────
    if color_by == "energy" and "Energy_per_meter" in df.columns:
        values   = df["Energy_per_meter"].fillna(0).values
        label    = "Energy/meter (Wh/m)"
        cmap     = "RdYlGn_r"
        norm     = mcolors.Normalize(vmin=np.nanpercentile(values, 5),
                                     vmax=np.nanpercentile(values, 95))
        colors   = cm.get_cmap(cmap)(norm(values))
        use_cbar = True

    elif color_by == "speed" and "Avg_Speed" in df.columns:
        values   = df["Avg_Speed"].fillna(0).values
        label    = "Avg Speed (km/h)"
        cmap     = "plasma"
        norm     = mcolors.Normalize(vmin=0, vmax=values.max())
        colors   = cm.get_cmap(cmap)(norm(values))
        use_cbar = True

    elif color_by == "spray" and "Spray_Ratio" in df.columns:
        spraying = df["Spray_Ratio"].fillna(0).values > 0.5
        colors   = np.where(spraying[:, None],
                            [[0.1, 0.6, 0.1, 1.0]],   # green = spraying
                            [[0.7, 0.7, 0.7, 0.6]])    # grey  = not spraying
        label    = "Spray Active"
        use_cbar = False

    elif color_by == "fault" and "Window_Healthy" in df.columns:
        healthy = df["Window_Healthy"].fillna(1).values == 1
        colors  = np.where(healthy[:, None],
                           [[0.2, 0.6, 0.9, 1.0]],   # blue  = healthy
                           [[0.9, 0.2, 0.2, 1.0]])    # red   = fault
        label    = "Fault Status"
        use_cbar = False

    else:
        colors   = ["steelblue"] * len(lats)
        label    = ""
        use_cbar = False

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw track segments
    for i in range(len(lats) - 1):
        ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]],
                color=colors[i] if isinstance(colors, np.ndarray) else colors[i],
                linewidth=2.5, solid_capstyle="round")

    # Scatter the window start points
    sc = ax.scatter(lons, lats, c=colors if not use_cbar else values,
                    cmap=cmap if use_cbar else None,
                    norm=norm if use_cbar else None,
                    s=40, zorder=5, edgecolors="white", linewidths=0.4)

    # Start / End markers
    ax.scatter(lons[0],  lats[0],  s=150, color="lime",   marker="^",
               zorder=10, label="Start", edgecolors="black", linewidths=0.8)
    ax.scatter(lons[-1], lats[-1], s=150, color="red",    marker="s",
               zorder=10, label="End",   edgecolors="black", linewidths=0.8)

    # Colourbar
    if use_cbar:
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.7)
        cbar.set_label(label, fontsize=11)
    elif color_by == "spray":
        ax.legend(handles=[
            Patch(color=(0.1, 0.6, 0.1), label="Spraying"),
            Patch(color=(0.7, 0.7, 0.7), label="Not spraying"),
        ], loc="upper right")
    elif color_by == "fault":
        ax.legend(handles=[
            Patch(color=(0.2, 0.6, 0.9), label="Healthy window"),
            Patch(color=(0.9, 0.2, 0.2), label="Fault detected"),
        ], loc="upper right")

    ax.legend(loc="lower right")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude",  fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Stats panel ───────────────────────────────────────────────────────────
    n_windows    = len(df)
    total_dist   = df["Distance_m"].sum() if "Distance_m" in df.columns else 0
    total_energy = df["Energy_Wh"].sum()  if "Energy_Wh"  in df.columns else 0
    area_acres   = df["Area_acres"].sum() if "Area_acres"  in df.columns else 0
    n_faults     = int((df["Total_Faults"] > 0).sum()) if "Total_Faults" in df.columns else 0

    stats_text = (
        f"Windows:       {n_windows}\n"
        f"Total dist:    {total_dist:,.0f} m\n"
        f"Total energy:  {total_energy:,.1f} Wh\n"
        f"Area sprayed:  {area_acres:.2f} acres\n"
        f"Fault windows: {n_faults}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    plt.tight_layout()
    out = OUTPUT_DIR / f"track_{color_by}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Track plot saved → {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot GNSS mission tracks.")
    parser.add_argument("--source", type=str, default=str(DATASET_PATH),
                        help="Path to training_dataset.csv or raw/ directory")
    parser.add_argument("--file",   type=str, default=None,
                        help="Filter to a specific source MF4 filename")
    parser.add_argument("--color",  type=str, default="energy",
                        choices=["energy", "speed", "spray", "fault"],
                        help="What to colour the track by")
    parser.add_argument("--all",    action="store_true",
                        help="Generate all 4 colour modes")
    args = parser.parse_args()

    source = Path(args.source)

    if source.is_dir():
        logger.error("For raw MF4 directories, run build_dataset.py first, then use the CSV.")
        sys.exit(1)

    df = load_from_csv(source, source_file=args.file)
    logger.info(f"Loaded {len(df)} windows from {source.name}")

    title_suffix = f" — {args.file}" if args.file else ""

    if args.all:
        for mode in ["energy", "speed", "spray", "fault"]:
            plot_track(df, color_by=mode, title=f"Mission Track{title_suffix}")
    else:
        plot_track(df, color_by=args.color, title=f"Mission Track{title_suffix}")


if __name__ == "__main__":
    main()