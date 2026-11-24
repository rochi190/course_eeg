#!/usr/bin/env python3
"""Generate 300 DPI publication-grade V2 figures for SPEC A-V2."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V2_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = V2_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_fig4_ablation(df_ab: pd.DataFrame) -> None:
    """Grouped bar chart comparing A1, A2, A3-noLP, A3 across 4 disentangled tracks."""
    ov = df_ab[df_ab["snr_db"] == "Overall"].copy()
    tracks = ["Test-Pure-LP45", "Test-Pure-Burst", "Test-ExpDecay", "Test-RC1st"]
    track_labels = [
        "Pure LP45\n(Butterworth 4th, filtfilt)",
        "Pure Burst\n(0.3-1.5s Unfiltered)",
        "ExpDecay\n(exp(-0.015f))",
        "RC 1st-Order (filtfilt)\n(fc=40 Hz, -40 dB/dec)",
    ]
    exps = [
        ("Exp A1", "#d62728", "Exp A1: 1D-ResCNN (ID)"),
        ("Exp A2", "#ff7f0e", "Exp A2: 1D-ResUNet (ID)"),
        ("Exp A3-noLP", "#9467bd", "Exp A3-noLP (Burst+Clean, No LP)"),
        ("Exp A3", "#1f77b4", "Exp A3: Physics-ResUNet (Full)"),
    ]

    x = np.arange(len(tracks))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.2, 6.8))

    max_v = 0.0
    for i, (exp_key, color, label) in enumerate(exps):
        vals = [
            float(ov[(ov["track"] == trk) & (ov["experiment"] == exp_key)]["rrmse_t"].iloc[0])
            for trk in tracks
        ]
        max_v = max(max_v, max(vals))
        bars = ax.bar(
            x + (i - 1.5) * width, vals, width=width,
            color=color, edgecolor="black", linewidth=0.7, label=label,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 0.014,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(track_labels, fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Overall Mean RRMSE_t (lower is better)")
    ax.set_title(
        "Disentangled Lowpass vs. Burst Ablation & Cross-Filter Generalization",
        fontweight="bold", pad=16,
    )
    ax.set_ylim(0, max_v * 1.26)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout()
    out_path = FIG_DIR / "fig4_disentangle_ablation.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_fig5_zeroshot(cache_path: Path, zs_csv: Path) -> None:
    """2-panel Welch PSD on real uncleaned clinical EEG (High-EMG & Occipital Alpha)."""
    d = np.load(cache_path)
    df_zs = pd.read_csv(zs_csv).set_index("experiment")
    freqs = d["freqs"]
    mask = (freqs >= 1.0) & (freqs <= 80.0)
    fm = freqs[mask]
    to_db = lambda p: 10.0 * np.log10(np.maximum(p[mask], 1e-12))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 6.2))

    # Left: High-EMG frontal/temporal segments
    ax1.axvspan(30, 80, color="#e74c3c", alpha=0.10, label="EMG Band (30-80 Hz)")
    ax1.plot(fm, to_db(d["psd_emg_Raw"]), color="#111111", linewidth=2.0, label="Raw Clinical EEG (Top-20 High-EMG)")
    ax1.plot(fm, to_db(d["psd_emg_Exp A1"]), color="#d62728", linewidth=1.7, linestyle="--", label="Exp A1 (1D-ResCNN)")
    ax1.plot(fm, to_db(d["psd_emg_Exp A3"]), color="#1f77b4", linewidth=2.1, label="Exp A3 (Physics-ResUNet)")

    drop_a1 = float(df_zs.loc["Exp A1", "high_emg_30_80hz_drop_db_mean"])
    drop_a3 = float(df_zs.loc["Exp A3", "high_emg_30_80hz_drop_db_mean"])
    ax1.text(
        0.04, 0.06,
        f"30-80 Hz Suppression:\n  Exp A1: {drop_a1:.2f} dB\n  Exp A3: {drop_a3:.2f} dB",
        transform=ax1.transAxes, fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#7f8c8d", alpha=0.92),
    )
    ax1.set_title("Zero-Shot High-EMG Real Segments (Frontal/Temporal)", fontweight="bold")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Mean PSD (dB)")
    ax1.set_xlim(1, 80)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # Right: Occipital eyes-closed Alpha segments
    ax2.axvspan(8, 13, color="#2ecc71", alpha=0.14, label="Alpha Band (8-13 Hz)")
    ax2.plot(fm, to_db(d["psd_occ_Raw"]), color="#111111", linewidth=2.0, label="Raw Clinical EEG (sub-037 O1/O2)")
    ax2.plot(fm, to_db(d["psd_occ_Exp A1"]), color="#d62728", linewidth=1.7, linestyle="--", label="Exp A1 (1D-ResCNN)")
    ax2.plot(fm, to_db(d["psd_occ_Exp A3"]), color="#1f77b4", linewidth=2.1, label="Exp A3 (Physics-ResUNet)")

    al_a1 = float(df_zs.loc["Exp A1", "occipital_alpha_power_ratio"])
    al_a3 = float(df_zs.loc["Exp A3", "occipital_alpha_power_ratio"])
    paf_a1 = float(df_zs.loc["Exp A1", "occipital_paf_shift_hz_mean"])
    paf_a3 = float(df_zs.loc["Exp A3", "occipital_paf_shift_hz_mean"])
    ax2.text(
        0.04, 0.06,
        f"Alpha Power Ratio / PAF Shift:\n  Exp A1: {al_a1:.3f} ({paf_a1:.3f} Hz)\n  Exp A3: {al_a3:.3f} ({paf_a3:.3f} Hz)",
        transform=ax2.transAxes, fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#7f8c8d", alpha=0.92),
    )
    ax2.set_title("Zero-Shot Occipital Eyes-Closed (Alpha Preservation)", fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Mean PSD (dB)")
    ax2.set_xlim(1, 80)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)

    fig.tight_layout()
    out_path = FIG_DIR / "fig5_real_eeg_zeroshot.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df_ab = pd.read_csv(OUT_DIR / "ablation_metrics_v2.csv")
    plot_fig4_ablation(df_ab)
    plot_fig5_zeroshot(OUT_DIR / "real_eeg_zeroshot_cache.npz", OUT_DIR / "zeroshot_metrics_v2.csv")


if __name__ == "__main__":
    main()
