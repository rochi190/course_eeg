#!/usr/bin/env python3
"""Generate 300 DPI presentation figures for SPEC A."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

PROJ_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "Exp A0": "#7f8c8d",
    "Exp A1": "#d62728",
    "Exp A2": "#ff7f0e",
    "Exp A3": "#1f77b4",
}
LABELS = {
    "Exp A0": "Exp A0: Bandpass (1-40 Hz)",
    "Exp A1": "Exp A1: 1D-ResCNN (ID)",
    "Exp A2": "Exp A2: 1D-ResUNet (ID + MR-STFT)",
    "Exp A3": "Exp A3: 1D-ResUNet (Physics + MR-STFT)",
}
MARKERS = {"Exp A0": "s", "Exp A1": "^", "Exp A2": "D", "Exp A3": "o"}


def plot_fig1(df: pd.DataFrame) -> None:
    """Fig 1: Test-ID vs Test-OOD (fc=45 Hz) RRMSE_t and CC across SNRs."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    tracks = [("Test-ID", "Test-ID (Standard Linear Mixing)"), ("Test-OOD-45", "Test-OOD (fc = 45 Hz + Burst)")]
    exps = ["Exp A0", "Exp A1", "Exp A2", "Exp A3"]

    for col, (trk, title) in enumerate(tracks):
        sub_t = df[(df["track"] == trk) & (df["snr_db"] != "Overall")].copy()
        sub_t["snr_num"] = sub_t["snr_db"].astype(int)
        sub_t = sub_t.sort_values("snr_num")

        ax_r = axes[0, col]
        ax_c = axes[1, col]
        for exp in exps:
            se = sub_t[sub_t["experiment"] == exp]
            lw = 2.6 if exp == "Exp A3" else 2.0
            ax_r.plot(
                se["snr_num"], se["rrmse_t"],
                color=COLORS[exp], marker=MARKERS[exp], linewidth=lw, markersize=6,
                label=LABELS[exp],
            )
            ax_c.plot(
                se["snr_num"], se["cc"],
                color=COLORS[exp], marker=MARKERS[exp], linewidth=lw, markersize=6,
                label=LABELS[exp],
            )

        ax_r.set_title(title, fontweight="bold")
        ax_r.grid(True, linestyle="--", alpha=0.35)
        ax_c.grid(True, linestyle="--", alpha=0.35)
        ax_c.set_xlabel("Input SNR (dB)")
        ax_c.set_xticks(range(-7, 3))

    axes[0, 0].set_ylabel("Temporal RRMSE (lower is better)")
    axes[1, 0].set_ylabel("Pearson Correlation CC (higher is better)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.995)
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = FIG_DIR / "fig1_snr_rrmse.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_fig2(sample_path: Path) -> None:
    """Fig 2: Time-domain waveform & Welch PSD (1-80 Hz) on SNR=-5 dB Test-OOD sample."""
    d = np.load(sample_path)
    noisy = d["noisy"]
    clean = d["clean"]
    pred_a1 = d["pred_a1"]
    pred_a3 = d["pred_a3"]
    r_a1 = float(d["rrmse_a1"])
    r_a3 = float(d["rrmse_a3"])
    c_a1 = float(d["cc_a1"])
    c_a3 = float(d["cc_a3"])

    t = np.arange(len(clean)) / 512.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.0))

    # Top: time-domain waveform
    ax1.plot(t, noisy, color="#bdc3c7", linewidth=1.0, alpha=0.85, label="Noisy Input (Test-OOD fc=45 Hz, -5 dB)")
    ax1.plot(t, clean, color="#111111", linewidth=1.6, label="Ground Truth Clean EEG")
    ax1.plot(t, pred_a1, color="#d62728", linewidth=1.3, linestyle="--", label=f"Exp A1 ResCNN (RRMSE={r_a1:.3f}, CC={c_a1:.3f})")
    ax1.plot(t, pred_a3, color="#1f77b4", linewidth=1.5, label=f"Exp A3 Physics-ResUNet (RRMSE={r_a3:.3f}, CC={c_a3:.3f})")
    ax1.set_title("Time-Domain Reconstruction on OOD Scalp-Attenuated Burst EMG (SNR = -5 dB)", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.set_xlim(0, 2.0)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", ncol=2, fontsize=10.5)

    # Bottom: Welch PSD (1-80 Hz)
    freqs, p_noisy = welch(noisy.astype(np.float64), fs=512, nperseg=256)
    _, p_clean = welch(clean.astype(np.float64), fs=512, nperseg=256)
    _, p_a1 = welch(pred_a1.astype(np.float64), fs=512, nperseg=256)
    _, p_a3 = welch(pred_a3.astype(np.float64), fs=512, nperseg=256)

    mask = (freqs >= 1.0) & (freqs <= 80.0)
    f_m = freqs[mask]
    to_db = lambda p: 10.0 * np.log10(np.maximum(p[mask], 1e-12))

    ax2.axvspan(8, 13, color="#2ecc71", alpha=0.14, label="Alpha Band (8-13 Hz)")
    ax2.axvspan(13, 30, color="#9b59b6", alpha=0.12, label="Beta Band (13-30 Hz)")
    ax2.plot(f_m, to_db(p_noisy), color="#95a5a6", linewidth=1.3, label="Noisy Input")
    ax2.plot(f_m, to_db(p_clean), color="#111111", linewidth=1.8, label="Ground Truth EEG")
    ax2.plot(f_m, to_db(p_a1), color="#d62728", linewidth=1.5, linestyle="--", label="Exp A1 (1D-ResCNN)")
    ax2.plot(f_m, to_db(p_a3), color="#1f77b4", linewidth=1.8, label="Exp A3 (Physics-ResUNet)")

    ax2.set_title("Welch Power Spectral Density (1-80 Hz) Showing Alpha & Beta Preservation", fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (dB)")
    ax2.set_xlim(1, 80)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", ncol=3, fontsize=10.5)

    fig.tight_layout()
    out_path = FIG_DIR / "fig2_waveform_psd.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_fig3(df: pd.DataFrame) -> None:
    """Fig 3: Grouped bar chart of Test-OOD-60, Test-OOD-45, and Test-Clean RRMSE_t."""
    ov = df[df["snr_db"] == "Overall"].copy()
    tracks = ["Test-OOD-60", "Test-OOD-45", "Test-Clean"]
    track_labels = [
        "Test-OOD\n(fc = 60 Hz)",
        "Test-OOD\n(fc = 45 Hz)",
        "Test-Clean\n(Over-denoising)",
    ]
    exps = ["Exp A1", "Exp A2", "Exp A3"]

    x = np.arange(len(tracks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    max_v = 0.0
    for idx, exp in enumerate(exps):
        vals = [
            float(ov[(ov["track"] == trk) & (ov["experiment"] == exp)]["rrmse_t"].iloc[0])
            for trk in tracks
        ]
        max_v = max(max_v, max(vals))
        bars = ax.bar(
            x + (idx - 1) * width, vals, width=width,
            color=COLORS[exp], edgecolor="black", linewidth=0.7,
            label=LABELS[exp],
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_v * 0.015,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(track_labels, fontsize=12)
    ax.set_ylabel("Overall Mean RRMSE_t (lower is better)")
    ax.set_title(
        "Volume-Conduction OOD Robustness & Clean EEG Fidelity (Exp A1 vs. A2 vs. A3)",
        fontweight="bold", pad=14,
    )
    ax.set_ylim(0, max_v * 1.18)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out_path = FIG_DIR / "fig3_clean_preservation.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DIR / "metrics_summary.csv")
    plot_fig1(df)
    plot_fig2(OUT_DIR / "fig2_sample.npz")
    plot_fig3(df)


if __name__ == "__main__":
    main()
