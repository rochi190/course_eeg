#!/usr/bin/env python3
"""Generate 3 publication-grade academic diagrams for Slides 2, 4, and 5 (300 DPI)."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

ASSET_FIG_DIR = Path("/home/neil/course_eeg/ppt_assets_proj_a/figures")
ASSET_FIG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1B365D"
BLUE = "#1F77B4"
LIGHT_BLUE = "#EBF5FB"
RED = "#D62728"
LIGHT_RED = "#FDEDEC"
ORANGE = "#E67E22"
LIGHT_ORANGE = "#FEF5E7"
PURPLE = "#8E44AD"
LIGHT_PURPLE = "#F5EEF8"
GREEN = "#27AE60"
LIGHT_GREEN = "#EAFAF1"
SLATE = "#4A5568"
LIGHT_GRAY = "#F4F6F8"

plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans"],
        "axes.edgecolor": "#CBD5E0",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.14,
    }
)


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.02):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.010,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=SLATE, lw=1.8, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=13,
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}", zorder=4,
    )
    ax.add_patch(arr)
    return arr


def make_slide2_wearable_overlap() -> None:
    """Slide 2 diagram: Left = 64-Ch ICA vs 1-4 Ch Wearable; Right = EEG/EMG Spectral Overlap."""
    fig = plt.figure(figsize=(14.4, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.16)

    # ---- Left Panel: Channel Regime Comparison ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A. Why Multi-Channel ICA Fails on Wearables", fontsize=12.8, fontweight="bold", color=NAVY, pad=10)

    # Top card: 64-Ch Lab EEG
    box(ax, 0.03, 0.54, 0.94, 0.41, LIGHT_GREEN, GREEN, lw=1.5)
    ax.text(0.06, 0.885, "Hospital / Lab EEG (32–64 Channels)", fontsize=11.3, fontweight="bold", color="#1E8449", va="center")
    cx, cy, rad = 0.19, 0.70, 0.115
    ax.add_patch(Circle((cx, cy), rad, fc="white", ec=GREEN, lw=1.5, zorder=3))
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ax.add_patch(Circle((cx + 0.078 * np.cos(a), cy + 0.078 * np.sin(a)), 0.011, fc=GREEN, ec="none", zorder=4))
    for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        ax.add_patch(Circle((cx + 0.038 * np.cos(a), cy + 0.038 * np.sin(a)), 0.011, fc=GREEN, ec="none", zorder=4))
    ax.add_patch(Circle((cx, cy), 0.011, fc=GREEN, ec="none", zorder=4))
    ax.text(0.36, 0.785, "Sensors (M) >= Sources (N)", fontsize=10.3, fontweight="bold", color="#1E8449", va="center")
    ax.text(0.36, 0.695, "Spatial ICA cleanly separates\nbrain rhythms from facial EMG.", fontsize=9.6, color="#2C3E50", va="center")
    ax.text(0.36, 0.585, "Status: Mature Solution", fontsize=9.5, fontweight="bold", color=GREEN, va="center")

    # Bottom card: 1-4 Ch Wearable
    box(ax, 0.03, 0.05, 0.94, 0.43, LIGHT_RED, RED, lw=1.6)
    ax.text(0.06, 0.425, "Wearable / Ear-EEG / Headband (1–4 Channels)", fontsize=11.3, fontweight="bold", color="#922B21", va="center")
    cx2, cy2 = 0.19, 0.225
    ax.add_patch(Circle((cx2, cy2), rad, fc="white", ec=RED, lw=1.5, zorder=3))
    for ex, ey in [(cx2 - 0.055, cy2 + 0.065), (cx2 + 0.055, cy2 + 0.065)]:
        ax.add_patch(Circle((ex, ey), 0.016, fc=RED, ec="white", lw=0.8, zorder=4))
    ax.text(0.36, 0.34, "Sensors (M) << Sources (N)", fontsize=10.3, fontweight="bold", color=RED, va="center")
    ax.text(0.36, 0.225, "Mathematically underdetermined!\nICA collapses -> Need single-channel\ndeep neural denoising.", fontsize=9.6, color="#2C3E50", va="center")
    ax.text(0.36, 0.095, "Target Pain Point of Project A", fontsize=9.5, fontweight="bold", color=RED, va="center")

    # ---- Right Panel: Frequency Band Overlap Schematic (Log-spaced visual x for clarity!) ----
    ax2 = fig.add_subplot(gs[0, 1])
    # Map frequency to smooth perceptual scale: x = log10(f)
    ticks_hz = [1, 4, 8, 13, 30, 45, 80, 150]
    fx = lambda hz: np.log10(hz)
    ax2.set_xlim(fx(0.9), fx(175))
    ax2.set_ylim(0, 10.5)
    ax2.set_xticks([fx(h) for h in ticks_hz])
    ax2.set_xticklabels([str(h) for h in ticks_hz], fontsize=10.2)
    ax2.set_xlabel("Frequency (Hz, log scale for band clarity)", fontsize=11.0, fontweight="bold", color=NAVY)
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("B. Spectral Overlap: Why Fixed Bandpass Filters Fail", fontsize=12.8, fontweight="bold", color=NAVY, pad=10)

    # EEG physiological bands (now nicely spaced on log axis!)
    bands = [
        (1, 4, "Delta\n1–4 Hz", "#D6EAF8", "#2E86C1"),
        (4, 8, "Theta\n4–8 Hz", "#D1F2EB", "#17A589"),
        (8, 13, "Alpha\n8–13 Hz", "#D4EFDF", "#1E8449"),
        (13, 30, "Beta\n13–30 Hz", "#FCF3CF", "#B7950B"),
        (30, 80, "Gamma\n30–80 Hz", "#FDEBD0", "#CA6F1E"),
    ]
    ax2.text(fx(1.0), 9.0, "Clean Physiological EEG Bands (1–80 Hz)", fontsize=10.5, fontweight="bold", color=NAVY)
    for f0, f1, lbl, fc, ec in bands:
        x0, x1 = fx(f0) + 0.012, fx(f1) - 0.012
        rect = FancyBboxPatch((x0, 6.4), x1 - x0, 2.0, boxstyle="round,pad=0.01,rounding_size=0.04",
                              fc=fc, ec=ec, lw=1.3, zorder=2)
        ax2.add_patch(rect)
        ax2.text((x0 + x1) / 2, 7.4, lbl, ha="center", va="center", fontsize=8.3 if "Alpha" in lbl else 9.2, fontweight="bold", color=ec, zorder=3)

    # Broadband EMG bar
    ax2.text(fx(15.0), 5.3, "Facial Muscle Artifact (EMG: Clenching / Frowning)", fontsize=10.5, fontweight="bold", color="#922B21")
    emg_rect = FancyBboxPatch((fx(15), 3.2), fx(165) - fx(15), 1.7, boxstyle="round,pad=0.01,rounding_size=0.04",
                              fc=LIGHT_RED, ec=RED, lw=1.6, zorder=2)
    ax2.add_patch(emg_rect)
    ax2.text((fx(15) + fx(165)) / 2, 4.05, "Broadband EMG (15–200+ Hz)",
             ha="center", va="center", fontsize=10.4, fontweight="bold", color="#922B21", zorder=3)

    # Overlap shaded region (15-45 Hz)
    ax2.axvspan(fx(15), fx(45), ymin=0.28, ymax=0.82, color="#E74C3C", alpha=0.12, zorder=1)
    ax2.annotate(
        "Severe In-Band Overlap (15–45 Hz)\n1–40 Hz Bandpass CANNOT separate\nBeta/Gamma EEG from Scalp EMG!",
        xy=(fx(26), 3.15), xytext=(fx(2.2), 0.65),
        fontsize=9.5, fontweight="bold", color="#922B21",
        bbox=dict(boxstyle="round,pad=0.38", fc="#FFF5F5", ec=RED, lw=1.2),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )

    # Hard 40 Hz Bandpass cutoff line
    ax2.axvline(fx(40), color=SLATE, ls="--", lw=1.6, ymin=0.18, ymax=0.96)
    ax2.text(fx(43), 9.65, "40 Hz Bandpass Cutoff\n(Leaves 15–40 Hz EMG inside)", fontsize=9.0, fontweight="bold", color=SLATE, va="center")

    ax2.grid(True, axis="x", ls=":", alpha=0.35)
    out = ASSET_FIG_DIR / "00_concept_wearable_ica_and_overlap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[saved] {out.name}")


def make_slide4_protocol_controls() -> None:
    """Slide 4 diagram: Top = Leak-free 80/10/10 split pipeline; Bottom = 5-step Controlled Ablation Ladder."""
    fig, ax = plt.subplots(figsize=(14.6, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- Top Section: Leak-Free Split Protocol ----
    ax.text(0.015, 0.945, "A. Strict Leak-Free Experimental Protocol (EEGdenoiseNet, 512 Hz, 2-Second Epochs)",
            fontsize=12.6, fontweight="bold", color=NAVY)

    box(ax, 0.015, 0.72, 0.205, 0.16, LIGHT_BLUE, BLUE, lw=1.5)
    ax.text(0.117, 0.825, "Clean EEG Pool", ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(0.117, 0.76, "4,514 epochs (512 Hz)\n64-ch ICA cleaned", ha="center", va="center", fontsize=9.0, color=SLATE)

    box(ax, 0.015, 0.52, 0.205, 0.16, LIGHT_RED, RED, lw=1.5)
    ax.text(0.117, 0.625, "Raw EMG Pool", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#922B21")
    ax.text(0.117, 0.56, "5,598 epochs (512 Hz)\nFacial bipolar sEMG", ha="center", va="center", fontsize=9.0, color=SLATE)

    # Independent split boxes
    box(ax, 0.285, 0.52, 0.245, 0.36, LIGHT_GREEN, GREEN, lw=1.6)
    ax.text(0.407, 0.835, "Independent 80 / 10 / 10 Split", ha="center", va="center", fontsize=10.4, fontweight="bold", color="#1E8449")
    ax.text(0.407, 0.748, "Train: 3,611 EEG  |  4,478 EMG", ha="center", va="center", fontsize=9.2, color="#1E8449")
    ax.text(0.407, 0.678, "Val:     451 EEG  |    559 EMG", ha="center", va="center", fontsize=9.2, color="#1E8449")
    ax.text(0.407, 0.608, "Test:    452 EEG  |    561 EMG", ha="center", va="center", fontsize=9.2, color="#1E8449")
    ax.text(0.407, 0.550, "Zero Epoch Overlap (Asserted)", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#145A32")

    arrow(ax, 0.232, 0.80, 0.274, 0.76, BLUE)
    arrow(ax, 0.232, 0.60, 0.274, 0.64, RED)

    # Separate synthesis inside each split
    box(ax, 0.595, 0.52, 0.39, 0.36, LIGHT_GRAY, NAVY, lw=1.5)
    ax.text(0.79, 0.835, "Mixing Performed ONLY Inside Each Split", ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(0.79, 0.735, "Train Synthesis:  SNR in [-7, +2] dB\n(Standard Linear ID  vs.  Physics-Informed Aug)", ha="center", va="center", fontsize=9.3, color="#2C3E50")
    ax.text(0.79, 0.615, "Held-Out Evaluation Suites:\nTest-ID  |  Test-OOD (60/45 Hz)  |  Test-Clean\n+ Disentangled & Cross-Filter Stress Tracks", ha="center", va="center", fontsize=9.0, color="#1B4F72", fontweight="bold")
    arrow(ax, 0.542, 0.70, 0.584, 0.70, NAVY)

    # ---- Bottom Section: 5-Group Controlled Ablation Ladder ----
    ax.text(0.015, 0.435, "B. Progressive Controlled-Variable Design (Isolating Architecture, Loss, Burst+Clean & Lowpass)",
            fontsize=12.6, fontweight="bold", color=NAVY)

    cards = [
        ("Exp A0", "DSP Floor", "Bandpass 1–40 Hz\n(No Training)", "Traditional\nfilter baseline", LIGHT_GRAY, SLATE, SLATE),
        ("Exp A1", "Literature", "1D-ResCNN\nID Mix + MSE", "Standard SOTA\nparadigm (Sun '20)", LIGHT_RED, RED, "#922B21"),
        ("Exp A2", "Arch + Loss", "1D-ResUNet\nID Mix + MR-STFT", "Tests if U-Net +\nSTFT alone suffices", LIGHT_ORANGE, ORANGE, "#A04000"),
        ("Exp A3-noLP", "V2 Ablation", "1D-ResUNet\nBurst + Clean ONLY", "Removes Lowpass\nto isolate its role", LIGHT_PURPLE, PURPLE, "#6C3483"),
        ("Exp A3 (Ours)", "Full Physics", "1D-ResUNet\nLP + Burst + Clean", "Complete Physics\n+ MR-STFT Solution", LIGHT_BLUE, BLUE, "#1B4F72"),
    ]

    w = 0.162
    gap = 0.039
    x0 = 0.018
    for i, (name, tag, spec, role, fc, ec, tc) in enumerate(cards):
        xi = x0 + i * (w + gap)
        box(ax, xi, 0.045, w, 0.34, fc, ec, lw=2.2 if i == 4 else 1.4)
        box(ax, xi + 0.016, 0.325, w - 0.032, 0.038, ec, ec, lw=0.5, r=0.012)
        ax.text(xi + w / 2, 0.344, tag, ha="center", va="center", fontsize=8.8, fontweight="bold", color="white", zorder=5)
        ax.text(xi + w / 2, 0.275, name, ha="center", va="center", fontsize=10.8, fontweight="bold", color=tc)
        ax.text(xi + w / 2, 0.195, spec, ha="center", va="center", fontsize=9.0, color="#222222")
        ax.text(xi + w / 2, 0.095, role, ha="center", va="center", fontsize=8.5, style="italic", color=SLATE)
        if i < len(cards) - 1:
            arrow(ax, xi + w + 0.012, 0.215, xi + w + gap - 0.012, 0.215, SLATE, lw=1.7)

    out = ASSET_FIG_DIR / "00b_protocol_and_ablation_design.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[saved] {out.name}")


def make_slide5_method_pipeline() -> None:
    """Slide 5 diagram: Physics-Informed Augmentation + Exact 4-Level 1D-ResUNet (with 2x MHSA) + Dual-Domain Loss."""
    fig, ax = plt.subplots(figsize=(14.8, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- Left Container: Innovation 1 - Physics-Informed Synthesis ----
    box(ax, 0.015, 0.04, 0.435, 0.90, "#F8FBFE", BLUE, lw=1.8, r=0.02)
    ax.text(0.232, 0.90, "Innovation 1: Physics-Informed Data Synthesis", ha="center", va="center", fontsize=11.8, fontweight="bold", color=NAVY)
    ax.text(0.232, 0.85, "Upgrading y = x + λn  to  y = x + (h_spatial * n)", ha="center", va="center", fontsize=9.8, fontweight="bold", color=BLUE)

    # Inputs (Clean on bottom-left, EMG on top-left so arrows NEVER cross!)
    box(ax, 0.032, 0.56, 0.132, 0.13, LIGHT_RED, RED, lw=1.4)
    ax.text(0.098, 0.625, "Raw EMG\nn (1024 pts)", ha="center", va="center", fontsize=9.4, fontweight="bold", color="#922B21")

    box(ax, 0.032, 0.19, 0.132, 0.13, "white", NAVY, lw=1.4)
    ax.text(0.098, 0.255, "Clean EEG\nx (1024 pts)", ha="center", va="center", fontsize=9.4, fontweight="bold", color=NAVY)

    # 3 Physics Operations (unified total-sample % denominator: 67.5% / 30% / 10%)
    ops = [
        (0.62, "1. Random Lowpass (67.5% total)", "75% of noisy | 4th-ord filtfilt, fc~U[35,110] Hz\nScalp attenuation; breaks >80 Hz shortcut", LIGHT_BLUE, BLUE, "#1B4F72"),
        (0.39, "2. Local Burst Mask (30% total)", "Tukey window (0.3–1.5 s active burst)\nSimulates transient jaw clenching", LIGHT_PURPLE, PURPLE, "#6C3483"),
        (0.16, "3. Clean Identity (10% total)", "Pure clean input (y = x -> x_hat = x)\nPrevents over-denoising on quiet epochs", LIGHT_GREEN, GREEN, "#1E8449"),
    ]
    for y0, title, desc, fc, ec, tc in ops:
        box(ax, 0.205, y0, 0.228, 0.165, fc, ec, lw=1.4)
        ax.text(0.319, y0 + 0.138, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=tc)
        ax.text(0.319, y0 + 0.062, desc, ha="center", va="center", fontsize=8.3, color="#2C3E50", linespacing=1.3)

    arrow(ax, 0.174, 0.65, 0.196, 0.70, BLUE, lw=1.5)
    arrow(ax, 0.174, 0.59, 0.196, 0.475, PURPLE, lw=1.5)
    arrow(ax, 0.174, 0.25, 0.196, 0.245, GREEN, lw=1.5)

    # Bottom summary pill
    box(ax, 0.038, 0.068, 0.39, 0.048, NAVY, NAVY, lw=0.5, r=0.012)
    ax.text(0.233, 0.092, "SNR-Scaled Mix [-7, +2] dB  +  Std Norm  ->  Input y",
            ha="center", va="center", fontsize=8.7, fontweight="bold", color="white")

    # ---- Bridge arrow ----
    arrow(ax, 0.456, 0.50, 0.490, 0.50, NAVY, lw=2.4)
    ax.text(0.473, 0.545, "y", ha="center", fontsize=11.5, fontweight="bold", color=NAVY)

    # ---- Right Container: Innovation 2 - Exact 4-Level 1D-ResUNet + 2x MHSA + Dual Loss ----
    box(ax, 0.495, 0.04, 0.490, 0.90, "#FDFEFE", NAVY, lw=1.8, r=0.02)
    ax.text(0.740, 0.90, "Innovation 2: 4-Level 1D-ResUNet + Dual-Domain Loss", ha="center", va="center", fontsize=11.8, fontweight="bold", color=NAVY)
    ax.text(0.740, 0.85, "4.11M Params  |  GroupNorm ResBlocks  +  2× Bottleneck MHSA (4 Heads)", ha="center", va="center", fontsize=9.3, color=SLATE)

    # Exact 4-level U-Net matching 03_models.py (32->64->128->256, Bottleneck L=64 C=256)
    enc = [
        (0.512, 0.67, 0.056, 0.105, "Enc 1\n32 ch\n1024", LIGHT_BLUE, BLUE),
        (0.556, 0.535, 0.056, 0.100, "Enc 2\n64 ch\n512", LIGHT_BLUE, BLUE),
        (0.600, 0.405, 0.056, 0.095, "Enc 3\n128 ch\n256", LIGHT_BLUE, BLUE),
        (0.644, 0.280, 0.056, 0.090, "Enc 4\n256 ch\n128", LIGHT_BLUE, BLUE),
    ]
    bot = (0.700, 0.215, 0.088, 0.095, "Bottleneck\n2× MHSA\n256 ch, L=64", "#D6EAF8", NAVY)
    dec = [
        (0.788, 0.280, 0.056, 0.090, "Dec 4\n256 ch\n128", "#E8F8F5", "#148F77"),
        (0.832, 0.405, 0.056, 0.095, "Dec 3\n128 ch\n256", "#E8F8F5", "#148F77"),
        (0.876, 0.535, 0.056, 0.100, "Dec 2\n64 ch\n512", "#E8F8F5", "#148F77"),
        (0.920, 0.67, 0.056, 0.105, "Dec 1\n32 ch\n1024", "#E8F8F5", "#148F77"),
    ]

    for x, y, w, h, lbl, fc, ec in enc + [bot] + dec:
        box(ax, x, y, w, h, fc, ec, lw=1.25, r=0.010)
        ax.text(x + w / 2, y + h / 2, lbl, ha="center", va="center", fontsize=7.6, fontweight="bold", color=NAVY)

    # Skip connections (horizontal dashed arrows)
    for (ex, ey, ew, eh, *_), (dx, dy, dw, dh, *_) in zip(enc, reversed(dec)):
        ax.annotate(
            "", xy=(dx - 0.008, dy + dh / 2), xytext=(ex + ew + 0.008, ey + eh / 2),
            arrowprops=dict(arrowstyle="->", ls="--", color=ORANGE, lw=1.25), zorder=3,
        )
    ax.text(0.744, 0.768, "Multi-Scale Skip Connections (Preserve Waveform Phase)",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#B9770E",
            bbox=dict(boxstyle="round,pad=0.22", fc=LIGHT_ORANGE, ec=ORANGE, lw=0.9))

    # Down / Up tiny flow arrows along U-shape
    for i in range(3):
        ex, ey, ew, eh, *_ = enc[i]
        nx, ny, nw, nh, *_ = enc[i + 1]
        arrow(ax, ex + ew / 2, ey - 0.008, nx + 0.005, ny + nh / 2, BLUE, lw=1.1)
    arrow(ax, 0.672, 0.272, 0.695, 0.262, NAVY, lw=1.1)
    arrow(ax, 0.793, 0.262, 0.816, 0.272, "#148F77", lw=1.1)
    for i in range(3):
        dx, dy, dw, dh, *_ = dec[i]
        nx, ny, nw, nh, *_ = dec[i + 1]
        arrow(ax, dx + dw - 0.005, dy + dh / 2, nx + nw / 2, ny - 0.008, "#148F77", lw=1.1)

    # Dual-Domain Loss box at bottom
    box(ax, 0.515, 0.068, 0.452, 0.112, LIGHT_ORANGE, ORANGE, lw=1.5)
    ax.text(0.741, 0.148, "Dual-Domain Loss:  L = L_MSE(x, x_hat) + 0.1 * L_MR-STFT(x, x_hat)",
            ha="center", va="center", fontsize=9.1, fontweight="bold", color="#935116")
    ax.text(0.741, 0.098, "Time MSE: waveform fidelity  |  MR-STFT (Yamamoto '20): preserves Alpha & Beta PSD",
            ha="center", va="center", fontsize=8.2, color="#5D4037")

    out = ASSET_FIG_DIR / "00c_method_physics_resunet_pipeline.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[saved] {out.name}")


def main() -> None:
    make_slide2_wearable_overlap()
    make_slide4_protocol_controls()
    make_slide5_method_pipeline()


if __name__ == "__main__":
    main()
