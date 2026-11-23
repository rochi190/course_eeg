#!/usr/bin/env python3
"""Generate 300 DPI presentation figures for SPEC B."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJ_ROOT / "data" / "cache"
OUT_DIR = PROJ_ROOT / "outputs"
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


def plot_fig1_leakage_gap(df: pd.DataFrame) -> None:
    """Fig 1: Exp B0 (Random Leaky) vs Exp B1 (Strict Subject-Independent) collapse."""
    tasks = ["AD_vs_CN", "AD_vs_FTD", "FTD_vs_CN"]
    task_lbls = ["AD vs. CN", "AD vs. FTD", "FTD vs. CN"]
    seg_df = df[df["level"] == "Segment"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    metrics = [
        ("accuracy_mean", "accuracy_std", "Segment-Level Accuracy (%)"),
        ("macro_f1_mean", "macro_f1_std", "Segment-Level Macro-F1 (%)"),
    ]

    x = np.arange(len(tasks))
    w = 0.34
    for ax, (m_col, s_col, ylabel) in zip(axes, metrics):
        b0_m, b0_s, b1_m, b1_s = [], [], [], []
        for t in tasks:
            r0 = seg_df[(seg_df["task"] == t) & (seg_df["experiment"].str.startswith("Exp B0"))].iloc[0]
            r1 = seg_df[(seg_df["task"] == t) & (seg_df["experiment"].str.startswith("Exp B1"))].iloc[0]
            b0_m.append(float(r0[m_col]) * 100.0)
            b0_s.append(float(r0[s_col]) * 100.0 if np.isfinite(r0[s_col]) else 0.0)
            b1_m.append(float(r1[m_col]) * 100.0)
            b1_s.append(float(r1[s_col]) * 100.0 if np.isfinite(r1[s_col]) else 0.0)

        bars0 = ax.bar(
            x - w / 2, b0_m, w, yerr=b0_s, capsize=4,
            color="#e74c3c", edgecolor="black", linewidth=0.7,
            label="Exp B0: Random Segment Split (Leaky)",
        )
        bars1 = ax.bar(
            x + w / 2, b1_m, w, yerr=b1_s, capsize=4,
            color="#2c3e50", edgecolor="black", linewidth=0.7,
            label="Exp B1: Strict 5-Fold Subject-Independent",
        )
        for idx, (bar0, bar1, v0, s0, v1, s1) in enumerate(zip(bars0, bars1, b0_m, b0_s, b1_m, b1_s)):
            ax.text(bar0.get_x() + bar0.get_width() / 2, v0 + s0 + 1.5, f"{v0:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#c0392b")
            ax.text(bar1.get_x() + bar1.get_width() / 2, v1 + s1 + 1.5, f"{v1:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#2c3e50")
            gap = v0 - v1
            ax.annotate(
                f"-{gap:.1f} pp",
                xy=(x[idx], (v0 + v1) / 2),
                ha="center", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fff3cd", ec="#f39c12", lw=0.8),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(task_lbls)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 114)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Epoch-Level Leakage Illusion vs. Strict Subject-Independent Reality (ShallowConvNet)", y=0.92, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    out_path = FIG_DIR / "fig1_leakage_gap.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_fig2_topomap_wpli(cache_path: Path) -> None:
    """Fig 2: Topomap of Theta/Alpha Relative PSD (AD, FTD, CN) + AD-FTD wPLI diff heatmaps."""
    data = np.load(cache_path, allow_pickle=False)
    node_feats = data["node_feats"]  # (N, 19, 13); idx 1=Theta rel PSD, idx 2=Alpha rel PSD
    wpli = data["wpli"]  # (N, 5, 19, 19); idx 1=Theta, idx 2=Alpha
    groups = data["groups"]
    ch_names = [str(c) for c in data["ch_names"]]

    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="raise")

    grp_order = [("A", "AD"), ("F", "FTD"), ("C", "CN")]
    # Compute subject-averaged maps first so every subject contributes equally
    subjs = data["subject_ids"]
    uniq_subjs = np.unique(subjs)
    subj_nf = {g: [] for g, _ in grp_order}
    subj_wpli = {g: [] for g, _ in grp_order}
    for s in uniq_subjs:
        m = subjs == s
        g = str(groups[m][0])
        if g in subj_nf:
            subj_nf[g].append(node_feats[m].mean(axis=0))
            subj_wpli[g].append(wpli[m].mean(axis=0))

    mean_nf = {g: np.mean(v, axis=0) for g, v in subj_nf.items()}
    mean_wpli = {g: np.mean(v, axis=0) for g, v in subj_wpli.items()}

    fig = plt.figure(figsize=(14, 11.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.45], hspace=0.40, wspace=0.35)

    # Top 2 rows: Theta (band_idx=1) and Alpha (band_idx=2) Topomaps across AD, FTD, CN
    band_specs = [
        (0, 1, "Theta (4-8 Hz) Relative PSD"),
        (1, 2, "Alpha (8-13 Hz) Relative PSD"),
    ]
    for row_idx, feat_idx, row_title in band_specs:
        vals_all = np.concatenate([mean_nf[g][:, feat_idx] for g, _ in grp_order])
        vmin, vmax = float(vals_all.min()), float(vals_all.max())
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-3
        im = None
        for col_idx, (g_code, g_label) in enumerate(grp_order):
            ax = fig.add_subplot(gs[row_idx, col_idx * 2 : col_idx * 2 + 2])
            im, _ = mne.viz.plot_topomap(
                mean_nf[g_code][:, feat_idx],
                info,
                axes=ax,
                show=False,
                vlim=(vmin, vmax),
                cmap="viridis",
                contours=4,
            )
            ax.set_title(f"{g_label} - {row_title}", fontsize=12, fontweight="bold")
        cbar_ax = fig.add_axes([0.92, 0.71 - row_idx * 0.25, 0.015, 0.17])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Relative Power", fontsize=11)

    # Bottom row: AD - FTD wPLI difference heatmaps (Theta & Alpha)
    diff_specs = [
        (0, 1, "AD - FTD wPLI Difference: Theta (4-8 Hz)"),
        (1, 2, "AD - FTD wPLI Difference: Alpha (8-13 Hz)"),
    ]
    for idx, b_idx, title in diff_specs:
        ax = fig.add_subplot(gs[2, idx * 3 : idx * 3 + 3])
        diff = mean_wpli["A"][b_idx] - mean_wpli["F"][b_idx]
        lim = max(float(np.abs(diff).max()), 1e-3)
        im_h = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xticks(range(19))
        ax.set_yticks(range(19))
        ax.set_xticklabels(ch_names, rotation=60, fontsize=9.5)
        ax.set_yticklabels(ch_names, fontsize=9.5)
        cb = fig.colorbar(im_h, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("wPLI Diff (AD - FTD)", fontsize=10.5)

    fig.suptitle("Group Spectral Topographies (Theta/Alpha) & AD vs. FTD Functional Connectivity Shifts", fontsize=14, fontweight="bold", y=0.98)
    out_path = FIG_DIR / "fig2_topomap_plv.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_fig3_f1_comparison(df: pd.DataFrame) -> None:
    """Fig 3: Subject-level Macro-F1 vs 2026 benchmarks across binary tasks."""
    tasks = ["AD_vs_CN", "AD_vs_FTD", "FTD_vs_CN"]
    task_lbls = ["AD vs. CN", "AD vs. FTD", "FTD vs. CN"]
    sota_refs = [75.82, 70.65, 56.35]

    exps = [
        ("Exp B1 (ShallowConvNet)", "#7f8c8d", "Exp B1: ShallowConvNet (Raw EEG)"),
        ("Exp B2a (LightGBM)", "#ff7f0e", "Exp B2a: LightGBM (PSD + wPLI)"),
        ("Exp B2b (Linear SVM)", "#2ca02c", "Exp B2b: Linear SVM (PSD + wPLI)"),
        ("Exp B3 (LightGAT)", "#1f77b4", "Exp B3: LightGAT (<50K Graph Net)"),
    ]

    sub_df = df[df["level"] == "Subject"]
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    x = np.arange(len(tasks))
    width = 0.19

    for i, (exp_key, color, label) in enumerate(exps):
        means, stds = [], []
        for t in tasks:
            row = sub_df[(sub_df["task"] == t) & (sub_df["experiment"] == exp_key)].iloc[0]
            means.append(float(row["macro_f1_mean"]) * 100.0)
            stds.append(float(row["macro_f1_std"]) * 100.0 if np.isfinite(row["macro_f1_std"]) else 0.0)

        bars = ax.bar(
            x + (i - 1.5) * width, means, width, yerr=stds, capsize=3.5,
            color=color, edgecolor="black", linewidth=0.7, label=label,
        )
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2, m + s + 1.2,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=9.8, fontweight="bold",
            )

    # 2026 Literature Benchmark horizontal dashed markers per task
    for idx, ref in enumerate(sota_refs):
        lbl = "2026 Literature Best (bioRxiv 2026.08)" if idx == 0 else None
        ax.hlines(
            ref, x[idx] - 0.43, x[idx] + 0.43,
            colors="#d62728", linestyles="--", linewidth=2.3, label=lbl, zorder=5,
        )
        ax.text(
            x[idx] + 0.42, ref + 0.8, f"2026 Ref: {ref:.2f}%",
            color="#d62728", fontsize=10, fontweight="bold", ha="right", va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(task_lbls, fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Subject-Level Macro-F1 (%)")
    ax.set_title(
        "Strict 5-Fold Subject-Independent Benchmark vs. 2026 Literature SOTA",
        fontweight="bold", pad=14,
    )
    ax.set_ylim(0, 102)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", ncol=2, frameon=True)

    fig.tight_layout()
    out_path = FIG_DIR / "fig3_f1_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-name", type=str, default="ahepa_dataset.npz")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DIR / "metrics_summary.csv")
    plot_fig1_leakage_gap(df)
    plot_fig2_topomap_wpli(CACHE_DIR / args.cache_name)
    plot_fig3_f1_comparison(df)


if __name__ == "__main__":
    main()
