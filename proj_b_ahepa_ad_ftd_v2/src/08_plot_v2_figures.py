#!/usr/bin/env python3
"""Generate 300 DPI publication-grade V2 figures for SPEC B-V2 with zero text overlap."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SRC_V2 = Path(__file__).resolve().parent
V2_ROOT = SRC_V2.parent
V1_ROOT = V2_ROOT.parent / "proj_b_ahepa_ad_ftd"
CACHE_DIR = V2_ROOT / "data" / "cache_v2"
OUT_DIR = V2_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.2,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_fig4_ad_ftd_rescue(v1_csv: Path, v2_csv: Path) -> None:
    """Fig 4: AD_vs_FTD V1 (ShallowConvNet, LightGAT) vs V2 (B4b, B5a, B6, B7)."""
    df1 = pd.read_csv(v1_csv)
    df1 = df1[(df1["level"] == "Subject") & (df1["task"] == "AD_vs_FTD")]
    df2 = pd.read_csv(v2_csv)
    df2 = df2[df2["task"] == "AD_vs_FTD"]

    models = [
        ("V1: ShallowConvNet", df1[df1["experiment"] == "Exp B1 (ShallowConvNet)"].iloc[0], "#95a5a6"),
        ("V1: LightGAT", df1[df1["experiment"] == "Exp B3 (LightGAT)"].iloc[0], "#7f8c8d"),
        ("V2: B4b Riemannian", df2[df2["experiment"] == "Exp B4b (Riemannian Subject)"].iloc[0], "#ff7f0e"),
        ("V2: B5a Neuro-LogReg", df2[df2["experiment"] == "Exp B5a (Neuro ElasticNet)"].iloc[0], "#2ca02c"),
        ("V2: B6 ResGCN (<25K)", df2[df2["experiment"] == "Exp B6 (Spectral ResGCN <25K)"].iloc[0], "#9467bd"),
        ("V2: B7 Ensemble", df2[df2["experiment"] == "Exp B7 (Multimodal Ensemble)"].iloc[0], "#1f77b4"),
    ]

    metrics = [
        ("macro_f1_mean", "Subject Macro-F1 (%)"),
        ("sensitivity_mean", "AD Sensitivity (%)"),
        ("specificity_mean", "FTD Specificity (%)"),
    ]

    x = np.arange(len(metrics))
    w = 0.13
    fig, ax = plt.subplots(figsize=(13.8, 6.8))

    for i, (lbl, row, col) in enumerate(models):
        vals = [float(row[m]) * 100.0 for m, _ in metrics]
        bars = ax.bar(
            x + (i - 2.5) * w, vals, width=w,
            color=col, edgecolor="black", linewidth=0.7, label=lbl,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.2,
                f"{v:.1f}",
                ha="center", va="bottom", fontsize=9.2, fontweight="bold",
            )

    ax.axhline(70.65, color="#d62728", linestyle="--", linewidth=1.8, label="2026 SOTA Macro-F1 (70.65%)")
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics], fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_title("AD vs. FTD Rescue: Recovering FTD Specificity & Macro-F1 (V1 vs. V2)", fontweight="bold", pad=16)
    ax.set_ylim(0, 112)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout()
    out_path = FIG_DIR / "fig4_ad_ftd_rescue.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_fig5_v2_vs_2026sota(v1_csv: Path, v2_csv: Path) -> None:
    """Fig 5: Clean multi-task Subject Macro-F1 vs 2026 SOTA with non-overlapping pill labels."""
    df1 = pd.read_csv(v1_csv)
    df1 = df1[df1["level"] == "Subject"]
    df2 = pd.read_csv(v2_csv)

    tasks = ["AD_vs_CN", "AD_vs_FTD", "FTD_vs_CN"]
    task_lbls = ["AD vs. CN", "AD vs. FTD", "FTD vs. CN"]
    sota_refs = [75.82, 70.65, 56.35]

    series = [
        ("V1: ShallowConvNet", df1, "Exp B1 (ShallowConvNet)", "#7f8c8d"),
        ("V2: B4b Riemannian", df2, "Exp B4b (Riemannian Subject)", "#ff7f0e"),
        ("V2: B5a Neuro-LogReg", df2, "Exp B5a (Neuro ElasticNet)", "#2ca02c"),
        ("V2: B6 ResGCN (<25K)", df2, "Exp B6 (Spectral ResGCN <25K)", "#9467bd"),
        ("V2: B7 Ensemble", df2, "Exp B7 (Multimodal Ensemble)", "#1f77b4"),
    ]

    x = np.arange(len(tasks))
    w = 0.15
    fig, ax = plt.subplots(figsize=(13.6, 7.2))

    top_per_task = np.zeros(len(tasks))
    for i, (lbl, df_src, exp_key, col) in enumerate(series):
        means, stds = [], []
        for t_idx, t in enumerate(tasks):
            r = df_src[(df_src["task"] == t) & (df_src["experiment"] == exp_key)].iloc[0]
            m = float(r["macro_f1_mean"]) * 100.0
            s = float(r["macro_f1_std"]) * 100.0 if np.isfinite(r["macro_f1_std"]) else 0.0
            means.append(m)
            stds.append(s)
            top_per_task[t_idx] = max(top_per_task[t_idx], m + s)

        bars = ax.bar(
            x + (i - 2.0) * w, means, width=w, yerr=stds, capsize=3,
            color=col, edgecolor="black", linewidth=0.7, label=lbl,
        )
        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, m + s + 1.2,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold",
            )

    # Draw 2026 SOTA dashed lines and place clean white pill annotations ABOVE the tallest bar
    for idx, ref in enumerate(sota_refs):
        lbl = "2026 SOTA (bioRxiv 2026.08)" if idx == 0 else None
        ax.hlines(
            ref, x[idx] - 0.41, x[idx] + 0.41,
            colors="#d62728", linestyles="--", linewidth=2.3, label=lbl, zorder=5,
        )
        y_pill = max(ref + 4.5, top_per_task[idx] + 7.5)
        ax.annotate(
            f"2026 SOTA: {ref:.2f}%",
            xy=(x[idx] + 0.36, ref),
            xytext=(x[idx], y_pill),
            ha="center", va="bottom",
            fontsize=10.0, fontweight="bold", color="#c0392b",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#d62728", lw=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2, shrinkA=2, shrinkB=2),
            zorder=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(task_lbls, fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Subject-Level Macro-F1 (%)")
    ax.set_title("Strict 5-Fold Subject-Independent V2 Rescue vs. 2026 Literature SOTA", fontweight="bold", pad=16)
    ax.set_ylim(0, 114)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout()
    out_path = FIG_DIR / "fig5_v2_all_tasks_vs_2026sota.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_fig6_biomarkers(cache_path: Path) -> None:
    """Fig 6: Boxplots + stripplots of AD/FTD/CN neuro-biomarkers with Mann-Whitney U p-values."""
    d = np.load(cache_path, allow_pickle=False)
    grps = d["subj_grps"]
    metrics = [
        (d["subj_ap_alpha"], "Frontal / PO Alpha Log-Ratio\nlog(P_F / P_PO, 8-13 Hz)"),
        (d["subj_slowing"], "Global Slowing Ratio\n(Delta + Theta) / (Alpha + Beta)"),
        (d["subj_slope"], "Global Aperiodic 1/f Slope\n(2-40 Hz excl. 7-14 Hz)"),
    ]
    grp_order = [("A", "AD", "#d62728"), ("F", "FTD", "#ff7f0e"), ("C", "CN", "#2ca02c")]
    pairs = [(0, 2, "AD vs CN"), (0, 1, "AD vs FTD"), (1, 2, "FTD vs CN")]

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.8))

    for ax, (vals, title) in zip(axes, metrics):
        data_by_g = [vals[grps == g] for g, _, _ in grp_order]
        bp = ax.boxplot(
            data_by_g, positions=[0, 1, 2], widths=0.48, patch_artist=True,
            showfliers=False, medianprops=dict(color="black", linewidth=2.0),
        )
        for patch, (_, _, col) in zip(bp["boxes"], grp_order):
            patch.set_facecolor(col)
            patch.set_alpha(0.32)

        for idx, (arr, (_, _, col)) in enumerate(zip(data_by_g, grp_order)):
            jx = idx + rng.uniform(-0.12, 0.12, size=len(arr))
            ax.scatter(jx, arr, color=col, s=26, alpha=0.80, edgecolors="white", linewidths=0.4, zorder=4)

        y_min = float(min(a.min() for a in data_by_g))
        y_max = float(max(a.max() for a in data_by_g))
        span = max(y_max - y_min, 1e-3)

        # Pairwise Mann-Whitney U brackets cleanly stacked above whiskers
        for lvl, (i, j, _) in enumerate(pairs):
            _, pval = mannwhitneyu(data_by_g[i], data_by_g[j], alternative="two-sided")
            p_str = f"p={pval:.1e}" if pval < 1e-3 else f"p={pval:.3f}"
            yb = y_max + span * (0.10 + lvl * 0.14)
            ax.plot([i, i, j, j], [yb - span * 0.02, yb, yb, yb - span * 0.02], color="#333333", lw=1.1)
            ax.text((i + j) / 2.0, yb + span * 0.015, p_str, ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"{lbl}\n(n={len(arr)})" for (_, lbl, _), arr in zip(grp_order, data_by_g)], fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(y_min - span * 0.08, y_max + span * 0.52)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Subject-Level Neurophysiological Biomarkers Across AD, FTD, and CN (ds004504)", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIG_DIR / "fig6_neuro_biomarkers_boxplot.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    v1_csv = V1_ROOT / "outputs" / "metrics_summary.csv"
    v2_csv = OUT_DIR / "rescue_metrics_summary.csv"
    plot_fig4_ad_ftd_rescue(v1_csv, v2_csv)
    plot_fig5_v2_vs_2026sota(v1_csv, v2_csv)
    plot_fig6_biomarkers(CACHE_DIR / "ahepa_v2_features.npz")


if __name__ == "__main__":
    main()
