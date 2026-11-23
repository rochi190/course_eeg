#!/usr/bin/env python3
"""Run Exp B0 (leaky control) & Exp B1-B3 (strict 5-fold subject-independent CV)."""

from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
import random
import time
import warnings

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _load_local(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SRC_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SRC_DIR.parent
CACHE_DIR = PROJ_ROOT / "data" / "cache"
OUT_DIR = PROJ_ROOT / "outputs"

models = _load_local("b_models_mod", SRC_DIR / "03_models.py")

TASKS = [
    ("AD_vs_CN", ["A", "C"], ["C", "A"]),   # C=0 (neg), A=1 (pos)
    ("AD_vs_FTD", ["A", "F"], ["F", "A"]),  # F=0, A=1
    ("FTD_vs_CN", ["F", "C"], ["C", "F"]),  # C=0 (neg), F=1 (pos)
    ("AD_vs_FTD_vs_CN", ["A", "F", "C"], ["C", "F", "A"]),  # C=0, F=1, A=2
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(
    y_true: np.ndarray, probs: np.ndarray, n_classes: int
) -> dict[str, float]:
    preds = probs.argmax(axis=1)
    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    if n_classes == 2:
        try:
            auroc = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            auroc = float("nan")
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / max(1, tp + fn))
        spec = float(tn / max(1, tn + fp))
    else:
        try:
            auroc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auroc = float("nan")
        sens = float("nan")
        spec = float("nan")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "sensitivity": sens,
        "specificity": spec,
    }


def subject_soft_vote(
    subject_ids: np.ndarray, y_true: np.ndarray, probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average predicted probabilities across all epochs of each subject."""
    uniq_subjs = np.unique(subject_ids)
    subj_y = np.zeros(len(uniq_subjs), dtype=np.int64)
    subj_probs = np.zeros((len(uniq_subjs), probs.shape[1]), dtype=np.float64)
    for i, s in enumerate(uniq_subjs):
        mask = subject_ids == s
        subj_y[i] = int(y_true[mask][0])
        subj_probs[i] = probs[mask].mean(axis=0)
    return subj_y, subj_probs


def train_shallow_fold(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Train ShallowConvNet on train fold (with train-only channel z-score) and return test probs."""
    set_seed(seed)
    # Channel-wise standardization fit strictly on train fold, computed in-place on GPU to save RAM
    x_tr_n = x_tr.to(device).clone()
    mean = x_tr_n.mean(dim=(0, 2), keepdim=True)
    std = x_tr_n.std(dim=(0, 2), keepdim=True) + 1e-6
    x_tr_n.sub_(mean).div_(std)
    x_va_n = x_va.to(device).clone()
    x_va_n.sub_(mean).div_(std)
    y_tr_d = y_tr.to(device)

    cw = compute_class_weights(y_tr.numpy(), n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    model = models.ShallowConvNet(n_chans=19, n_times=1000, n_classes=n_classes, dropout=0.5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n = x_tr_n.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x_tr_n[idx])
                loss = criterion(logits.float(), y_tr_d[idx])
            assert torch.isfinite(loss), "Non-finite ShallowConvNet loss"
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

    # Exact train-fold-only BatchNorm running-stat recalibration before eval
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for s in range(0, n, 512):
            _ = model(x_tr_n[s : s + 512])

    model.eval()
    probs_list = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for s in range(0, x_va_n.size(0), 512):
            logits = model(x_va_n[s : s + 512])
            probs_list.append(F.softmax(logits.float(), dim=-1).cpu().numpy())

    out = np.concatenate(probs_list, axis=0)
    del x_tr_n, x_va_n, y_tr_d, mean, std, model, opt, sched
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def train_lightgat_fold(
    nf_tr: np.ndarray,
    wpli_tr: np.ndarray,
    plv_tr: np.ndarray,
    y_tr: np.ndarray,
    nf_va: np.ndarray,
    wpli_va: np.ndarray,
    plv_va: np.ndarray,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Train LightGAT (<50K params) with train-fold-only node feature scaling."""
    set_seed(seed)
    n_tr, n_nodes, n_feats = nf_tr.shape
    scaler = StandardScaler()
    nf_tr_s = scaler.fit_transform(nf_tr.reshape(-1, n_feats)).reshape(n_tr, n_nodes, n_feats)
    nf_va_s = scaler.transform(nf_va.reshape(-1, n_feats)).reshape(nf_va.shape[0], n_nodes, n_feats)

    nf_tr_t = torch.from_numpy(nf_tr_s.astype(np.float32)).to(device)
    w_tr_t = torch.from_numpy(wpli_tr.astype(np.float32)).to(device)
    p_tr_t = torch.from_numpy(plv_tr.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64)).to(device)

    nf_va_t = torch.from_numpy(nf_va_s.astype(np.float32)).to(device)
    w_va_t = torch.from_numpy(wpli_va.astype(np.float32)).to(device)
    p_va_t = torch.from_numpy(plv_va.astype(np.float32)).to(device)

    cw = compute_class_weights(y_tr, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    model = models.LightGAT(
        in_feats=n_feats, edge_dim=10, hidden_dim=32, num_heads=4, n_nodes=19, n_classes=n_classes, dropout=0.5, topk=7
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_tr, device=device)
        for s in range(0, n_tr, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(nf_tr_t[idx], w_tr_t[idx], p_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            assert torch.isfinite(loss), "Non-finite LightGAT loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

    model.eval()
    probs_list = []
    with torch.no_grad():
        for s in range(0, nf_va_t.size(0), 512):
            logits = model(nf_va_t[s : s + 512], w_va_t[s : s + 512], p_va_t[s : s + 512])
            probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs_list, axis=0)


def run_classical_fold(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train LightGBM and Linear SVM on train-fold-scaled tabular features."""
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_va_s = scaler.transform(x_va)

    lgbm = LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        num_leaves=15,
        max_bin=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.25,
        min_child_samples=30,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(x_tr_s, y_tr)
    probs_lgbm = lgbm.predict_proba(x_va_s)

    svm = LinearSVC(C=0.005, dual=False, class_weight="balanced", max_iter=3000, random_state=seed)
    svm.fit(x_tr_s, y_tr)
    dec = svm.decision_function(x_va_s)
    if n_classes == 2:
        p1 = 1.0 / (1.0 + np.exp(-np.clip(dec, -20.0, 20.0)))
        probs_svm = np.stack([1.0 - p1, p1], axis=1)
    else:
        ex = np.exp(dec - dec.max(axis=1, keepdims=True))
        probs_svm = ex / ex.sum(axis=1, keepdims=True)
    return probs_lgbm, probs_svm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-name", type=str, default="ahepa_dataset.npz")
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    data = np.load(CACHE_DIR / args.cache_name, allow_pickle=False)
    raw_epochs = data["raw_epochs"]
    node_feats = data["node_feats"]
    wpli = data["wpli"]
    plv = data["plv"]
    subject_ids = data["subject_ids"]
    groups = data["groups"]

    # Build flat feature matrix for classical ML: node_feats (247) + upper tri of 5-band wPLI (855) = 1102
    tri_r, tri_c = np.triu_indices(19, k=1)
    wpli_tri = wpli[:, :, tri_r, tri_c].reshape(len(raw_epochs), -1)
    tab_feats = np.concatenate([node_feats.reshape(len(raw_epochs), -1), wpli_tri], axis=1).astype(np.float32)

    # Verify LightGAT param count < 50K
    for nc in [2, 3]:
        m_chk = models.LightGAT(in_feats=node_feats.shape[-1], n_classes=nc)
        print(f"[model] LightGAT (classes={nc}) params = {sum(p.numel() for p in m_chk.parameters()):,}")
    m_scn = models.ShallowConvNet(n_classes=2)
    print(f"[model] ShallowConvNet (classes=2) params = {sum(p.numel() for p in m_scn.parameters()):,}")

    if args.smoke:
        n_splits = 2
        n_seeds_b0 = 2
        n_seeds_strict = 2
        ep_b0, ep_b1, ep_b3 = 5, 5, 5
        print("[mode] SMOKE TEST (2 folds, 2 seeds, 5 epochs)")
    else:
        n_splits = 5
        n_seeds_b0 = min(3, args.n_seeds)
        n_seeds_strict = args.n_seeds
        ep_b0, ep_b1, ep_b3 = 20, 30, 35
        print(f"[mode] FULL BENCHMARK (5-fold CV, B0 seeds={n_seeds_b0}, strict seeds={n_seeds_strict})")

    t_start = time.time()
    records = []

    for task_name, valid_grps, label_order in TASKS:
        mask = np.isin(groups, valid_grps)
        label_map = {g: idx for idx, g in enumerate(label_order)}
        y_task = np.array([label_map[g] for g in groups[mask]], dtype=np.int64)
        subjs_task = subject_ids[mask]
        raw_task = torch.from_numpy(raw_epochs[mask])
        nf_task = node_feats[mask]
        w_task = wpli[mask]
        p_task = plv[mask]
        tab_task = tab_feats[mask]
        n_classes = len(label_order)

        print(f"\n==================== TASK: {task_name} (epochs={len(y_task)}, subjects={len(np.unique(subjs_task))}) ====================")

        # --- Exp B0: Intentional Segment-Level Random Leakage ---
        for seed in range(n_seeds_b0):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=100 + seed)
            oof_probs = np.zeros((len(y_task), n_classes), dtype=np.float64)
            for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_task)), y_task)):
                oof_probs[va_idx] = train_shallow_fold(
                    raw_task[tr_idx], torch.from_numpy(y_task[tr_idx]),
                    raw_task[va_idx], n_classes=n_classes,
                    epochs=ep_b0, batch_size=256, lr=1e-3, weight_decay=1e-3,
                    seed=seed * 10 + fold, device=device,
                )
            seg_m = compute_metrics(y_task, oof_probs, n_classes)
            sy, sp = subject_soft_vote(subjs_task, y_task, oof_probs)
            sub_m = compute_metrics(sy, sp, n_classes)
            records.append({"task": task_name, "experiment": "Exp B0 (Leaky ShallowConvNet)", "seed": seed, "level": "Segment", **seg_m})
            records.append({"task": task_name, "experiment": "Exp B0 (Leaky ShallowConvNet)", "seed": seed, "level": "Subject", **sub_m})
            print(
                f"  [{task_name} B0 seed={seed}] Segment Acc={seg_m['accuracy']:.3f} "
                f"F1={seg_m['macro_f1']:.3f} | Subject F1={sub_m['macro_f1']:.3f}"
            )

        # --- Exp B1, B2a, B2b, B3: Strict StratifiedGroupKFold ---
        for seed in range(n_seeds_strict):
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42 + seed)
            oof_b1 = np.zeros((len(y_task), n_classes), dtype=np.float64)
            oof_b2a = np.zeros((len(y_task), n_classes), dtype=np.float64)
            oof_b2b = np.zeros((len(y_task), n_classes), dtype=np.float64)
            oof_b3 = np.zeros((len(y_task), n_classes), dtype=np.float64)

            for fold, (tr_idx, va_idx) in enumerate(sgkf.split(np.zeros(len(y_task)), y_task, groups=subjs_task)):
                tr_subjs = set(subjs_task[tr_idx])
                va_subjs = set(subjs_task[va_idx])
                assert tr_subjs.isdisjoint(va_subjs), f"Subject leakage detected in {task_name} fold {fold}!"

                fold_seed = seed * 100 + fold
                oof_b1[va_idx] = train_shallow_fold(
                    raw_task[tr_idx], torch.from_numpy(y_task[tr_idx]),
                    raw_task[va_idx], n_classes=n_classes,
                    epochs=ep_b1, batch_size=256, lr=1e-3, weight_decay=1e-2,
                    seed=fold_seed, device=device,
                )
                p_lgbm, p_svm = run_classical_fold(
                    tab_task[tr_idx], y_task[tr_idx], tab_task[va_idx], n_classes=n_classes, seed=fold_seed
                )
                oof_b2a[va_idx] = p_lgbm
                oof_b2b[va_idx] = p_svm

                oof_b3[va_idx] = train_lightgat_fold(
                    nf_task[tr_idx], w_task[tr_idx], p_task[tr_idx], y_task[tr_idx],
                    nf_task[va_idx], w_task[va_idx], p_task[va_idx],
                    n_classes=n_classes, epochs=ep_b3, batch_size=256,
                    lr=1e-3, weight_decay=1e-2, seed=fold_seed, device=device,
                )

            for exp_lbl, oof_arr in [
                ("Exp B1 (ShallowConvNet)", oof_b1),
                ("Exp B2a (LightGBM)", oof_b2a),
                ("Exp B2b (Linear SVM)", oof_b2b),
                ("Exp B3 (LightGAT)", oof_b3),
            ]:
                seg_m = compute_metrics(y_task, oof_arr, n_classes)
                sy, sp = subject_soft_vote(subjs_task, y_task, oof_arr)
                sub_m = compute_metrics(sy, sp, n_classes)
                records.append({"task": task_name, "experiment": exp_lbl, "seed": seed, "level": "Segment", **seg_m})
                records.append({"task": task_name, "experiment": exp_lbl, "seed": seed, "level": "Subject", **sub_m})

            print(
                f"  [{task_name} seed={seed}] Subject Macro-F1 -> "
                f"B1={f1_score(subject_soft_vote(subjs_task, y_task, oof_b1)[0], subject_soft_vote(subjs_task, y_task, oof_b1)[1].argmax(1), average='macro'):.3f} | "
                f"B2a={f1_score(subject_soft_vote(subjs_task, y_task, oof_b2a)[0], subject_soft_vote(subjs_task, y_task, oof_b2a)[1].argmax(1), average='macro'):.3f} | "
                f"B2b={f1_score(subject_soft_vote(subjs_task, y_task, oof_b2b)[0], subject_soft_vote(subjs_task, y_task, oof_b2b)[1].argmax(1), average='macro'):.3f} | "
                f"B3={f1_score(subject_soft_vote(subjs_task, y_task, oof_b3)[0], subject_soft_vote(subjs_task, y_task, oof_b3)[1].argmax(1), average='macro'):.3f}"
            )

        del raw_task, nf_task, w_task, p_task, tab_task
        gc.collect()

    raw_df = pd.DataFrame(records)
    raw_df.to_csv(OUT_DIR / "metrics_per_seed.csv", index=False)

    agg_df = (
        raw_df.groupby(["task", "level", "experiment"], sort=False)
        .agg(
            n_seeds=("seed", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
        )
        .reset_index()
    )
    summary_csv = OUT_DIR / "metrics_summary.csv"
    agg_df.to_csv(summary_csv, index=False)
    print(f"\n[saved] {summary_csv} (total runtime={time.time() - t_start:.1f}s)")

    # Generate REPORT_SUMMARY.md
    sota_2026 = {"AD_vs_CN": 0.7582, "AD_vs_FTD": 0.7065, "FTD_vs_CN": 0.5635}
    lines = [
        "# SPEC B Experiment Report: Strict Subject-Independent AHEPA (ds004504) Classification",
        "",
        "## 1. Subject-Level Performance (Soft Voting Across 4s Epochs, Mean ± Std)",
        "",
        "| Task | Experiment | Macro-F1 (%) | Accuracy (%) | AUROC | Sens / Spec (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    sub_df = agg_df[agg_df["level"] == "Subject"]
    for row in sub_df.itertuples(index=False):
        f1_str = f"{row.macro_f1_mean*100:.2f} ± {row.macro_f1_std*100:.2f}"
        acc_str = f"{row.accuracy_mean*100:.2f} ± {row.accuracy_std*100:.2f}"
        auc_str = f"{row.auroc_mean:.4f} ± {row.auroc_std:.4f}"
        ss_str = (
            f"{row.sensitivity_mean*100:.1f} / {row.specificity_mean*100:.1f}"
            if np.isfinite(row.sensitivity_mean)
            else "N/A (3-class)"
        )
        lines.append(f"| {row.task} | {row.experiment} | {f1_str} | {acc_str} | {auc_str} | {ss_str} |")

    lines.extend(
        [
            "",
            "## 2. Segment-Level Performance (Mean ± Std)",
            "",
            "| Task | Experiment | Macro-F1 (%) | Accuracy (%) | AUROC |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ]
    )
    seg_df = agg_df[agg_df["level"] == "Segment"]
    for row in seg_df.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.experiment} | "
            f"{row.macro_f1_mean*100:.2f} ± {row.macro_f1_std*100:.2f} | "
            f"{row.accuracy_mean*100:.2f} ± {row.accuracy_std*100:.2f} | "
            f"{row.auroc_mean:.4f} ± {row.auroc_std:.4f} |"
        )

    lines.extend(["", "## 3. Comparison vs. 2026 Literature Subject-Level Macro-F1 Benchmarks", ""])
    for task_name, ref in sota_2026.items():
        b0 = sub_df[(sub_df["task"] == task_name) & (sub_df["experiment"].str.startswith("Exp B0"))].iloc[0]
        b1 = sub_df[(sub_df["task"] == task_name) & (sub_df["experiment"].str.startswith("Exp B1"))].iloc[0]
        b2a = sub_df[(sub_df["task"] == task_name) & (sub_df["experiment"].str.startswith("Exp B2a"))].iloc[0]
        b2b = sub_df[(sub_df["task"] == task_name) & (sub_df["experiment"].str.startswith("Exp B2b"))].iloc[0]
        b3 = sub_df[(sub_df["task"] == task_name) & (sub_df["experiment"].str.startswith("Exp B3"))].iloc[0]
        leak_drop = (b0["macro_f1_mean"] - b1["macro_f1_mean"]) * 100.0
        delta_b3 = (b3["macro_f1_mean"] - ref) * 100.0
        lines.append(
            f"- **{task_name}** (2026 SOTA = {ref*100:.2f}%): "
            f"B0 Leaky = {b0['macro_f1_mean']*100:.2f}% -> B1 Strict = {b1['macro_f1_mean']*100:.2f}% (leakage collapse = -{leak_drop:.2f} pp) | "
            f"B2a LightGBM = {b2a['macro_f1_mean']*100:.2f}% | B2b Linear SVM = {b2b['macro_f1_mean']*100:.2f}% | "
            f"B3 LightGAT = {b3['macro_f1_mean']*100:.2f}% ({delta_b3:+.2f} pp vs. 2026 SOTA)."
        )

    lines.extend(
        [
            "",
            "## 4. Output Figures",
            "",
            "- `outputs/figures/fig1_leakage_gap.png`",
            "- `outputs/figures/fig2_topomap_plv.png`",
            "- `outputs/figures/fig3_f1_comparison.png`",
            "",
        ]
    )
    report_path = OUT_DIR / "REPORT_SUMMARY.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}")
    print(sub_df[["task", "experiment", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "auroc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
