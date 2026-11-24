#!/usr/bin/env python3
"""SPEC B-V2 Rescue Benchmarks: Riemannian, Neuro-Biomarkers, Spectral ResGCN (<25K), Ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time
import warnings

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SRC_V2 = Path(__file__).resolve().parent
V2_ROOT = SRC_V2.parent
V1_ROOT = V2_ROOT.parent / "proj_b_ahepa_ad_ftd"
CACHE_DIR = V2_ROOT / "data" / "cache_v2"
OUT_DIR = V2_ROOT / "outputs"

SEEDS = [42, 100, 2024, 2025, 2026]
TASKS = [
    ("AD_vs_CN", ["A", "C"], ["C", "A"]),   # C=0 (neg), A=1 (pos)
    ("AD_vs_FTD", ["A", "F"], ["F", "A"]),  # F=0 (FTD, Spec), A=1 (AD, Sens)
    ("FTD_vs_CN", ["F", "C"], ["C", "F"]),  # C=0 (neg), F=1 (pos)
    ("AD_vs_FTD_vs_CN", ["A", "F", "C"], ["C", "F", "A"]),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpectralResGCNLayer(nn.Module):
    """Residual Graph Convolution layer with LayerNorm, GELU, Dropout."""

    def __init__(self, dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # norm_adj: (B, 19, 19), h: (B, 19, D)
        agg = torch.bmm(norm_adj, self.lin(h))
        return F.gelu(self.norm(h + self.drop(agg)))


class UltraLightSpectralResGCN(nn.Module):
    """Stabilized <25K-param 2-layer Residual GCN over 20s Theta/Alpha wPLI brain graphs."""

    def __init__(self, in_feats: int = 8, hidden: int = 24, n_nodes: int = 19, n_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, n_nodes, hidden) * 0.02)
        self.band_logits = nn.Parameter(torch.zeros(2))  # learnable blend of Theta & Alpha wPLI
        self.gcn1 = SpectralResGCNLayer(hidden, dropout=dropout)
        self.gcn2 = SpectralResGCNLayer(hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )
        self.register_buffer("eye", torch.eye(n_nodes), persistent=False)

        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < 25000, f"ResGCN exceeds 25K params: {n_params}"

    def _norm_adj(self, wpli: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.band_logits, dim=0)
        a = w[0] * wpli[:, 0] + w[1] * wpli[:, 1] + self.eye.unsqueeze(0)
        deg = a.sum(dim=-1).clamp(min=1e-6).pow(-0.5)
        return deg.unsqueeze(-1) * a * deg.unsqueeze(-2)

    def forward(self, nf: torch.Tensor, wpli: torch.Tensor) -> torch.Tensor:
        h = self.node_enc(nf) + self.pos_emb
        adj = self._norm_adj(wpli)
        h = self.gcn1(h, adj)
        h = self.gcn2(h, adj)
        g = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(g)


def tune_binary_threshold(y_true: np.ndarray, pos_probs: np.ndarray) -> float:
    """Strictly choose threshold on inner-CV OOF train probabilities maximizing Macro-F1."""
    best_th = 0.50
    best_f1 = -1.0
    for th in np.linspace(0.25, 0.75, 51):
        preds = (pos_probs >= th).astype(np.int64)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) <= 1e-9 and abs(th - 0.5) < abs(best_th - 0.5)):
            best_f1 = f1
            best_th = float(th)
    return best_th


def eval_subject_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int, threshold: float = 0.5) -> dict[str, float]:
    if n_classes == 2:
        preds = (probs[:, 1] >= threshold).astype(np.int64)
        try:
            auroc = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            auroc = float("nan")
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / max(1, tp + fn))
        spec = float(tn / max(1, tn + fp))
    else:
        preds = probs.argmax(axis=1)
        try:
            auroc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auroc = float("nan")
        sens = float("nan")
        spec = float("nan")

    return {
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "auroc": auroc,
        "sensitivity": sens,
        "specificity": spec,
    }


def _subject_tangent_fit_transform(
    cov_tr: np.ndarray, cov_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit TangentSpace strictly on train-fold SPD covariances per band and concatenate."""
    tr_feats, te_feats = [], []
    for b in range(cov_tr.shape[1]):
        ts = TangentSpace(metric="riemann")
        tr_feats.append(ts.fit_transform(cov_tr[:, b].astype(np.float64)))
        te_feats.append(ts.transform(cov_te[:, b].astype(np.float64)))
    return np.concatenate(tr_feats, axis=1), np.concatenate(te_feats, axis=1)


def _predict_b4a_fold(
    cov_w_tr: np.ndarray,
    w_sub_tr: np.ndarray,
    y_w_tr: np.ndarray,
    cov_w_te: np.ndarray,
    w_sub_te: np.ndarray,
    tr_subs: np.ndarray,
    te_subs: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exp B4a: Multi-band epoch Riemannian Tangent Space -> LogReg -> subject soft vote."""
    x_tr, x_te = _subject_tangent_fit_transform(cov_w_tr, cov_w_te)
    sc = StandardScaler()
    x_tr_s = sc.fit_transform(x_tr)
    x_te_s = sc.transform(x_te)
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(x_tr_s, y_w_tr)
    p_w_tr = clf.predict_proba(x_tr_s)
    p_w_te = clf.predict_proba(x_te_s)
    p_sub_tr = np.vstack([p_w_tr[w_sub_tr == s].mean(axis=0) for s in tr_subs])
    p_sub_te = np.vstack([p_w_te[w_sub_te == s].mean(axis=0) for s in te_subs])
    return p_sub_tr, p_sub_te


def _predict_b4b_fold(
    cov_s_tr: np.ndarray, y_s_tr: np.ndarray, cov_s_te: np.ndarray, n_classes: int, seed: int
) -> np.ndarray:
    """Exp B4b: Subject-level Riemannian Mean Covariance Tangent Space -> LogReg + LinearSVC."""
    x_tr, x_te = _subject_tangent_fit_transform(cov_s_tr, cov_s_te)
    sc = StandardScaler()
    x_tr_s = sc.fit_transform(x_tr)
    x_te_s = sc.transform(x_te)

    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=seed)
    lr.fit(x_tr_s, y_s_tr)
    p_lr = lr.predict_proba(x_te_s)

    svc = CalibratedClassifierCV(
        LinearSVC(C=0.05, class_weight="balanced", dual=False, max_iter=3000, random_state=seed),
        cv=min(3, int(np.bincount(y_s_tr).min())),
        method="sigmoid",
    )
    svc.fit(x_tr_s, y_s_tr)
    p_svc = svc.predict_proba(x_te_s)
    return 0.5 * p_lr + 0.5 * p_svc


def _predict_b5a_fold(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int
) -> np.ndarray:
    """Exp B5a: Subject-level Neuro-Biomarkers -> Balanced ElasticNet LogReg."""
    sc = StandardScaler()
    x_tr_s = sc.fit_transform(x_tr)
    x_te_s = sc.transform(x_te)
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.2,
        class_weight="balanced", max_iter=3000, random_state=seed,
    )
    clf.fit(x_tr_s, y_tr)
    return clf.predict_proba(x_te_s)


def _predict_b5b_fold(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, seed: int
) -> np.ndarray:
    """Exp B5b: Subject-level Neuro-Biomarkers -> Balanced Regularized LightGBM."""
    sc = StandardScaler()
    x_tr_s = sc.fit_transform(x_tr)
    x_te_s = sc.transform(x_te)
    lgbm = LGBMClassifier(
        n_estimators=100, max_depth=3, num_leaves=7, min_child_samples=8,
        subsample=0.8, colsample_bytree=0.7, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(x_tr_s, y_tr)
    return lgbm.predict_proba(x_te_s)


def _predict_b6_fold(
    nf_w_tr: np.ndarray,
    wpli_w_tr: np.ndarray,
    w_sub_tr: np.ndarray,
    y_w_tr: np.ndarray,
    nf_w_te: np.ndarray,
    wpli_w_te: np.ndarray,
    w_sub_te: np.ndarray,
    tr_subs: np.ndarray,
    te_subs: np.ndarray,
    n_classes: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Exp B6: Ultra-Light Spectral ResGCN (<25K params) with inner-val early stopping."""
    set_seed(seed)
    # Split train subjects into inner train / inner val (80/20) for early stopping without peeking at test
    tr_y_sub = np.array([int(y_w_tr[w_sub_tr == s][0]) for s in tr_subs], dtype=np.int64)
    in_skf = StratifiedKFold(n_splits=min(4, max(2, int(np.bincount(tr_y_sub).min()))), shuffle=True, random_state=seed)
    in_tr_i, in_va_i = next(in_skf.split(tr_subs, tr_y_sub))
    in_tr_subs = set(tr_subs[in_tr_i])
    in_va_subs = set(tr_subs[in_va_i])
    assert in_tr_subs.isdisjoint(in_va_subs)

    m_in_tr = np.isin(w_sub_tr, list(in_tr_subs))
    m_in_va = np.isin(w_sub_tr, list(in_va_subs))

    # Scale node features strictly on inner-train
    n_f = nf_w_tr.shape[-1]
    sc = StandardScaler()
    nf_in_tr = sc.fit_transform(nf_w_tr[m_in_tr].reshape(-1, n_f)).reshape(-1, 19, n_f)
    nf_in_va = sc.transform(nf_w_tr[m_in_va].reshape(-1, n_f)).reshape(-1, 19, n_f)
    nf_te_s = sc.transform(nf_w_te.reshape(-1, n_f)).reshape(-1, 19, n_f)

    x_tr_t = torch.from_numpy(nf_in_tr.astype(np.float32)).to(device)
    a_tr_t = torch.from_numpy(wpli_w_tr[m_in_tr].astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_w_tr[m_in_tr].astype(np.int64)).to(device)

    x_va_t = torch.from_numpy(nf_in_va.astype(np.float32)).to(device)
    a_va_t = torch.from_numpy(wpli_w_tr[m_in_va].astype(np.float32)).to(device)
    y_va_t = torch.from_numpy(y_w_tr[m_in_va].astype(np.int64)).to(device)

    x_te_t = torch.from_numpy(nf_te_s.astype(np.float32)).to(device)
    a_te_t = torch.from_numpy(wpli_w_te.astype(np.float32)).to(device)

    counts = np.bincount(y_w_tr[m_in_tr], minlength=n_classes).astype(np.float64)
    cw = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    model = UltraLightSpectralResGCN(in_feats=n_f, hidden=24, n_nodes=19, n_classes=n_classes, dropout=0.5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_imp = 0
    bs = 64

    for ep in range(35):
        model.train()
        perm = torch.randperm(x_tr_t.size(0), device=device)
        for s in range(0, x_tr_t.size(0), bs):
            idx = perm[s : s + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(x_tr_t[idx], a_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            assert torch.isfinite(loss), "Non-finite ResGCN loss"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(x_va_t, a_va_t)
            va_loss = float(criterion(va_logits, y_va_t).item())
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 10 and ep >= 12:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        te_logits = model(x_te_t, a_te_t)
        p_w_te = F.softmax(te_logits, dim=-1).cpu().numpy()

    p_sub_te = np.vstack([p_w_te[w_sub_te == s].mean(axis=0) for s in te_subs])
    return p_sub_te


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-name", type=str, default="ahepa_v2_features.npz")
    args = parser.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    d = np.load(CACHE_DIR / args.cache_name, allow_pickle=False)
    x_neuro_subj = d["x_neuro_subj"]
    x_cov_subj = d["x_cov_subj"]
    x_cov_win = d["x_cov_win"]
    gcn_nf = d["gcn_node_feats"]
    wpli_20s = d["wpli_20s"]
    win_subj_id = d["win_subj_id"]
    subj_ids = d["subj_ids"]
    subj_grps = d["subj_grps"]

    for nc in [2, 3]:
        m_chk = UltraLightSpectralResGCN(in_feats=gcn_nf.shape[-1], n_classes=nc)
        print(f"[model] UltraLightSpectralResGCN (classes={nc}) params = {sum(p.numel() for p in m_chk.parameters()):,}")

    records = []
    model_names = [
        "Exp B4a (Riemannian Epoch)",
        "Exp B4b (Riemannian Subject)",
        "Exp B5a (Neuro ElasticNet)",
        "Exp B5b (Neuro LightGBM)",
        "Exp B6 (Spectral ResGCN <25K)",
        "Exp B7 (Multimodal Ensemble)",
    ]

    for task_name, valid_grps, label_order in TASKS:
        s_mask = np.isin(subj_grps, valid_grps)
        t_subj_ids = subj_ids[s_mask]
        label_map = {g: i for i, g in enumerate(label_order)}
        y_subj = np.array([label_map[g] for g in subj_grps[s_mask]], dtype=np.int64)
        xn_task = x_neuro_subj[s_mask]
        xc_subj_task = x_cov_subj[s_mask]

        w_mask = np.isin(win_subj_id, t_subj_ids)
        w_subs = win_subj_id[w_mask]
        y_win = np.array([label_map[subj_grps[subj_ids == s][0]] for s in w_subs], dtype=np.int64)
        xc_win_task = x_cov_win[w_mask]
        gnf_task = gcn_nf[w_mask]
        wpli_task = wpli_20s[w_mask]
        n_classes = len(label_order)

        print(f"\n==================== TASK: {task_name} (subjects={len(t_subj_ids)}, 20s_windows={len(y_win)}) ====================")

        for seed in SEEDS:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            oof_probs = {m: np.zeros((len(t_subj_ids), n_classes), dtype=np.float64) for m in model_names}
            oof_preds = {m: np.zeros(len(t_subj_ids), dtype=np.int64) for m in model_names}

            # Subject-level StratifiedGroupKFold (each subject is its own group)
            for fold, (tr_i, te_i) in enumerate(sgkf.split(xn_task, y_subj, groups=t_subj_ids)):
                tr_subs = t_subj_ids[tr_i]
                te_subs = t_subj_ids[te_i]
                assert set(tr_subs).isdisjoint(set(te_subs)), "Subject leakage in outer fold!"

                w_tr_m = np.isin(w_subs, tr_subs)
                w_te_m = np.isin(w_subs, te_subs)
                fseed = seed * 100 + fold

                # --- Inner 4-fold CV strictly inside tr_subs to calibrate binary thresholds ---
                if n_classes == 2:
                    in_cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fseed)
                    in_oof = {m: np.zeros((len(tr_subs), 2), dtype=np.float64) for m in model_names}
                    for infold, (itr_i, iva_i) in enumerate(in_cv.split(xn_task[tr_i], y_subj[tr_i], groups=tr_subs)):
                        itr_subs = tr_subs[itr_i]
                        iva_subs = tr_subs[iva_i]
                        assert set(itr_subs).isdisjoint(set(iva_subs))
                        iw_tr = np.isin(w_subs, itr_subs)
                        iw_va = np.isin(w_subs, iva_subs)
                        ifseed = fseed * 10 + infold

                        _, in_oof["Exp B4a (Riemannian Epoch)"][iva_i] = _predict_b4a_fold(
                            xc_win_task[iw_tr], w_subs[iw_tr], y_win[iw_tr],
                            xc_win_task[iw_va], w_subs[iw_va], itr_subs, iva_subs, n_classes, ifseed,
                        )
                        in_oof["Exp B4b (Riemannian Subject)"][iva_i] = _predict_b4b_fold(
                            xc_subj_task[tr_i][itr_i], y_subj[tr_i][itr_i], xc_subj_task[tr_i][iva_i], n_classes, ifseed
                        )
                        in_oof["Exp B5a (Neuro ElasticNet)"][iva_i] = _predict_b5a_fold(
                            xn_task[tr_i][itr_i], y_subj[tr_i][itr_i], xn_task[tr_i][iva_i], ifseed
                        )
                        in_oof["Exp B5b (Neuro LightGBM)"][iva_i] = _predict_b5b_fold(
                            xn_task[tr_i][itr_i], y_subj[tr_i][itr_i], xn_task[tr_i][iva_i], ifseed
                        )
                        in_oof["Exp B6 (Spectral ResGCN <25K)"][iva_i] = _predict_b6_fold(
                            gnf_task[iw_tr], wpli_task[iw_tr], w_subs[iw_tr], y_win[iw_tr],
                            gnf_task[iw_va], wpli_task[iw_va], w_subs[iw_va],
                            itr_subs, iva_subs, n_classes, ifseed, device,
                        )
                    in_oof["Exp B7 (Multimodal Ensemble)"] = (
                        0.4 * in_oof["Exp B4b (Riemannian Subject)"]
                        + 0.4 * in_oof["Exp B5a (Neuro ElasticNet)"]
                        + 0.2 * in_oof["Exp B6 (Spectral ResGCN <25K)"]
                    )
                    ths = {m: tune_binary_threshold(y_subj[tr_i], in_oof[m][:, 1]) for m in model_names}
                else:
                    ths = {m: 0.5 for m in model_names}

                # --- Fit on full outer train fold and predict outer test fold ---
                _, p_b4a = _predict_b4a_fold(
                    xc_win_task[w_tr_m], w_subs[w_tr_m], y_win[w_tr_m],
                    xc_win_task[w_te_m], w_subs[w_te_m], tr_subs, te_subs, n_classes, fseed,
                )
                p_b4b = _predict_b4b_fold(xc_subj_task[tr_i], y_subj[tr_i], xc_subj_task[te_i], n_classes, fseed)
                p_b5a = _predict_b5a_fold(xn_task[tr_i], y_subj[tr_i], xn_task[te_i], fseed)
                p_b5b = _predict_b5b_fold(xn_task[tr_i], y_subj[tr_i], xn_task[te_i], fseed)
                p_b6 = _predict_b6_fold(
                    gnf_task[w_tr_m], wpli_task[w_tr_m], w_subs[w_tr_m], y_win[w_tr_m],
                    gnf_task[w_te_m], wpli_task[w_te_m], w_subs[w_te_m],
                    tr_subs, te_subs, n_classes, fseed, device,
                )
                p_b7 = 0.4 * p_b4b + 0.4 * p_b5a + 0.2 * p_b6

                fold_probs = {
                    "Exp B4a (Riemannian Epoch)": p_b4a,
                    "Exp B4b (Riemannian Subject)": p_b4b,
                    "Exp B5a (Neuro ElasticNet)": p_b5a,
                    "Exp B5b (Neuro LightGBM)": p_b5b,
                    "Exp B6 (Spectral ResGCN <25K)": p_b6,
                    "Exp B7 (Multimodal Ensemble)": p_b7,
                }
                for m, pm in fold_probs.items():
                    oof_probs[m][te_i] = pm
                    if n_classes == 2:
                        oof_preds[m][te_i] = (pm[:, 1] >= ths[m]).astype(np.int64)
                    else:
                        oof_preds[m][te_i] = pm.argmax(axis=1)

            # Evaluate full OOF across all subjects for this repeat seed
            summary_str = []
            for m in model_names:
                pm = oof_probs[m]
                pr = oof_preds[m]
                macro_f1 = float(f1_score(y_subj, pr, average="macro", zero_division=0))
                acc = float(accuracy_score(y_subj, pr))
                if n_classes == 2:
                    auroc = float(roc_auc_score(y_subj, pm[:, 1]))
                    tn, fp, fn, tp = confusion_matrix(y_subj, pr, labels=[0, 1]).ravel()
                    sens = float(tp / max(1, tp + fn))
                    spec = float(tn / max(1, tn + fp))
                else:
                    auroc = float(roc_auc_score(y_subj, pm, multi_class="ovr", average="macro"))
                    sens, spec = float("nan"), float("nan")
                records.append(
                    {
                        "task": task_name,
                        "experiment": m,
                        "seed": seed,
                        "macro_f1": macro_f1,
                        "accuracy": acc,
                        "auroc": auroc,
                        "sensitivity": sens,
                        "specificity": spec,
                    }
                )
                short = m.split()[1]
                summary_str.append(f"{short}={macro_f1:.3f}")
            print(f"  [{task_name} seed={seed}] Subject Macro-F1 -> " + " | ".join(summary_str))

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(OUT_DIR / "rescue_metrics_per_seed.csv", index=False)

    df_agg = (
        df_raw.groupby(["task", "experiment"], sort=False)
        .agg(
            n_seeds=("seed", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
        )
        .reset_index()
    )
    df_agg.to_csv(OUT_DIR / "rescue_metrics_summary.csv", index=False)
    print(f"\n[saved] rescue_metrics_summary.csv in {time.time() - t0:.1f}s")

    # Load V1 summary if available for comparison table
    v1_csv = V1_ROOT / "outputs" / "metrics_summary.csv"
    v1_sub = None
    if v1_csv.exists():
        v1_df = pd.read_csv(v1_csv)
        v1_sub = v1_df[v1_df["level"] == "Subject"]

    sota_2026 = {"AD_vs_CN": 75.82, "AD_vs_FTD": 70.65, "FTD_vs_CN": 56.35}
    lines = [
        "# SPEC B-V2 Report: Rescuing Strict Subject-Independent AD vs. FTD Classification",
        "",
        "## 1. V2 Subject-Level Performance (Strict 5-Fold StratifiedGroupKFold × 5 Seeds)",
        "",
        "| Task | Experiment | Macro-F1 (%) | Accuracy (%) | AUROC | Sens (%) | Spec (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in df_agg.itertuples(index=False):
        s_str = f"{r.sensitivity_mean*100:.1f} ± {r.sensitivity_std*100:.1f}" if np.isfinite(r.sensitivity_mean) else "N/A"
        sp_str = f"{r.specificity_mean*100:.1f} ± {r.specificity_std*100:.1f}" if np.isfinite(r.specificity_mean) else "N/A"
        lines.append(
            f"| {r.task} | {r.experiment} | {r.macro_f1_mean*100:.2f} ± {r.macro_f1_std*100:.2f} | "
            f"{r.accuracy_mean*100:.2f} ± {r.accuracy_std*100:.2f} | {r.auroc_mean:.4f} ± {r.auroc_std:.4f} | {s_str} | {sp_str} |"
        )

    if v1_sub is not None:
        lines.extend(["", "## 2. Direct V1 vs. V2 & 2026 SOTA Comparison (Subject Macro-F1 %)", ""])
        for tname in ["AD_vs_CN", "AD_vs_FTD", "FTD_vs_CN", "AD_vs_FTD_vs_CN"]:
            v1_rows = v1_sub[(v1_sub["task"] == tname) & (~v1_sub["experiment"].str.startswith("Exp B0"))]
            v1_best = v1_rows.sort_values("macro_f1_mean", ascending=False).iloc[0]
            v2_rows = df_agg[df_agg["task"] == tname]
            v2_best = v2_rows.sort_values("macro_f1_mean", ascending=False).iloc[0]
            ref_txt = f" (2026 SOTA = {sota_2026[tname]:.2f}%)" if tname in sota_2026 else ""
            delta = (v2_best["macro_f1_mean"] - float(v1_best["macro_f1_mean"])) * 100.0
            lines.append(
                f"- **{tname}**{ref_txt}: V1 Best `{v1_best['experiment']}` = **{float(v1_best['macro_f1_mean'])*100:.2f}%** -> "
                f"V2 Best `{v2_best['experiment']}` = **{v2_best['macro_f1_mean']*100:.2f}%** (**{delta:+.2f} pp vs. V1**)."
            )

    lines.extend(
        [
            "",
            "## 3. Output Figures",
            "",
            "- `outputs/figures/fig4_ad_ftd_rescue.png`",
            "- `outputs/figures/fig5_v2_all_tasks_vs_2026sota.png`",
            "- `outputs/figures/fig6_neuro_biomarkers_boxplot.png`",
            "",
        ]
    )
    (OUT_DIR / "REPORT_SUMMARY_V2.md").write_text("\n".join(lines), encoding="utf-8")
    print("[saved] REPORT_SUMMARY_V2.md")
    print(df_agg[["task", "experiment", "macro_f1_mean", "accuracy_mean", "auroc_mean", "sensitivity_mean", "specificity_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
