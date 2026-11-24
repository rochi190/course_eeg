#!/usr/bin/env python3
"""SPEC B-V2 Feature Extraction: 20s Neuro-Biomarkers (incl. 1/f slope, debiased wPLI) & Multi-Band Riemannian SPD Covariances."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from pathlib import Path
import time
import warnings

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import mne
import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.linear_model import HuberRegressor
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

SRC_V2 = Path(__file__).resolve().parent
V2_ROOT = SRC_V2.parent
V1_ROOT = V2_ROOT.parent / "proj_b_ahepa_ad_ftd"
BIDS_DIR = V2_ROOT / "data" / "ds004504"
V1_BIDS_DIR = V1_ROOT / "data" / "ds004504"
CACHE_DIR = V2_ROOT / "data" / "cache_v2"

BUCKET = "openneuro.org"
PREFIX = "ds004504"

LEGACY_RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
CANONICAL_CHS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]
ROI_MAP = {
    "F": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "C": ["C3", "Cz", "C4"],
    "T": ["T7", "T8"],
    "PO": ["P7", "P3", "Pz", "P4", "P8", "O1", "O2"],
}
ROI_NAMES = ["F", "C", "T", "PO"]
ROI_IDX = {r: [CANONICAL_CHS.index(c) for c in chs] for r, chs in ROI_MAP.items()}

PSD_BANDS = [
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 25.0),
    ("Gamma", 25.0, 45.0),
]
COV_BANDS = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 25.0),
    ("Broad", 1.0, 40.0),
]

FS = 250
WIN_LEN = 5000   # 20s @ 250 Hz
STRIDE = 2500    # 10s stride (50% overlap)


def _make_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def ensure_bids_data(workers: int = 8) -> Path:
    """Reuse V1 BIDS dir if complete, otherwise download participants.tsv + 88 .set files."""
    if (V1_BIDS_DIR / "participants.tsv").exists():
        n_v1 = len(list((V1_BIDS_DIR / "derivatives").glob("sub-*/eeg/*_eeg.set")))
        if n_v1 == 88:
            print(f"[data] Reusing complete V1 BIDS dataset ({n_v1} subjects) at {V1_BIDS_DIR}")
            return V1_BIDS_DIR

    BIDS_DIR.mkdir(parents=True, exist_ok=True)
    s3 = _make_s3()
    tsv_path = BIDS_DIR / "participants.tsv"
    if not tsv_path.exists():
        s3.download_file(BUCKET, f"{PREFIX}/participants.tsv", str(tsv_path))
    df = pd.read_csv(tsv_path, sep="\t")

    paginator = s3.get_paginator("list_objects_v2")
    set_map: dict[str, tuple[str, int]] = {}
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/derivatives/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("_eeg.set"):
                sub = next(p for p in key.split("/") if p.startswith("sub-"))
                set_map[sub] = (key, int(item["Size"]))

    def _dl(sub: str) -> str:
        key, size = set_map[sub]
        dst = BIDS_DIR / key[len(PREFIX) + 1 :]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == size:
            return f"cached:{sub}"
        tmp = dst.with_suffix(".set.part")
        _make_s3().download_file(BUCKET, key, str(tmp))
        assert tmp.stat().st_size == size
        tmp.replace(dst)
        return f"dl:{sub}"

    subs = df["participant_id"].tolist()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_dl, s) for s in subs]
        for idx, f in enumerate(as_completed(futs), 1):
            f.result()
            if idx % 20 == 0 or idx == len(subs):
                print(f"[download] {idx}/{len(subs)} subjects ready")
    return BIDS_DIR


def fit_huber_1f_vectorized(
    log_freqs: np.ndarray, log_psd: np.ndarray, n_iter: int = 15, eps: float = 1.35
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized IRLS Huber regression for y = slope * x + intercept.
    log_freqs: (F,), log_psd: (M, F). Returns (slope (M,), intercept (M,)).
    """
    x = log_freqs.astype(np.float64)
    y = log_psd.astype(np.float64)
    M, F = y.shape

    # Initialize via OLS
    xm = x.mean()
    xc = x - xm
    slope = np.sum(xc[None, :] * (y - y.mean(axis=1, keepdims=True)), axis=1) / (np.sum(xc ** 2) + 1e-12)
    intercept = y.mean(axis=1) - slope * xm

    for _ in range(n_iter):
        pred = slope[:, None] * x[None, :] + intercept[:, None]
        res = y - pred
        # Robust scale MAD
        med = np.median(res, axis=1, keepdims=True)
        mad = np.median(np.abs(res - med), axis=1, keepdims=True) / 0.6745 + 1e-8
        r_norm = np.abs(res) / mad
        w = np.where(r_norm <= eps, 1.0, eps / (r_norm + 1e-12))
        w_sum = np.sum(w, axis=1) + 1e-12
        xw = np.sum(w * x[None, :], axis=1) / w_sum
        yw = np.sum(w * y, axis=1) / w_sum
        dx = x[None, :] - xw[:, None]
        dy = y - yw[:, None]
        slope = np.sum(w * dx * dy, axis=1) / (np.sum(w * dx * dx, axis=1) + 1e-12)
        intercept = yw - slope * xw

    return slope.astype(np.float32), intercept.astype(np.float32)


def self_test_huber() -> None:
    """Cross-check vectorized Huber IRLS against sklearn.linear_model.HuberRegressor."""
    rng = np.random.default_rng(7)
    freqs = np.linspace(2.0, 40.0, 80)
    mask = (freqs <= 7.0) | (freqs >= 14.0)
    lf = np.log10(freqs[mask])
    y = -1.6 * lf[None, :] + 2.3 + rng.normal(0, 0.05, size=(12, len(lf)))
    y[0, 5] += 1.5  # outlier
    sl_v, int_v = fit_huber_1f_vectorized(lf, y)
    for i in range(12):
        hr = HuberRegressor(epsilon=1.35, max_iter=300).fit(lf[:, None], y[i])
        assert abs(sl_v[i] - hr.coef_[0]) < 0.05, (sl_v[i], hr.coef_[0])
        assert abs(int_v[i] - hr.intercept_) < 0.08, (int_v[i], hr.intercept_)
    print("[self-test PASS] Vectorized 1/f Huber regression matches sklearn HuberRegressor")


def load_subject_20s_windows(set_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Load .set, map canonical 19 channels, filter 0.5-45 Hz @ 250 Hz, cut 20s 50%-overlap windows."""
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    ren = {k: v for k, v in LEGACY_RENAME.items() if k in raw.ch_names}
    if ren:
        raw.rename_channels(ren)
    raw.pick(CANONICAL_CHS)
    assert raw.ch_names == CANONICAL_CHS, f"Channel mismatch: {raw.ch_names}"
    raw.set_montage("standard_1020", on_missing="raise")

    raw.filter(l_freq=0.5, h_freq=45.0, method="fir", fir_design="firwin", verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - FS) > 1e-3:
        raw.resample(FS, npad="auto", verbose="ERROR")

    data = raw.get_data()
    if float(np.median(np.std(data, axis=1))) < 1e-2:
        data = data * 1e6  # V -> uV
    data = data.astype(np.float32)

    n_pts = data.shape[1]
    starts = list(range(0, n_pts - WIN_LEN + 1, STRIDE))
    assert len(starts) >= 5, f"Too few 20s windows in {set_path}: {len(starts)}"
    wins = np.stack([data[:, s : s + WIN_LEN] for s in starts], axis=0)  # (W, 19, 5000)

    # Reject extreme artifacts (> median + 5 * IQR or > 1000 uV or flat)
    ptp = np.ptp(wins, axis=-1).max(axis=-1)
    ch_std = np.std(wins, axis=-1).min(axis=-1)
    q75, q25 = np.percentile(ptp, [75, 25])
    iqr = max(float(q75 - q25), 10.0)
    thresh = min(1000.0, float(np.median(ptp)) + 5.0 * iqr)
    keep = (ptp <= thresh) & (ch_std >= 1e-3)
    if int(keep.sum()) < 5:
        best_idx = np.argsort(ptp)[: min(5, len(wins))]
        keep_wins = wins[best_idx]
    else:
        keep_wins = wins[keep]
    return np.ascontiguousarray(keep_wins, dtype=np.float32), {"total": len(wins), "kept": len(keep_wins)}


def compute_20s_wpli_batched(
    wins: np.ndarray, device: torch.device, batch_size: int = 32
) -> np.ndarray:
    """
    Compute stabilized debiased wPLI (Vinck et al. 2011) on 20s windows over 9 overlapping
    4s Hann-tapered sub-segments for Theta (4-8 Hz) and Alpha (8-13 Hz).
    Returns shape (N_win, 2, 19, 19) in [0, 1].
    """
    n_win = len(wins)
    wpli_out = np.zeros((n_win, 2, 19, 19), dtype=np.float32)
    seg_len = 1000   # 4s @ 250 Hz -> 0.25 Hz bin spacing
    hop = 500        # 9 sub-segments per 5000-sample 20s window
    freqs = np.fft.rfftfreq(seg_len, d=1.0 / FS)
    band_masks = [
        torch.from_numpy((freqs >= 4.0) & (freqs <= 8.0)).to(device),
        torch.from_numpy((freqs > 8.0) & (freqs <= 13.0)).to(device),
    ]
    hann = torch.hann_window(seg_len, device=device, dtype=torch.float32)
    eye_mask = ~torch.eye(19, dtype=torch.bool, device=device)

    for start in range(0, n_win, batch_size):
        end = min(n_win, start + batch_size)
        xb = torch.from_numpy(wins[start:end]).to(device=device, dtype=torch.float32)  # (B, 19, 5000)
        # Unfold into 9 segments: (B, 19, 9, 1000)
        segs = xb.unfold(-1, seg_len, hop) * hann
        spec = torch.fft.rfft(segs, dim=-1)  # (B, 19, 9, F)

        for b_idx, fmask in enumerate(band_masks):
            sb = spec[..., fmask]  # (B, 19, 9, Fb)
            # Flatten segments and freq bins within band as independent taper observations
            B_cur, C, S, Fb = sb.shape
            obs = sb.reshape(B_cur, C, S * Fb)  # (B, 19, K)
            sr = obs.real
            si = obs.imag
            # Im(z_i * conj(z_j)) = si_i * sr_j - sr_i * si_j -> shape (B, 19, 19, K)
            im = si.unsqueeze(2) * sr.unsqueeze(1) - sr.unsqueeze(2) * si.unsqueeze(1)
            sum_im = im.sum(dim=-1)
            sum_abs_im = im.abs().sum(dim=-1)
            sum_im2 = (im * im).sum(dim=-1)
            # Vinck et al. (2011) debiased squared wPLI -> take sqrt of positive part
            num = sum_im * sum_im - sum_im2
            den = sum_abs_im * sum_abs_im - sum_im2 + 1e-12
            w2 = torch.clamp(num / den, min=0.0, max=1.0)
            w = torch.sqrt(w2) * eye_mask.unsqueeze(0)
            wpli_out[start:end, b_idx] = w.cpu().numpy()

    assert np.isfinite(wpli_out).all()
    assert np.allclose(wpli_out, np.swapaxes(wpli_out, -2, -1), atol=1e-5)
    return wpli_out


def compute_neuro_features_chunked(
    wins: np.ndarray, wpli_20s: np.ndarray, chunk_size: int = 256
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Compute per-window X_neuro_win (N_win, 173), gcn_node_feats (N_win, 19, 8),
    and raw biomarker summary components per window for fig6.
    """
    n_win = len(wins)
    x_neuro_win = np.empty((n_win, 173), dtype=np.float32)
    gcn_feats = np.empty((n_win, 19, 8), dtype=np.float32)
    ap_alpha_arr = np.empty(n_win, dtype=np.float32)
    slowing_arr = np.empty(n_win, dtype=np.float32)
    slope_arr = np.empty(n_win, dtype=np.float32)

    roi_pairs = [(i, j) for i in range(4) for j in range(i, 4)]  # 10 pairs

    for start in range(0, n_win, chunk_size):
        end = min(n_win, start + chunk_size)
        B = end - start
        freqs, psd = welch(
            wins[start:end].astype(np.float64), fs=FS, nperseg=1000, noverlap=500, axis=-1
        )
        df_hz = freqs[1] - freqs[0]  # 0.25 Hz
        tot_mask = (freqs >= 0.5) & (freqs <= 45.0)
        tot_p = np.sum(psd[..., tot_mask], axis=-1, keepdims=True) * df_hz + 1e-12  # (B, 19, 1)

        ch_rel = []
        ch_abs = []
        for _, f_lo, f_hi in PSD_BANDS:
            m = (freqs >= f_lo) & (freqs <= f_hi) if f_lo == 0.5 else (freqs > f_lo) & (freqs <= f_hi)
            bp = np.sum(psd[..., m], axis=-1, keepdims=True) * df_hz
            ch_abs.append(bp)
            ch_rel.append(bp / tot_p)
        ch_rel_arr = np.concatenate(ch_rel, axis=-1).astype(np.float32)  # (B, 19, 5)
        ch_abs_arr = np.concatenate(ch_abs, axis=-1)  # (B, 19, 5)

        # ROI relative PSD: (B, 4, 5)
        roi_rel = np.stack([ch_rel_arr[:, ROI_IDX[r], :].mean(axis=1) for r in ROI_NAMES], axis=1)
        roi_abs = np.stack([ch_abs_arr[:, ROI_IDX[r], :].mean(axis=1) for r in ROI_NAMES], axis=1)

        # Anterior-Posterior log(F/PO) and Frontal-Temporal log(F/T) for Delta..Beta (idx 0..3)
        f_idx, t_idx, po_idx = 0, 2, 3
        ap_ratios = np.log((roi_abs[:, f_idx, :4] + 1e-12) / (roi_abs[:, po_idx, :4] + 1e-12)).astype(np.float32)  # (B, 4)
        ft_ratios = np.log((roi_abs[:, f_idx, :4] + 1e-12) / (roi_abs[:, t_idx, :4] + 1e-12)).astype(np.float32)  # (B, 4)

        # Slowing ratio: (Delta + Theta) / (Alpha + Beta)
        roi_slow = ((roi_abs[:, :, 0] + roi_abs[:, :, 1]) / (roi_abs[:, :, 2] + roi_abs[:, :, 3] + 1e-12)).astype(np.float32)  # (B, 4)
        glob_slow = roi_slow.mean(axis=1, keepdims=True)  # (B, 1)

        # IAF (center of gravity 7-13 Hz) per channel -> (B, 19) and per ROI -> (B, 4)
        iaf_m = (freqs >= 7.0) & (freqs <= 13.0)
        f_iaf = freqs[iaf_m][None, None, :]
        p_iaf = psd[..., iaf_m]
        ch_iaf = (np.sum(p_iaf * f_iaf, axis=-1) / (np.sum(p_iaf, axis=-1) + 1e-12)).astype(np.float32)  # (B, 19)
        roi_iaf = np.stack([ch_iaf[:, ROI_IDX[r]].mean(axis=1) for r in ROI_NAMES], axis=1)  # (B, 4)
        glob_iaf = ch_iaf.mean(axis=1, keepdims=True)

        # Spectral Entropy (1-45 Hz) per channel -> (B, 19) and per ROI -> (B, 4)
        ent_m = (freqs >= 1.0) & (freqs <= 45.0)
        p_ent = psd[..., ent_m]
        p_ent_n = p_ent / (np.sum(p_ent, axis=-1, keepdims=True) + 1e-12)
        ch_ent = (-np.sum(p_ent_n * np.log(p_ent_n + 1e-12), axis=-1) / np.log(p_ent_n.shape[-1])).astype(np.float32)
        roi_ent = np.stack([ch_ent[:, ROI_IDX[r]].mean(axis=1) for r in ROI_NAMES], axis=1)
        glob_ent = ch_ent.mean(axis=1, keepdims=True)

        # Aperiodic 1/f slope & intercept over 2-40 Hz excluding 7-14 Hz
        ap_m = ((freqs >= 2.0) & (freqs <= 7.0)) | ((freqs >= 14.0) & (freqs <= 40.0))
        lf = np.log10(freqs[ap_m])
        lpsd_ch = np.log10(psd[..., ap_m] + 1e-12)  # (B, 19, F_ap)
        ch_slope, _ = fit_huber_1f_vectorized(lf, lpsd_ch.reshape(B * 19, -1))
        ch_slope = ch_slope.reshape(B, 19)

        psd_roi = np.stack([psd[:, ROI_IDX[r], :].mean(axis=1) for r in ROI_NAMES], axis=1)  # (B, 4, F)
        lpsd_roi = np.log10(psd_roi[..., ap_m] + 1e-12)
        roi_slope, roi_off = fit_huber_1f_vectorized(lf, lpsd_roi.reshape(B * 4, -1))
        roi_slope = roi_slope.reshape(B, 4)
        roi_off = roi_off.reshape(B, 4)
        glob_slope = ch_slope.mean(axis=1, keepdims=True)

        # Stabilized wPLI ROI-to-ROI means (10 pairs x 2 bands = 20) + top-3 eigenvalues x 2 bands = 6
        w_cur = wpli_20s[start:end]  # (B, 2, 19, 19)
        roi_conn = []
        for b_i in range(2):
            for ri, rj in roi_pairs:
                sub_m = w_cur[:, b_i][:, ROI_IDX[ROI_NAMES[ri]], :][:, :, ROI_IDX[ROI_NAMES[rj]]]
                roi_conn.append(sub_m.mean(axis=(1, 2)))
        roi_conn_arr = np.stack(roi_conn, axis=1).astype(np.float32)  # (B, 20)

        eig_list = []
        for b_i in range(2):
            vals = np.linalg.eigvalsh(w_cur[:, b_i])  # ascending (B, 19)
            eig_list.append(vals[:, -3:][:, ::-1])
        eig_arr = np.concatenate(eig_list, axis=1).astype(np.float32)  # (B, 6)

        feat_block = np.concatenate(
            [
                ch_rel_arr.reshape(B, 95),
                roi_rel.reshape(B, 20),
                ap_ratios,
                ft_ratios,
                roi_slow,
                glob_slow,
                roi_iaf,
                glob_iaf,
                roi_ent,
                glob_ent,
                roi_slope,
                roi_off,
                glob_slope,
                roi_conn_arr,
                eig_arr,
            ],
            axis=1,
        )
        assert feat_block.shape == (B, 173), feat_block.shape
        x_neuro_win[start:end] = feat_block

        gcn_feats[start:end] = np.stack(
            [
                ch_rel_arr[..., 0],
                ch_rel_arr[..., 1],
                ch_rel_arr[..., 2],
                ch_rel_arr[..., 3],
                ch_rel_arr[..., 4],
                ch_iaf,
                ch_slope,
                ch_ent,
            ],
            axis=-1,
        )
        ap_alpha_arr[start:end] = ap_ratios[:, 2]  # log(F/PO) Alpha
        slowing_arr[start:end] = glob_slow[:, 0]
        slope_arr[start:end] = glob_slope[:, 0]

    assert np.isfinite(x_neuro_win).all(), "Non-finite X_neuro_win"
    assert np.isfinite(gcn_feats).all(), "Non-finite gcn_feats"
    return x_neuro_win, gcn_feats, {
        "ap_alpha_ratio": ap_alpha_arr,
        "slowing_ratio": slowing_arr,
        "aperiodic_slope": slope_arr,
    }


def compute_riemannian_covariances(
    wins: np.ndarray, subj_idx_per_win: np.ndarray, n_subjects: int, chunk_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (N_win, 5, 19, 19) LWF SPD covariances & (n_subjects, 5, 19, 19) Fréchet means."""
    n_win = len(wins)
    x_cov_win = np.empty((n_win, len(COV_BANDS), 19, 19), dtype=np.float32)
    cov_est = Covariances(estimator="lwf")

    for b_idx, (_, f_lo, f_hi) in enumerate(COV_BANDS):
        sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=FS, output="sos")
        for start in range(0, n_win, chunk_size):
            end = min(n_win, start + chunk_size)
            filt = sosfiltfilt(sos, wins[start:end].astype(np.float64), axis=-1)
            covs = cov_est.fit_transform(filt)
            # Ensure exact symmetry & strictly positive definiteness
            covs = 0.5 * (covs + np.swapaxes(covs, -2, -1))
            x_cov_win[start:end, b_idx] = covs.astype(np.float32)

    x_cov_subj = np.empty((n_subjects, len(COV_BANDS), 19, 19), dtype=np.float32)
    for s_i in range(n_subjects):
        m = subj_idx_per_win == s_i
        for b_idx in range(len(COV_BANDS)):
            c_sub = x_cov_win[m, b_idx].astype(np.float64)
            c_mean = mean_riemann(c_sub)
            c_mean = 0.5 * (c_mean + c_mean.T)
            min_eig = float(np.linalg.eigvalsh(c_mean).min())
            assert min_eig > 0.0, f"Non-SPD Fréchet mean for subj {s_i} band {b_idx}: min_eig={min_eig}"
            x_cov_subj[s_i, b_idx] = c_mean.astype(np.float32)

    assert np.isfinite(x_cov_win).all() and np.isfinite(x_cov_subj).all()
    return x_cov_win, x_cov_subj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-name", type=str, default="ahepa_v2_features.npz")
    args = parser.parse_args()

    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    self_test_huber()

    bids_root = ensure_bids_data()
    part_df = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract-v2] Using device {device}")

    wins_list, win_subj_id, win_subj_idx, win_grp = [], [], [], []
    subj_ids_uniq, subj_grps_uniq = [], []

    for s_idx, row in enumerate(part_df.itertuples(index=False)):
        sub = str(row.participant_id)
        grp = str(row.Group).strip()
        set_path = bids_root / "derivatives" / sub / "eeg" / f"{sub}_task-eyesclosed_eeg.set"
        assert set_path.exists(), f"Missing {set_path}"
        w, st = load_subject_20s_windows(set_path)
        subj_ids_uniq.append(sub)
        subj_grps_uniq.append(grp)
        wins_list.append(w)
        win_subj_id.extend([sub] * len(w))
        win_subj_idx.extend([s_idx] * len(w))
        win_grp.extend([grp] * len(w))
        if s_idx < 4 or (s_idx + 1) % 20 == 0 or s_idx + 1 == len(part_df):
            print(f"  [{s_idx+1:02d}/88] {sub} ({grp}): kept {st['kept']}/{st['total']} 20s windows")

    wins_all = np.concatenate(wins_list, axis=0)
    del wins_list
    gc.collect()

    win_subj_id_arr = np.asarray(win_subj_id)
    win_subj_idx_arr = np.asarray(win_subj_idx, dtype=np.int64)
    win_grp_arr = np.asarray(win_grp)
    subj_ids_arr = np.asarray(subj_ids_uniq)
    subj_grps_arr = np.asarray(subj_grps_uniq)
    n_subj = len(subj_ids_arr)
    print(f"[info] Total 20s windows={len(wins_all)} across {n_subj} subjects")

    wpli_20s = compute_20s_wpli_batched(wins_all, device=device, batch_size=32)
    x_neuro_win, gcn_node_feats, bio_win = compute_neuro_features_chunked(wins_all, wpli_20s, chunk_size=256)

    # Aggregate subject-level X_neuro_subj = [mean, IQR] across 20s windows -> (88, 346)
    x_neuro_subj = np.empty((n_subj, x_neuro_win.shape[1] * 2), dtype=np.float32)
    subj_ap_alpha = np.empty(n_subj, dtype=np.float32)
    subj_slowing = np.empty(n_subj, dtype=np.float32)
    subj_slope = np.empty(n_subj, dtype=np.float32)
    for s_i in range(n_subj):
        m = win_subj_idx_arr == s_i
        xw = x_neuro_win[m]
        mean_v = xw.mean(axis=0)
        q75, q25 = np.percentile(xw, [75, 25], axis=0)
        iqr_v = q75 - q25
        x_neuro_subj[s_i] = np.concatenate([mean_v, iqr_v]).astype(np.float32)
        subj_ap_alpha[s_i] = float(np.mean(bio_win["ap_alpha_ratio"][m]))
        subj_slowing[s_i] = float(np.mean(bio_win["slowing_ratio"][m]))
        subj_slope[s_i] = float(np.mean(bio_win["aperiodic_slope"][m]))

    x_cov_win, x_cov_subj = compute_riemannian_covariances(wins_all, win_subj_idx_arr, n_subj)
    del wins_all
    gc.collect()

    out_path = CACHE_DIR / args.cache_name
    np.savez(
        out_path,
        x_neuro_subj=x_neuro_subj,
        x_cov_subj=x_cov_subj,
        x_cov_win=x_cov_win,
        gcn_node_feats=gcn_node_feats,
        wpli_20s=wpli_20s,
        win_subj_id=win_subj_id_arr,
        win_subj_idx=win_subj_idx_arr,
        win_grp=win_grp_arr,
        subj_ids=subj_ids_arr,
        subj_grps=subj_grps_arr,
        subj_ap_alpha=subj_ap_alpha,
        subj_slowing=subj_slowing,
        subj_slope=subj_slope,
        ch_names=np.asarray(CANONICAL_CHS),
    )
    print(f"[done] Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
