#!/usr/bin/env python3
"""Preprocess ds004504 .set files (250 Hz, 4s epochs) and extract per-epoch PSD/wPLI/PLV."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
import warnings

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt, welch
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

PROJ_ROOT = Path(__file__).resolve().parent.parent
BIDS_DIR = PROJ_ROOT / "data" / "ds004504"
CACHE_DIR = PROJ_ROOT / "data" / "cache"

LEGACY_RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
CANONICAL_CHS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]
BANDS = [
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 25.0),
    ("Gamma", 25.0, 45.0),
]
FS = 250
WIN_LEN = 1000  # 4s * 250 Hz
PTP_THRESH_UV = 500.0


def extract_node_features(epochs_uv: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Compute (N, 19, 13) per-epoch node features in chunks to bound peak RAM."""
    n_ep = len(epochs_uv)
    feats = np.empty((n_ep, 19, 13), dtype=np.float32)
    for start in range(0, n_ep, batch_size):
        end = min(n_ep, start + batch_size)
        freqs, psd = welch(
            epochs_uv[start:end].astype(np.float64),
            fs=FS,
            nperseg=500,
            noverlap=250,
            axis=-1,
        )
        df_hz = freqs[1] - freqs[0]  # 0.5 Hz
        tot_mask = (freqs >= 0.5) & (freqs <= 45.0)
        tot_psd = psd[..., tot_mask]
        tot_power = np.sum(tot_psd, axis=-1, keepdims=True) * df_hz + 1e-12

        rel_powers = []
        log_powers = []
        for _, f_lo, f_hi in BANDS:
            if f_lo == 0.5:
                m = (freqs >= f_lo) & (freqs <= f_hi)
            else:
                m = (freqs > f_lo) & (freqs <= f_hi)
            bp = np.sum(psd[..., m], axis=-1, keepdims=True) * df_hz
            rel_powers.append(bp / tot_power)
            log_powers.append(np.log10(bp + 1e-12))

        rel_arr = np.concatenate(rel_powers, axis=-1)
        log_arr = np.concatenate(log_powers, axis=-1)

        p_norm = tot_psd / (np.sum(tot_psd, axis=-1, keepdims=True) + 1e-12)
        n_bins = p_norm.shape[-1]
        spec_ent = -np.sum(p_norm * np.log(p_norm + 1e-12), axis=-1, keepdims=True) / np.log(n_bins)

        theta_p = rel_powers[1] * tot_power
        alpha_p = rel_powers[2] * tot_power
        ta_ratio = np.log((theta_p + 1e-12) / (alpha_p + 1e-12))

        paf_mask = (freqs >= 7.0) & (freqs <= 13.0)
        f_paf = freqs[paf_mask][None, None, :]
        p_paf = psd[..., paf_mask]
        paf = np.sum(p_paf * f_paf, axis=-1, keepdims=True) / (
            np.sum(p_paf, axis=-1, keepdims=True) + 1e-12
        )

        feats[start:end] = np.concatenate([rel_arr, log_arr, spec_ent, ta_ratio, paf], axis=-1).astype(np.float32)

    assert feats.shape == (n_ep, 19, 13)
    assert np.isfinite(feats).all(), "Non-finite node features detected"
    return feats


def extract_connectivity(
    epochs_uv: np.ndarray, device: torch.device, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-epoch 5-band wPLI and PLV matrices (N, 5, 19, 19)."""
    n_ep = epochs_uv.shape[0]
    wpli_all = np.zeros((n_ep, len(BANDS), 19, 19), dtype=np.float32)
    plv_all = np.zeros((n_ep, len(BANDS), 19, 19), dtype=np.float32)
    trim = 25  # trim edge artifacts from Hilbert transform

    eye_mask = ~torch.eye(19, dtype=torch.bool, device=device)

    for b_idx, (_, f_lo, f_hi) in enumerate(BANDS):
        sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=FS, output="sos")
        for start in range(0, n_ep, batch_size):
            end = min(n_ep, start + batch_size)
            chunk = epochs_uv[start:end].astype(np.float64)
            filt = sosfiltfilt(sos, chunk, axis=-1)
            analytic = hilbert(filt, axis=-1)[..., trim:-trim]  # (B, 19, T')

            z = torch.from_numpy(analytic).to(device=device, dtype=torch.complex64)
            u = z / (torch.abs(z) + 1e-12)

            # PLV_ij = |mean_t(u_i * conj(u_j))|
            plv_mat = torch.abs(torch.einsum("bct,bdt->bcd", u, u.conj()) / u.shape[-1])

            # wPLI_ij = |mean_t(Im(z_i * conj(z_j)))| / (mean_t(|Im(z_i * conj(z_j))|) + eps)
            # Compute cross-spectrum imag part explicitly over time without huge broadcast:
            # z_i * conj(z_j) = (zr_i*zr_j + zi_i*zi_j) + i*(zi_i*zr_j - zr_i*zi_j)
            zr = z.real  # (B, 19, T')
            zi = z.imag
            im_cross = zi.unsqueeze(2) * zr.unsqueeze(1) - zr.unsqueeze(2) * zi.unsqueeze(1)  # (B, 19, 19, T')
            num = torch.abs(im_cross.mean(dim=-1))
            den = torch.abs(im_cross).mean(dim=-1) + 1e-12
            wpli_mat = num / den

            plv_mat = torch.clamp(plv_mat * eye_mask.unsqueeze(0), 0.0, 1.0)
            wpli_mat = torch.clamp(wpli_mat * eye_mask.unsqueeze(0), 0.0, 1.0)

            plv_all[start:end, b_idx] = plv_mat.cpu().numpy()
            wpli_all[start:end, b_idx] = wpli_mat.cpu().numpy()

    assert np.isfinite(wpli_all).all() and np.isfinite(plv_all).all()
    assert (wpli_all >= 0.0).all() and (wpli_all <= 1.0001).all()
    assert (plv_all >= 0.0).all() and (plv_all <= 1.0001).all()
    assert np.allclose(wpli_all, np.swapaxes(wpli_all, -2, -1), atol=1e-5)
    assert np.allclose(plv_all, np.swapaxes(plv_all, -2, -1), atol=1e-5)
    return wpli_all, plv_all


def preprocess_subject(set_path: Path) -> tuple[np.ndarray, dict[str, float]]:
    """Load one subject .set, standardize channels/montage, filter/resample, cut 4s windows."""
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    rename_map = {k: v for k, v in LEGACY_RENAME.items() if k in raw.ch_names}
    if rename_map:
        raw.rename_channels(rename_map)

    raw.pick(CANONICAL_CHS)
    assert raw.ch_names == CANONICAL_CHS, f"Channel mismatch: {raw.ch_names}"
    raw.set_montage("standard_1020", on_missing="raise")

    raw.filter(l_freq=0.5, h_freq=45.0, method="fir", fir_design="firwin", verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - FS) > 1e-3:
        raw.resample(FS, npad="auto", verbose="ERROR")

    data = raw.get_data()  # Volts if EEGLAB microV was scaled by MNE
    med_std = float(np.median(np.std(data, axis=1)))
    if med_std < 1e-2:
        data_uv = data * 1e6
    else:
        data_uv = data
    data_uv = data_uv.astype(np.float32)

    n_win = data_uv.shape[1] // WIN_LEN
    assert n_win > 0, f"Recording too short in {set_path}: {data_uv.shape}"
    epochs = data_uv[:, : n_win * WIN_LEN].reshape(19, n_win, WIN_LEN).transpose(1, 0, 2)

    # Artifact rejection
    ptp = np.ptp(epochs, axis=-1).max(axis=-1)
    ch_std = np.std(epochs, axis=-1).min(axis=-1)
    keep = (ptp <= PTP_THRESH_UV) & (ch_std >= 1e-3)
    kept_epochs = np.ascontiguousarray(epochs[keep], dtype=np.float32)
    stats = {
        "n_total": int(n_win),
        "n_kept": int(keep.sum()),
        "med_std_uv": float(np.median(np.std(data_uv, axis=1))),
    }
    return kept_epochs, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-name", type=str, default="ahepa_dataset.npz")
    args = parser.parse_args()

    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    part_df = pd.read_csv(BIDS_DIR / "participants.tsv", sep="\t")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[preprocess] Using device {device} for connectivity")

    epochs_list, subj_list, grp_list = [], [], []
    med_stds = []
    processed_subjs = 0
    for row in part_df.itertuples(index=False):
        subj = str(row.participant_id)
        grp = str(row.Group).strip()
        set_path = BIDS_DIR / "derivatives" / subj / "eeg" / f"{subj}_task-eyesclosed_eeg.set"
        if not set_path.exists():
            continue
        ep, st = preprocess_subject(set_path)
        med_stds.append(st["med_std_uv"])
        processed_subjs += 1
        if st["n_kept"] < 10:
            print(f"[warn] {subj} ({grp}) retained only {st['n_kept']}/{st['n_total']} epochs!")
        if processed_subjs <= 5 or processed_subjs % 15 == 0:
            print(
                f"  {subj} ({grp}): kept {st['n_kept']}/{st['n_total']} epochs | med_std={st['med_std_uv']:.2f} uV"
            )
        epochs_list.append(ep)
        subj_list.extend([subj] * len(ep))
        grp_list.extend([grp] * len(ep))

    assert epochs_list, "No .set files found in derivatives directory!"
    raw_epochs = np.concatenate(epochs_list, axis=0)
    del epochs_list
    gc.collect()
    subject_ids = np.asarray(subj_list)
    groups = np.asarray(grp_list)

    print(
        f"[info] Preprocessed {processed_subjs} subjects -> {len(raw_epochs)} epochs | "
        f"uV med_std range=[{min(med_stds):.2f}, {max(med_stds):.2f}]"
    )
    print(
        "[info] Group epoch counts:",
        {g: int((groups == g).sum()) for g in sorted(set(groups))},
    )

    node_feats = extract_node_features(raw_epochs)
    wpli, plv = extract_connectivity(raw_epochs, device=device, batch_size=64)

    out_path = CACHE_DIR / args.cache_name
    np.savez(
        out_path,
        raw_epochs=raw_epochs,
        node_feats=node_feats,
        wpli=wpli,
        plv=plv,
        subject_ids=subject_ids,
        groups=groups,
        ch_names=np.asarray(CANONICAL_CHS),
    )
    print(f"[done] Cached {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
