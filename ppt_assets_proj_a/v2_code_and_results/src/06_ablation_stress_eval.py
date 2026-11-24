#!/usr/bin/env python3
"""SPEC A-V2: Disentangled OOD ablation, cross-filter stress tests, real EEG zero-shot & latency."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import random
import time

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


def _load_local(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SRC_V2 = Path(__file__).resolve().parent
V2_ROOT = SRC_V2.parent
V1_ROOT = V2_ROOT.parent / "proj_a_eegdenoise"
V1_SRC = V1_ROOT / "src"

dl_mod = _load_local("v1_download", V1_SRC / "01_download_data.py")
syn = _load_local("v1_syn", V1_SRC / "02_synthesis.py")
models = _load_local("v1_models", V1_SRC / "03_models.py")
tr_eval = _load_local("v1_train_eval", V1_SRC / "04_train_eval.py")

V1_PROC = V1_ROOT / "data" / "processed"
V1_CKPT = V1_ROOT / "outputs" / "checkpoints"
OUT_DIR = V2_ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
CLINICAL_DIR = V2_ROOT / "data" / "raw_clinical"

V2_SNRS = [-7, -5, -3, 0, 2]
_RC1_SOS = butter(1, 40.0, btype="low", fs=syn.FS, output="sos")
_LP45_SOS = butter(4, 45.0, btype="low", fs=syn.FS, output="sos")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NoLPAugDataset(Dataset):
    """Ablation dataset: Burst (30%) + Clean (10%), NO lowpass filter on EMG."""

    def __init__(self, eeg: np.ndarray, emg: np.ndarray, repeats: int = 4, seed: int = 42) -> None:
        self.eeg = np.ascontiguousarray(eeg, dtype=np.float32)
        self.emg = np.ascontiguousarray(emg, dtype=np.float32)
        self.repeats = repeats
        self.base_seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.eeg) * self.repeats

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.base_seed + self.epoch * 1_000_003 + idx)
        x = self.eeg[idx % len(self.eeg)]
        if float(rng.random()) < 0.10:
            y = x.copy()
        else:
            n = self.emg[int(rng.integers(0, len(self.emg)))].copy()
            if float(rng.random()) < (1.0 / 3.0):
                n = n * syn.make_tukey_burst_mask(syn.T, syn.FS, dur_range=(0.3, 1.5), rng=rng)
            snr_db = float(rng.uniform(-7.0, 2.0))
            y = syn.mix_snr(x, n, snr_db)
        yn, xn, _ = syn.normalize_pair(y, x)
        return torch.from_numpy(yn).unsqueeze(0), torch.from_numpy(xn).unsqueeze(0)


def make_nolp_val_tensors(
    eeg_val: np.ndarray, emg_val: np.ndarray, seed: int = 777
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    y_list, x_list = [], []
    for i in range(len(eeg_val)):
        for _ in range(2):
            x = eeg_val[i]
            if float(rng.random()) < 0.10:
                y = x.copy()
            else:
                n = emg_val[int(rng.integers(0, len(emg_val)))].copy()
                if float(rng.random()) < (1.0 / 3.0):
                    n = n * syn.make_tukey_burst_mask(syn.T, syn.FS, (0.3, 1.5), rng)
                y = syn.mix_snr(x, n, float(rng.uniform(-7.0, 2.0)))
            yn, xn, _ = syn.normalize_pair(y, x)
            y_list.append(yn)
            x_list.append(xn)
    return (
        torch.from_numpy(np.asarray(y_list, dtype=np.float32)).unsqueeze(1),
        torch.from_numpy(np.asarray(x_list, dtype=np.float32)).unsqueeze(1),
    )


def train_nolp_ablation(
    eeg_train: np.ndarray,
    emg_train: np.ndarray,
    eeg_val: np.ndarray,
    emg_val: np.ndarray,
    epochs: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
) -> nn.Module:
    """Train Exp A3-noLP with AMP forward + fp32 CombinedDenoiseLoss."""
    set_seed(42)
    model = models.ResUNet1D(base_ch=32).to(device)
    criterion = models.CombinedDenoiseLoss(alpha=0.1).to(device)
    ds = NoLPAugDataset(eeg_train, emg_train, repeats=repeats, seed=42)
    nw = min(2, os.cpu_count() or 2) if device.type == "cuda" else 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=(device.type == "cuda"))
    vy, vx = make_nolp_val_tensors(eeg_val, emg_val, seed=777)
    val_loader = DataLoader(TensorDataset(vy, vx), batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    no_imp = 0
    ckpt_path = CKPT_DIR / "exp_a3_nolp_best.pt"
    t0 = time.time()

    for ep in range(1, epochs + 1):
        ds.set_epoch(ep)
        model.train()
        tr_loss, tr_n = 0.0, 0
        for yb, xb in loader:
            yb = yb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(yb)
            loss = criterion(pred.float(), xb.float())
            assert torch.isfinite(loss), f"Non-finite loss in Exp A3-noLP epoch {ep}"
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            bs = yb.size(0)
            tr_loss += float(loss.item()) * bs
            tr_n += bs
        sched.step()

        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for yb, xb in val_loader:
                yb = yb.to(device, non_blocking=True)
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    pred = model(yb)
                loss = criterion(pred.float(), xb.float())
                bs = yb.size(0)
                va_loss += float(loss.item()) * bs
                va_n += bs

        va_avg = va_loss / max(1, va_n)
        tr_avg = tr_loss / max(1, tr_n)
        if va_avg < best_val:
            best_val = va_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            no_imp = 0
        else:
            no_imp += 1

        if ep == 1 or ep % 10 == 0 or ep == epochs or no_imp == 0:
            print(f"[Exp A3-noLP] Ep {ep:02d}/{epochs:02d} | train={tr_avg:.5f} | val={va_avg:.5f} | best={best_val:.5f}")
        if no_imp >= 12 and ep >= min(epochs, 15):
            print(f"[Exp A3-noLP] Early stopping at epoch {ep}")
            break

    print(f"[Exp A3-noLP] Finished in {time.time() - t0:.1f}s | best_val={best_val:.5f}")
    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


def make_raised_cosine_burst_mask(
    length: int = 1024, fs: int = 512, dur_range: tuple[float, float] = (0.3, 1.5), rng: np.random.Generator | None = None
) -> np.ndarray:
    """Random 0.3-1.5 s rectangular burst window with 25 ms raised-cosine edges."""
    if rng is None:
        rng = np.random.default_rng()
    dur_s = float(rng.uniform(dur_range[0], dur_range[1]))
    win_len = max(32, min(length, int(round(dur_s * fs))))
    edge = min(int(round(0.025 * fs)), win_len // 3)  # 25 ms = 13 samples
    win = np.ones(win_len, dtype=np.float32)
    if edge > 0:
        taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, edge + 1, dtype=np.float32) / (edge + 1)))
        win[:edge] = taper
        win[-edge:] = taper[::-1]
    start = int(rng.integers(0, length - win_len + 1))
    mask = np.zeros(length, dtype=np.float32)
    mask[start : start + win_len] = win
    return mask


def apply_exp_decay_filter(sig: np.ndarray, alpha: float = 0.015, fs: int = 512) -> np.ndarray:
    """Apply H(f) = exp(-alpha * f) in rfft frequency domain."""
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spec = np.fft.rfft(sig.astype(np.float64))
    atten = np.exp(-alpha * freqs)
    return np.fft.irfft(spec * atten, n=len(sig)).astype(np.float32)


def build_v2_stress_tracks(
    eeg_test: np.ndarray, emg_test: np.ndarray, seed: int = 202602, max_per_snr: int | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Build all 5 V2 stress tracks plus V1 reference tracks across V2_SNRS."""
    rng = np.random.default_rng(seed)
    n_eeg = len(eeg_test) if max_per_snr is None else min(len(eeg_test), max_per_snr)
    n_snrs = len(V2_SNRS)
    emg_idx = rng.integers(0, len(emg_test), size=(n_snrs, n_eeg))

    burst_masks = np.zeros((n_snrs, n_eeg, syn.T), dtype=np.float32)
    drift_freqs = rng.uniform(0.5, 1.5, size=(n_snrs, n_eeg))
    drift_phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_snrs, n_eeg))
    t_sec = np.arange(syn.T, dtype=np.float64) / syn.FS

    for s_i in range(n_snrs):
        for i in range(n_eeg):
            burst_masks[s_i, i] = make_raised_cosine_burst_mask(syn.T, syn.FS, (0.3, 1.5), rng)

    tracks: dict[str, dict[str, list]] = {
        k: {"y": [], "x": [], "snr": []}
        for k in [
            "Test-ID",
            "Test-Pure-LP45",
            "Test-Pure-Burst",
            "Test-ExpDecay",
            "Test-RC1st",
            "Test-Composite-Drift",
            "Test-OOD-45",
        ]
    }
    # Pre-sample 50% burst flag for Test-OOD-45 reference
    ood_burst_flag = rng.random((n_snrs, n_eeg)) < 0.50

    for s_i, snr_db in enumerate(V2_SNRS):
        for i in range(n_eeg):
            x = eeg_test[i]
            n_raw = emg_test[int(emg_idx[s_i, i])]

            n_lp45 = sosfiltfilt(_LP45_SOS, n_raw.astype(np.float64)).astype(np.float32)
            n_burst = n_raw * burst_masks[s_i, i]
            n_exp = apply_exp_decay_filter(n_raw, alpha=0.015, fs=syn.FS)
            n_rc1 = sosfiltfilt(_RC1_SOS, n_raw.astype(np.float64)).astype(np.float32)
            n_ood45 = n_lp45 * burst_masks[s_i, i] if ood_burst_flag[s_i, i] else n_lp45

            y_id = syn.mix_snr(x, n_raw, float(snr_db))
            y_lp45 = syn.mix_snr(x, n_lp45, float(snr_db))
            y_burst = syn.mix_snr(x, n_burst, float(snr_db))
            y_exp = syn.mix_snr(x, n_exp, float(snr_db))
            y_rc1 = syn.mix_snr(x, n_rc1, float(snr_db))
            y_ood45 = syn.mix_snr(x, n_ood45, float(snr_db))

            # Drift: f in [0.5, 1.5] Hz, RMS = 0.35 * RMS(clean_EEG)
            sine = np.sin(2.0 * np.pi * drift_freqs[s_i, i] * t_sec + drift_phases[s_i, i])
            drift = (sine / (syn.compute_rms(sine) + 1e-12) * (0.35 * syn.compute_rms(x))).astype(np.float32)
            y_drift = y_lp45 + drift

            for name, y_sig in [
                ("Test-ID", y_id),
                ("Test-Pure-LP45", y_lp45),
                ("Test-Pure-Burst", y_burst),
                ("Test-ExpDecay", y_exp),
                ("Test-RC1st", y_rc1),
                ("Test-Composite-Drift", y_drift),
                ("Test-OOD-45", y_ood45),
            ]:
                yn, xn, _ = syn.normalize_pair(y_sig, x)
                tracks[name]["y"].append(yn)
                tracks[name]["x"].append(xn)
                tracks[name]["snr"].append(snr_db)

    out: dict[str, dict[str, np.ndarray]] = {}
    for name, d in tracks.items():
        out[name] = {
            "y": np.asarray(d["y"], dtype=np.float32),
            "x": np.asarray(d["x"], dtype=np.float32),
            "snr": np.asarray(d["snr"], dtype=np.int32),
        }

    # Test-Clean
    cy, cx = [], []
    for i in range(n_eeg):
        x = eeg_test[i]
        yn, xn, _ = syn.normalize_pair(x.copy(), x)
        cy.append(yn)
        cx.append(xn)
    out["Test-Clean"] = {
        "y": np.asarray(cy, dtype=np.float32),
        "x": np.asarray(cx, dtype=np.float32),
        "snr": np.zeros(n_eeg, dtype=np.int32),
    }
    return out


def run_real_clinical_zeroshot(
    eval_models: dict[str, nn.Module | None], device: torch.device
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Download uncleaned ds004504 sub-001 (AD) and sub-037 (CN) and evaluate zero-shot."""
    CLINICAL_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    files = {
        "sub-001": "ds004504/sub-001/eeg/sub-001_task-eyesclosed_eeg.set",
        "sub-037": "ds004504/sub-037/eeg/sub-037_task-eyesclosed_eeg.set",
    }
    local_paths = {}
    for sub, key in files.items():
        dst = CLINICAL_DIR / f"{sub}_raw_eeg.set"
        if not dst.exists():
            print(f"[zeroshot] Downloading {key} ...")
            s3.download_file("openneuro.org", key, str(dst))
        local_paths[sub] = dst

    emg_chs = ["Fp1", "Fp2", "F7", "F8", "T7", "T8"]
    occ_chs = ["O1", "O2"]
    rename_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

    emg_candidates = []
    emg_scores = []
    occ_candidates = []
    occ_scores = []

    for sub, path in local_paths.items():
        raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
        cur_ren = {k: v for k, v in rename_map.items() if k in raw.ch_names}
        if cur_ren:
            raw.rename_channels(cur_ren)
        # Minimal preprocessing: 1 Hz highpass + 50 Hz notch, resample 500 -> 512 Hz
        raw.filter(l_freq=1.0, h_freq=None, method="fir", fir_design="firwin", verbose="ERROR")
        raw.notch_filter(freqs=[50.0], verbose="ERROR")
        raw.resample(512, npad="auto", verbose="ERROR")

        data = raw.get_data()
        if float(np.median(np.std(data, axis=1))) < 1e-2:
            data = data * 1e6  # V -> uV
        ch_names = raw.ch_names

        n_win = data.shape[1] // 1024
        for ch_idx, ch in enumerate(ch_names):
            if ch not in emg_chs and not (sub == "sub-037" and ch in occ_chs):
                continue
            wins = data[ch_idx, : n_win * 1024].reshape(n_win, 1024).astype(np.float32)
            ptp = np.ptp(wins, axis=1)
            std = np.std(wins, axis=1)
            valid = (ptp <= 1500.0) & (std >= 0.5)
            wins = wins[valid]
            if len(wins) == 0:
                continue
            freqs, psd = welch(wins.astype(np.float64), fs=512, nperseg=256, axis=1)
            if ch in emg_chs:
                hf_m = (freqs >= 30.0) & (freqs <= 80.0)
                tot_m = (freqs >= 1.0) & (freqs <= 100.0)
                ratio = psd[:, hf_m].sum(axis=1) / (psd[:, tot_m].sum(axis=1) + 1e-12)
                for w_i, r in enumerate(ratio):
                    emg_candidates.append(wins[w_i])
                    emg_scores.append(float(r))
            if sub == "sub-037" and ch in occ_chs:
                al_m = (freqs >= 8.0) & (freqs <= 13.0)
                tot_m = (freqs >= 1.0) & (freqs <= 45.0)
                ratio = psd[:, al_m].sum(axis=1) / (psd[:, tot_m].sum(axis=1) + 1e-12)
                for w_i, r in enumerate(ratio):
                    occ_candidates.append(wins[w_i])
                    occ_scores.append(float(r))

    top_emg_idx = np.argsort(emg_scores)[::-1][:20]
    top_occ_idx = np.argsort(occ_scores)[::-1][:20]

    def _norm_wins(arr: np.ndarray) -> np.ndarray:
        m = arr.mean(axis=1, keepdims=True)
        s = arr.std(axis=1, keepdims=True) + 1e-8
        return ((arr - m) / s).astype(np.float32)

    emg_wins = _norm_wins(np.asarray([emg_candidates[i] for i in top_emg_idx], dtype=np.float32))
    occ_wins = _norm_wins(np.asarray([occ_candidates[i] for i in top_occ_idx], dtype=np.float32))
    print(f"[zeroshot] Selected 20 High-EMG (top 30-80Hz ratio={max(emg_scores):.3f}) & 20 Occipital Alpha (top ratio={max(occ_scores):.3f})")

    freqs, psd_emg_raw = welch(emg_wins.astype(np.float64), fs=512, nperseg=256, axis=1)
    _, psd_occ_raw = welch(occ_wins.astype(np.float64), fs=512, nperseg=256, axis=1)

    hf_mask = (freqs >= 30.0) & (freqs <= 80.0)
    al_mask = (freqs >= 8.0) & (freqs <= 13.0)
    f_al = freqs[al_mask]

    def _paf_cog(psd_arr: np.ndarray) -> np.ndarray:
        p = psd_arr[:, al_mask]
        return np.sum(p * f_al[None, :], axis=1) / (np.sum(p, axis=1) + 1e-12)

    paf_occ_raw = _paf_cog(psd_occ_raw)
    raw_hf = psd_emg_raw[:, hf_mask].sum(axis=1)
    raw_emg_al = psd_emg_raw[:, al_mask].sum(axis=1)
    raw_occ_al = psd_occ_raw[:, al_mask].sum(axis=1)

    rows = []
    cache: dict[str, np.ndarray] = {
        "freqs": freqs,
        "psd_emg_Raw": psd_emg_raw.mean(axis=0),
        "psd_occ_Raw": psd_occ_raw.mean(axis=0),
    }
    for name, mdl in eval_models.items():
        pred_emg = tr_eval.predict_numpy(mdl, emg_wins, device=device)
        pred_occ = tr_eval.predict_numpy(mdl, occ_wins, device=device)
        _, psd_emg_p = welch(pred_emg.astype(np.float64), fs=512, nperseg=256, axis=1)
        _, psd_occ_p = welch(pred_occ.astype(np.float64), fs=512, nperseg=256, axis=1)
        cache[f"psd_emg_{name}"] = psd_emg_p.mean(axis=0)
        cache[f"psd_occ_{name}"] = psd_occ_p.mean(axis=0)

        hf_drop_db = 10.0 * np.log10((raw_hf + 1e-12) / (psd_emg_p[:, hf_mask].sum(axis=1) + 1e-12))
        emg_al_ret = psd_emg_p[:, al_mask].sum(axis=1) / (raw_emg_al + 1e-12)
        occ_al_ratio = psd_occ_p[:, al_mask].sum(axis=1) / (raw_occ_al + 1e-12)
        paf_shift = np.abs(_paf_cog(psd_occ_p) - paf_occ_raw)

        rows.append(
            {
                "experiment": name,
                "high_emg_30_80hz_drop_db_mean": float(hf_drop_db.mean()),
                "high_emg_30_80hz_drop_db_std": float(hf_drop_db.std()),
                "high_emg_alpha_retention_ratio": float(emg_al_ret.mean()),
                "occipital_alpha_power_ratio": float(occ_al_ratio.mean()),
                "occipital_paf_shift_hz_mean": float(paf_shift.mean()),
            }
        )

    df_zs = pd.DataFrame(rows)
    np.savez(OUT_DIR / "real_eeg_zeroshot_cache.npz", **cache)
    return df_zs, cache


def benchmark_latency(eval_models: dict[str, nn.Module | None], device: torch.device) -> pd.DataFrame:
    """Measure trainable parameters and 20-warmup + 200-timed single-window latency."""
    rows = []
    torch.set_num_threads(4)
    x_cpu = torch.randn(1, 1, 1024, dtype=torch.float32)

    for name, mdl in eval_models.items():
        if mdl is None:
            continue
        n_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)

        # CPU timing
        mdl_cpu = mdl.to("cpu").eval()
        with torch.no_grad():
            for _ in range(20):
                _ = mdl_cpu(x_cpu)
            t_cpu = []
            for _ in range(200):
                t0 = time.perf_counter()
                _ = mdl_cpu(x_cpu)
                t_cpu.append((time.perf_counter() - t0) * 1000.0)

        # GPU timing
        gpu_mean, gpu_std = float("nan"), float("nan")
        if device.type == "cuda":
            mdl_gpu = mdl.to(device).eval()
            x_gpu = x_cpu.to(device)
            with torch.no_grad():
                for _ in range(20):
                    _ = mdl_gpu(x_gpu)
                torch.cuda.synchronize()
                t_gpu = []
                for _ in range(200):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = mdl_gpu(x_gpu)
                    torch.cuda.synchronize()
                    t_gpu.append((time.perf_counter() - t0) * 1000.0)
            gpu_mean = float(np.mean(t_gpu))
            gpu_std = float(np.std(t_gpu))
        else:
            mdl.to(device)

        rows.append(
            {
                "experiment": name,
                "params": int(n_params),
                "cpu_ms_mean": float(np.mean(t_cpu)),
                "cpu_ms_std": float(np.std(t_cpu)),
                "gpu_ms_mean": gpu_mean,
                "gpu_ms_std": gpu_std,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Ensure processed 512 Hz data exists
    if not (V1_PROC / "eeg_512hz.npy").exists():
        dl_mod.main()

    eeg = np.load(V1_PROC / "eeg_512hz.npy")
    emg = np.load(V1_PROC / "emg_512hz.npy")
    splits = np.load(V1_PROC / "split_indices.npz")
    assert eeg.shape[1] == 1024 and emg.shape[1] == 1024
    assert set(splits["train_eeg_idx"]).isdisjoint(splits["test_eeg_idx"])
    assert set(splits["train_emg_idx"]).isdisjoint(splits["test_emg_idx"])

    train_eeg, val_eeg, test_eeg = eeg[splits["train_eeg_idx"]], eeg[splits["val_eeg_idx"]], eeg[splits["test_eeg_idx"]]
    train_emg, val_emg, test_emg = emg[splits["train_emg_idx"]], emg[splits["val_emg_idx"]], emg[splits["test_emg_idx"]]

    # Load existing V1 checkpoints or retrain if absent
    loaded_models: dict[str, nn.Module | None] = {"Exp A0": None}
    v1_specs = [
        ("Exp A1", models.ResCNN1D(), nn.MSELoss(), "id", "exp_a1_best.pt"),
        ("Exp A2", models.ResUNet1D(base_ch=32), models.CombinedDenoiseLoss(alpha=0.1), "id", "exp_a2_best.pt"),
        ("Exp A3", models.ResUNet1D(base_ch=32), models.CombinedDenoiseLoss(alpha=0.1), "physics", "exp_a3_best.pt"),
    ]
    for name, mdl, crit, dmode, ckpt_name in v1_specs:
        p1 = V1_CKPT / ckpt_name
        p2 = CKPT_DIR / ckpt_name
        ckpt = p1 if p1.exists() else (p2 if p2.exists() else None)
        if ckpt is not None:
            print(f"[checkpoint] Loaded {name} from {ckpt}")
            mdl.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            mdl.to(device).eval()
            loaded_models[name] = mdl
        else:
            print(f"[warn] Missing {ckpt_name}; training {name} from scratch...")
            ep = 2 if args.smoke else args.epochs
            rep = 1 if args.smoke else 4
            loaded_models[name] = tr_eval.train_experiment(
                name, mdl, crit, train_eeg, train_emg, val_eeg, val_emg, dmode,
                epochs=ep, batch_size=args.batch_size, lr=1e-3, weight_decay=1e-4, patience=12, repeats=rep, device=device,
            )

    # Train Exp A3-noLP
    nolp_ckpt = CKPT_DIR / "exp_a3_nolp_best.pt"
    if nolp_ckpt.exists() and not args.smoke:
        print(f"[checkpoint] Loaded Exp A3-noLP from {nolp_ckpt}")
        m_nolp = models.ResUNet1D(base_ch=32)
        m_nolp.load_state_dict(torch.load(nolp_ckpt, map_location=device, weights_only=True))
        m_nolp.to(device).eval()
    else:
        ep = 2 if args.smoke else args.epochs
        rep = 1 if args.smoke else 4
        te = train_eeg[:256] if args.smoke else train_eeg
        ve = val_eeg[:64] if args.smoke else val_eeg
        m_nolp = train_nolp_ablation(te, train_emg, ve, val_emg, epochs=ep, batch_size=args.batch_size, repeats=rep, device=device)

    eval_models: dict[str, nn.Module | None] = {
        "Exp A0": loaded_models["Exp A0"],
        "Exp A1": loaded_models["Exp A1"],
        "Exp A2": loaded_models["Exp A2"],
        "Exp A3-noLP": m_nolp,
        "Exp A3": loaded_models["Exp A3"],
    }

    # Build V2 stress tracks
    max_per_snr = 10 if args.smoke else None
    tracks = build_v2_stress_tracks(test_eeg, test_emg, seed=202602, max_per_snr=max_per_snr)

    rows = []
    for exp_name, mdl in eval_models.items():
        for track_name, tdata in tracks.items():
            preds = tr_eval.predict_numpy(mdl, tdata["y"], device=device)
            m = tr_eval.compute_metrics_batch(preds, tdata["x"])
            rows.append(
                {
                    "experiment": exp_name,
                    "track": track_name,
                    "snr_db": "Overall",
                    "n_samples": len(preds),
                    "rrmse_t": float(m["rrmse_t"].mean()),
                    "rrmse_s": float(m["rrmse_s"].mean()),
                    "cc": float(m["cc"].mean()),
                    "alpha_mae": float(m["alpha_mae"].mean()),
                    "beta_mae": float(m["beta_mae"].mean()),
                }
            )
            if track_name != "Test-Clean":
                for snr_v in V2_SNRS:
                    msk = tdata["snr"] == snr_v
                    rows.append(
                        {
                            "experiment": exp_name,
                            "track": track_name,
                            "snr_db": int(snr_v),
                            "n_samples": int(msk.sum()),
                            "rrmse_t": float(m["rrmse_t"][msk].mean()),
                            "rrmse_s": float(m["rrmse_s"][msk].mean()),
                            "cc": float(m["cc"][msk].mean()),
                            "alpha_mae": float(m["alpha_mae"][msk].mean()),
                            "beta_mae": float(m["beta_mae"][msk].mean()),
                        }
                    )

    df_ab = pd.DataFrame(rows)
    df_ab.to_csv(OUT_DIR / "ablation_metrics_v2.csv", index=False)

    # Real clinical EEG zero-shot
    df_zs, _ = run_real_clinical_zeroshot(eval_models, device=device)
    df_zs.to_csv(OUT_DIR / "zeroshot_metrics_v2.csv", index=False)

    # Latency benchmark
    df_lat = benchmark_latency(eval_models, device=device)
    df_lat.to_csv(OUT_DIR / "latency_benchmark_v2.csv", index=False)

    # Build REPORT_SUMMARY_V2.md
    ov = df_ab[df_ab["snr_db"] == "Overall"].copy()
    track_order = [
        "Test-ID",
        "Test-Pure-LP45",
        "Test-Pure-Burst",
        "Test-ExpDecay",
        "Test-RC1st",
        "Test-OOD-45",
        "Test-Composite-Drift",
        "Test-Clean",
    ]
    exp_order = ["Exp A0", "Exp A1", "Exp A2", "Exp A3-noLP", "Exp A3"]

    def _get(exp: str, trk: str, col: str) -> float:
        return float(ov[(ov["experiment"] == exp) & (ov["track"] == trk)][col].iloc[0])

    lines = [
        "# SPEC A-V2 Report: Disentangled OOD Ablation, Cross-Filter Stress Tests & Real-EEG Zero-Shot",
        "",
        "## 1. Disentangled Ablation & Cross-Filter Stress Tracks (Overall RRMSE_t ↓ / CC ↑)",
        "",
        "| Track | Exp A0 | Exp A1 (ResCNN) | Exp A2 (UNet ID) | Exp A3-noLP (Burst+Clean) | **Exp A3 (Full Physics)** | A3 vs A3-noLP RRMSE_t |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for trk in track_order:
        cells = [f"{_get(e, trk, 'rrmse_t'):.4f} ({_get(e, trk, 'cc'):.3f})" for e in exp_order]
        r_nolp = _get("Exp A3-noLP", trk, "rrmse_t")
        r_a3 = _get("Exp A3", trk, "rrmse_t")
        imp = (r_nolp - r_a3) / r_nolp * 100.0
        tag = f"**{imp:+.1f}%**"
        lines.append(f"| {trk} | {' | '.join(cells)} | {tag} |")

    lines.extend(
        [
            "",
            "*Note: `Test-Composite-Drift` adds unseen 0.5–1.5 Hz baseline drift (`RMS = 0.35 * RMS(clean_EEG)`) as an out-of-scope limitation stress test.*",
            "",
            "## 2. Real Clinical EEG Zero-Shot Validation (`ds004504` Raw `sub-001` & `sub-037`, 512 Hz)",
            "",
            "| Experiment | High-EMG 30–80 Hz Drop (dB) ↑ | High-EMG Alpha Retention | Occipital Alpha Power Ratio | Occipital PAF Shift (Hz) ↓ |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in df_zs.itertuples(index=False):
        lines.append(
            f"| {r.experiment} | {r.high_emg_30_80hz_drop_db_mean:.2f} ± {r.high_emg_30_80hz_drop_db_std:.2f} | "
            f"{r.high_emg_alpha_retention_ratio:.3f} | {r.occipital_alpha_power_ratio:.3f} | {r.occipital_paf_shift_hz_mean:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Single-Window (2s, 1,024 pts) Inference Latency & Parameters",
            "",
            "| Experiment | Trainable Params | CPU Latency (ms, 4 threads) | GPU Latency (ms) |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )
    for r in df_lat.itertuples(index=False):
        gpu_str = f"{r.gpu_ms_mean:.2f} ± {r.gpu_ms_std:.2f}" if np.isfinite(r.gpu_ms_mean) else "N/A"
        lines.append(
            f"| {r.experiment} | {r.params:,} | {r.cpu_ms_mean:.2f} ± {r.cpu_ms_std:.2f} | {gpu_str} |"
        )

    lines.extend(
        [
            "",
            "## 4. Output Figures",
            "",
            "- `outputs/figures/fig4_disentangle_ablation.png`",
            "- `outputs/figures/fig5_real_eeg_zeroshot.png`",
            "",
        ]
    )
    (OUT_DIR / "REPORT_SUMMARY_V2.md").write_text("\n".join(lines), encoding="utf-8")
    print("[saved] REPORT_SUMMARY_V2.md")
    print(ov[ov["track"].isin(["Test-Pure-LP45", "Test-Pure-Burst", "Test-ExpDecay", "Test-RC1st", "Test-Composite-Drift", "Test-Clean"])][["track", "experiment", "rrmse_t", "cc"]].to_string(index=False))
    print("\n--- Zero-Shot ---")
    print(df_zs.to_string(index=False))
    print("\n--- Latency ---")
    print(df_lat.to_string(index=False))


if __name__ == "__main__":
    main()
