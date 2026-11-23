#!/usr/bin/env python3
"""Standard linear mixing and physics-informed volume-conduction / burst synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey
import torch
from torch.utils.data import Dataset

FS = 512
T = 1024
SNR_LEVELS = list(range(-7, 3))  # -7 to +2 dB inclusive

# Pre-computed 4th-order Butterworth lowpass SOS bank for fc in [35, 110] Hz
_LOWPASS_SOS_BANK: dict[int, np.ndarray] = {
    fc: butter(4, fc, btype="low", fs=FS, output="sos") for fc in range(35, 111)
}


def get_lowpass_sos(fc: int) -> np.ndarray:
    """Return cached 4th-order Butterworth lowpass SOS coefficients at cutoff fc Hz."""
    if fc not in _LOWPASS_SOS_BANK:
        _LOWPASS_SOS_BANK[fc] = butter(4, fc, btype="low", fs=FS, output="sos")
    return _LOWPASS_SOS_BANK[fc]


def apply_lowpass(sig: np.ndarray, fc: int) -> np.ndarray:
    """Apply zero-phase 4th-order Butterworth lowpass filter to a 1D epoch."""
    sos = get_lowpass_sos(int(fc))
    return sosfiltfilt(sos, sig.astype(np.float64)).astype(np.float32)


def make_tukey_burst_mask(
    length: int = T,
    fs: int = FS,
    dur_range: tuple[float, float] = (0.3, 1.5),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a smooth Tukey-window burst mask over a random temporal segment."""
    if rng is None:
        rng = np.random.default_rng()
    dur_s = float(rng.uniform(dur_range[0], dur_range[1]))
    win_len = max(16, min(length, int(round(dur_s * fs))))
    start = int(rng.integers(0, length - win_len + 1))
    mask = np.zeros(length, dtype=np.float32)
    mask[start : start + win_len] = tukey(win_len, alpha=0.25).astype(np.float32)
    return mask


def compute_rms(sig: np.ndarray) -> float:
    """Compute root-mean-square amplitude of a 1D signal."""
    return float(np.sqrt(np.mean(sig.astype(np.float64) ** 2) + 1e-12))


def mix_snr(x: np.ndarray, n: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix clean EEG x and artifact n at target SNR (dB) via RMS scaling."""
    rms_x = compute_rms(x)
    rms_n = compute_rms(n)
    if rms_n < 1e-10:
        return x.astype(np.float32).copy()
    lam = rms_x / (rms_n * (10.0 ** (snr_db / 20.0)))
    return (x + lam * n).astype(np.float32)


def normalize_pair(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize (y, x) by std(y) to preserve relative amplitude scale."""
    sigma = float(np.std(y)) + 1e-8
    return (y / sigma).astype(np.float32), (x / sigma).astype(np.float32), sigma


class EEGDenoisingTrainDataset(Dataset):
    """Training dataset supporting standard linear (ID) or physics-informed synthesis."""

    def __init__(
        self,
        eeg: np.ndarray,
        emg: np.ndarray,
        mode: Literal["id", "physics"] = "id",
        repeats: int = 4,
        seed: int = 42,
    ) -> None:
        self.eeg = np.ascontiguousarray(eeg, dtype=np.float32)
        self.emg = np.ascontiguousarray(emg, dtype=np.float32)
        self.mode = mode
        self.repeats = repeats
        self.base_seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance deterministic seed per training epoch while remaining reproducible."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.eeg) * self.repeats

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.base_seed + self.epoch * 1_000_003 + idx)
        x = self.eeg[idx % len(self.eeg)]

        if self.mode == "id":
            n = self.emg[int(rng.integers(0, len(self.emg)))]
            snr_db = float(rng.uniform(-7.0, 2.0))
            y = mix_snr(x, n, snr_db)
        elif self.mode == "physics":
            u = float(rng.random())
            if u < 0.10:
                # 10% pure clean EEG to prevent over-denoising
                y = x.copy()
            else:
                n = self.emg[int(rng.integers(0, len(self.emg)))].copy()
                # 75% volume-conduction lowpass fc in [35, 110] Hz; 25% unfiltered
                if float(rng.random()) < 0.75:
                    fc = int(rng.integers(35, 111))
                    n = apply_lowpass(n, fc)
                # 30% of all samples (= 1/3 of noisy 90%) get a local Tukey burst mask
                if float(rng.random()) < (1.0 / 3.0):
                    mask = make_tukey_burst_mask(T, FS, dur_range=(0.3, 1.5), rng=rng)
                    n = n * mask
                snr_db = float(rng.uniform(-7.0, 2.0))
                y = mix_snr(x, n, snr_db)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        y_norm, x_norm, _ = normalize_pair(y, x)
        return (
            torch.from_numpy(y_norm).unsqueeze(0),
            torch.from_numpy(x_norm).unsqueeze(0),
        )


def build_test_suites(
    eeg_test: np.ndarray,
    emg_test: np.ndarray,
    seed: int = 2026,
    max_per_snr: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Build deterministic Test-ID, Test-OOD-60, Test-OOD-45, and Test-Clean suites."""
    rng = np.random.default_rng(seed)
    n_eeg = len(eeg_test) if max_per_snr is None else min(len(eeg_test), max_per_snr)

    # Pre-sample paired EMG index, burst flag, and burst masks identically per (snr, i)
    n_snrs = len(SNR_LEVELS)
    emg_indices = rng.integers(0, len(emg_test), size=(n_snrs, n_eeg))
    burst_flags = rng.random(size=(n_snrs, n_eeg)) < 0.50
    burst_masks = np.ones((n_snrs, n_eeg, T), dtype=np.float32)
    for s_idx in range(n_snrs):
        for i in range(n_eeg):
            if burst_flags[s_idx, i]:
                burst_masks[s_idx, i] = make_tukey_burst_mask(
                    T, FS, dur_range=(0.5, 1.2), rng=rng
                )

    def _build_track(fc: int | None, use_burst: bool) -> dict[str, np.ndarray]:
        y_list, x_list, s_list, snr_list = [], [], [], []
        for s_idx, snr_db in enumerate(SNR_LEVELS):
            for i in range(n_eeg):
                x = eeg_test[i]
                n = emg_test[int(emg_indices[s_idx, i])].copy()
                if fc is not None:
                    n = apply_lowpass(n, fc)
                if use_burst and burst_flags[s_idx, i]:
                    n = n * burst_masks[s_idx, i]
                y = mix_snr(x, n, float(snr_db))
                y_norm, x_norm, sigma = normalize_pair(y, x)
                y_list.append(y_norm)
                x_list.append(x_norm)
                s_list.append(sigma)
                snr_list.append(snr_db)
        return {
            "y": np.asarray(y_list, dtype=np.float32),
            "x": np.asarray(x_list, dtype=np.float32),
            "sigma": np.asarray(s_list, dtype=np.float32),
            "snr": np.asarray(snr_list, dtype=np.int32),
        }

    # Test-Clean: all clean EEG epochs with lambda = 0
    clean_y, clean_x, clean_s = [], [], []
    for i in range(n_eeg):
        x = eeg_test[i]
        y_norm, x_norm, sigma = normalize_pair(x.copy(), x)
        clean_y.append(y_norm)
        clean_x.append(x_norm)
        clean_s.append(sigma)

    return {
        "Test-ID": _build_track(fc=None, use_burst=False),
        "Test-OOD-60": _build_track(fc=60, use_burst=True),
        "Test-OOD-45": _build_track(fc=45, use_burst=True),
        "Test-Clean": {
            "y": np.asarray(clean_y, dtype=np.float32),
            "x": np.asarray(clean_x, dtype=np.float32),
            "sigma": np.asarray(clean_s, dtype=np.float32),
            "snr": np.zeros(n_eeg, dtype=np.int32),
        },
    }


def _self_test() -> None:
    proc_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    eeg = np.load(proc_dir / "eeg_512hz.npy")
    emg = np.load(proc_dir / "emg_512hz.npy")
    splits = np.load(proc_dir / "split_indices.npz")

    # Verify SNR mixing accuracy
    x = eeg[splits["test_eeg_idx"][0]]
    n = emg[splits["test_emg_idx"][0]]
    for target_snr in [-7, -2, 0, 2]:
        y = mix_snr(x, n, target_snr)
        actual_snr = 20.0 * np.log10(compute_rms(x) / compute_rms(y - x))
        assert abs(actual_snr - target_snr) < 1e-3, (target_snr, actual_snr)

    # Verify dataset modes
    ds_id = EEGDenoisingTrainDataset(
        eeg[splits["train_eeg_idx"]], emg[splits["train_emg_idx"]], mode="id"
    )
    ds_ph = EEGDenoisingTrainDataset(
        eeg[splits["train_eeg_idx"]], emg[splits["train_emg_idx"]], mode="physics"
    )
    y0, x0 = ds_id[0]
    y1, x1 = ds_ph[0]
    assert y0.shape == (1, T) and x0.shape == (1, T)
    assert y1.shape == (1, T) and x1.shape == (1, T)
    assert torch.isfinite(y0).all() and torch.isfinite(y1).all()

    # Verify clean ratio in physics mode (~10%)
    clean_count = 0
    for i in range(200):
        yi, xi = ds_ph[i]
        if torch.allclose(yi, xi, atol=1e-6):
            clean_count += 1
    assert 5 <= clean_count <= 45, f"Unexpected clean sample count in 200: {clean_count}"

    suites = build_test_suites(
        eeg[splits["test_eeg_idx"]], emg[splits["test_emg_idx"]], max_per_snr=5
    )
    assert suites["Test-ID"]["y"].shape == (50, T)
    assert suites["Test-OOD-45"]["y"].shape == (50, T)
    assert suites["Test-Clean"]["y"].shape == (5, T)
    assert np.allclose(suites["Test-Clean"]["y"], suites["Test-Clean"]["x"])
    print(f"[self-test PASS] SNR accurate, physics clean_count={clean_count}/200, suites verified.")


if __name__ == "__main__":
    _self_test()
