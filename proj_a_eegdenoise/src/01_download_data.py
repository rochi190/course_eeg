#!/usr/bin/env python3
"""Download EEGdenoiseNet (512 Hz) and generate leak-free 80/10/10 splits."""

from __future__ import annotations

import os
from pathlib import Path
import urllib.request
import numpy as np
from scipy.signal import resample

PROJ_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJ_ROOT / "data" / "raw"
PROC_DIR = PROJ_ROOT / "data" / "processed"

GIN_URLS = {
    "eeg": (
        "EEG_all_epochs_512hz.npy",
        "https://gin.g-node.org/NCClab/EEGdenoiseNet/raw/master/data/EEG_all_epochs_512hz.npy",
    ),
    "emg": (
        "EMG_all_epochs_512hz.npy",
        "https://gin.g-node.org/NCClab/EEGdenoiseNet/raw/master/data/EMG_all_epochs_512hz.npy",
    ),
}

GITHUB_FALLBACK_URLS = {
    "eeg": (
        "EEG_all_epochs.npy",
        "https://raw.githubusercontent.com/ncclabsustech/EEGdenoiseNet/master/data/EEG_all_epochs.npy",
    ),
    "emg": (
        "EMG_all_epochs.npy",
        "https://raw.githubusercontent.com/ncclabsustech/EEGdenoiseNet/master/data/EMG_all_epochs.npy",
    ),
}


def download_file(url: str, dst: Path, timeout: int = 120) -> None:
    """Download a file to dst using streaming writes and verify NPY magic header."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3"})
    print(f"[download] Fetching {url} -> {dst}")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    with open(tmp, "rb") as f:
        magic = f.read(6)
    if magic != b"\x93NUMPY":
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Invalid NPY magic header from {url}: {magic!r}")
    tmp.replace(dst)


def fetch_or_load(kind: str) -> np.ndarray:
    """Fetch 512 Hz file from GIN or fall back to GitHub 256 Hz file."""
    fname_512, url_512 = GIN_URLS[kind]
    path_512 = RAW_DIR / fname_512
    fname_fb, url_fb = GITHUB_FALLBACK_URLS[kind]
    path_fb = RAW_DIR / fname_fb

    if path_512.exists():
        with open(path_512, "rb") as f:
            if f.read(6) == b"\x93NUMPY":
                print(f"[cache] Using existing {path_512}")
                return np.ascontiguousarray(np.load(path_512), dtype=np.float32)

    try:
        download_file(url_512, path_512)
        return np.ascontiguousarray(np.load(path_512), dtype=np.float32)
    except Exception as exc:
        print(f"[warn] Primary GIN download failed for {kind}: {exc}")
        if not path_fb.exists():
            download_file(url_fb, path_fb)
        return np.ascontiguousarray(np.load(path_fb), dtype=np.float32)


def ensure_512hz(arr: np.ndarray, name: str) -> np.ndarray:
    """Resample 256 Hz epochs (T=512) to 512 Hz (T=1024) and assert validity."""
    if arr.ndim != 2:
        raise ValueError(f"{name} expected 2D array, got shape {arr.shape}")
    if arr.shape[1] == 512:
        print(f"[resample] {name} shape[1]==512 (256 Hz) -> resampling to 1024 (512 Hz)")
        arr = resample(arr, 1024, axis=1)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    assert arr.shape[1] == 1024, f"{name} expected 1024 samples, got {arr.shape}"
    assert np.isfinite(arr).all(), f"{name} contains NaN or Inf"
    return arr


def split_indices(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create disjoint 80/10/10 train/val/test index arrays."""
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val :])
    return train_idx, val_idx, test_idx


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    eeg = ensure_512hz(fetch_or_load("eeg"), "EEG")
    emg = ensure_512hz(fetch_or_load("emg"), "EMG")
    assert eeg.shape[1] == 1024 and emg.shape[1] == 1024

    rng = np.random.default_rng(42)
    train_eeg_idx, val_eeg_idx, test_eeg_idx = split_indices(len(eeg), rng)
    train_emg_idx, val_emg_idx, test_emg_idx = split_indices(len(emg), rng)

    assert set(train_eeg_idx).isdisjoint(val_eeg_idx)
    assert set(train_eeg_idx).isdisjoint(test_eeg_idx)
    assert set(val_eeg_idx).isdisjoint(test_eeg_idx)

    assert set(train_emg_idx).isdisjoint(val_emg_idx)
    assert set(train_emg_idx).isdisjoint(test_emg_idx)
    assert set(val_emg_idx).isdisjoint(test_emg_idx)

    assert len(train_eeg_idx) + len(val_eeg_idx) + len(test_eeg_idx) == len(eeg)
    assert len(train_emg_idx) + len(val_emg_idx) + len(test_emg_idx) == len(emg)

    np.save(PROC_DIR / "eeg_512hz.npy", eeg)
    np.save(PROC_DIR / "emg_512hz.npy", emg)
    np.savez(
        PROC_DIR / "split_indices.npz",
        train_eeg_idx=train_eeg_idx,
        val_eeg_idx=val_eeg_idx,
        test_eeg_idx=test_eeg_idx,
        train_emg_idx=train_emg_idx,
        val_emg_idx=val_emg_idx,
        test_emg_idx=test_emg_idx,
    )

    print(
        f"[done] EEG: {eeg.shape} (train={len(train_eeg_idx)}, val={len(val_eeg_idx)}, test={len(test_eeg_idx)})"
    )
    print(
        f"[done] EMG: {emg.shape} (train={len(train_emg_idx)}, val={len(val_emg_idx)}, test={len(test_emg_idx)})"
    )


if __name__ == "__main__":
    main()
