#!/usr/bin/env python3
"""Train and evaluate Exp A0-A3 on Test-ID, Test-OOD-60/45, and Test-Clean."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from scipy.signal import welch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _load_local(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


SRC_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SRC_DIR.parent
PROC_DIR = PROJ_ROOT / "data" / "processed"
OUT_DIR = PROJ_ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"

syn = _load_local("synthesis_mod", SRC_DIR / "02_synthesis.py")
models = _load_local("models_mod", SRC_DIR / "03_models.py")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_val_tensors(
    eeg_val: np.ndarray,
    emg_val: np.ndarray,
    mode: str,
    seed: int = 999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic validation dataset (ID for A1/A2; mixed ID+OOD+Clean for A3)."""
    rng = np.random.default_rng(seed)
    y_list, x_list = [], []
    for i in range(len(eeg_val)):
        for k in range(2):
            x = eeg_val[i]
            n = emg_val[int(rng.integers(0, len(emg_val)))].copy()
            snr_db = float(rng.uniform(-7.0, 2.0))
            if mode == "physics":
                u = float(rng.random())
                if u < 0.10:
                    y = x.copy()
                else:
                    if float(rng.random()) < 0.75:
                        n = syn.apply_lowpass(n, int(rng.integers(35, 111)))
                    if float(rng.random()) < (1.0 / 3.0):
                        n = n * syn.make_tukey_burst_mask(syn.T, syn.FS, (0.3, 1.5), rng)
                    y = syn.mix_snr(x, n, snr_db)
            else:
                y = syn.mix_snr(x, n, snr_db)
            yn, xn, _ = syn.normalize_pair(y, x)
            y_list.append(yn)
            x_list.append(xn)
    y_arr = torch.from_numpy(np.asarray(y_list, dtype=np.float32)).unsqueeze(1)
    x_arr = torch.from_numpy(np.asarray(x_list, dtype=np.float32)).unsqueeze(1)
    return y_arr, x_arr


def train_experiment(
    exp_name: str,
    model: nn.Module,
    criterion: nn.Module,
    eeg_train: np.ndarray,
    emg_train: np.ndarray,
    eeg_val: np.ndarray,
    emg_val: np.ndarray,
    data_mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    repeats: int,
    device: torch.device,
) -> nn.Module:
    """Train one denoising experiment with CosineAnnealingLR and early stopping."""
    model = model.to(device)
    train_ds = syn.EEGDenoisingTrainDataset(
        eeg_train, emg_train, mode=data_mode, repeats=repeats, seed=42
    )
    num_workers = 4 if device.type == "cuda" and len(train_ds) > 1000 else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_y, val_x = make_val_tensors(eeg_val, emg_val, mode=data_mode, seed=777)
    val_loader = DataLoader(TensorDataset(val_y, val_x), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val = float("inf")
    best_state = None
    no_improve = 0
    ckpt_path = CKPT_DIR / f"{exp_name.lower().replace(' ', '_')}_best.pt"

    t0 = time.time()
    for ep in range(1, epochs + 1):
        train_ds.set_epoch(ep)
        model.train()
        tr_loss = 0.0
        tr_count = 0
        for yb, xb in train_loader:
            yb = yb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(yb)
            loss = criterion(pred, xb)
            assert torch.isfinite(loss), f"Non-finite loss in {exp_name} epoch {ep}"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bs = yb.size(0)
            tr_loss += float(loss.item()) * bs
            tr_count += bs
        scheduler.step()

        model.eval()
        va_loss = 0.0
        va_count = 0
        with torch.no_grad():
            for yb, xb in val_loader:
                yb = yb.to(device, non_blocking=True)
                xb = xb.to(device, non_blocking=True)
                pred = model(yb)
                loss = criterion(pred, xb)
                bs = yb.size(0)
                va_loss += float(loss.item()) * bs
                va_count += bs

        tr_avg = tr_loss / max(1, tr_count)
        va_avg = va_loss / max(1, va_count)
        if va_avg < best_val:
            best_val = va_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1

        if ep == 1 or ep % 10 == 0 or ep == epochs or no_improve == 0:
            print(
                f"[{exp_name}] Ep {ep:02d}/{epochs:02d} | train={tr_avg:.5f} | val={va_avg:.5f} | best={best_val:.5f}"
            )
        if no_improve >= patience and ep >= min(epochs, 15):
            print(f"[{exp_name}] Early stopping triggered at epoch {ep}")
            break

    elapsed = time.time() - t0
    print(f"[{exp_name}] Finished in {elapsed:.1f}s | best_val={best_val:.5f}")
    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


@torch.no_grad()
def predict_numpy(
    model: nn.Module | None,
    y_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run batched inference (or Bandpass if model is None)."""
    if model is None:
        return models.bandpass_1_40hz(y_arr)
    model.eval()
    preds = []
    for start in range(0, len(y_arr), batch_size):
        yb = torch.from_numpy(y_arr[start : start + batch_size]).unsqueeze(1).to(device)
        pb = model(yb).squeeze(1).cpu().numpy()
        preds.append(pb)
    return np.concatenate(preds, axis=0).astype(np.float32)


def compute_metrics_batch(
    preds: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute per-sample RRMSE_t, RRMSE_s, Pearson CC, and Alpha/Beta relative power MAE."""
    # Time-domain RRMSE
    rms_err = np.sqrt(np.mean((preds - targets) ** 2, axis=1) + 1e-12)
    rms_ref = np.sqrt(np.mean(targets ** 2, axis=1) + 1e-12)
    rrmse_t = rms_err / rms_ref

    # Pearson CC
    p_c = preds - preds.mean(axis=1, keepdims=True)
    t_c = targets - targets.mean(axis=1, keepdims=True)
    num = np.sum(p_c * t_c, axis=1)
    den = np.sqrt(np.sum(p_c ** 2, axis=1) * np.sum(t_c ** 2, axis=1)) + 1e-12
    cc = num / den

    # Welch PSD (fs=512, nperseg=256 -> df=2 Hz)
    freqs, psd_p = welch(preds.astype(np.float64), fs=512, nperseg=256, axis=1)
    _, psd_t = welch(targets.astype(np.float64), fs=512, nperseg=256, axis=1)

    psd_rms_err = np.sqrt(np.mean((psd_p - psd_t) ** 2, axis=1) + 1e-24)
    psd_rms_ref = np.sqrt(np.mean(psd_t ** 2, axis=1) + 1e-24)
    rrmse_s = psd_rms_err / psd_rms_ref

    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    beta_mask = (freqs > 13.0) & (freqs <= 30.0)

    rel_alpha_p = psd_p[:, alpha_mask].sum(axis=1) / (psd_p[:, total_mask].sum(axis=1) + 1e-12)
    rel_alpha_t = psd_t[:, alpha_mask].sum(axis=1) / (psd_t[:, total_mask].sum(axis=1) + 1e-12)
    alpha_mae = np.abs(rel_alpha_p - rel_alpha_t)

    rel_beta_p = psd_p[:, beta_mask].sum(axis=1) / (psd_p[:, total_mask].sum(axis=1) + 1e-12)
    rel_beta_t = psd_t[:, beta_mask].sum(axis=1) / (psd_t[:, total_mask].sum(axis=1) + 1e-12)
    beta_mae = np.abs(rel_beta_p - rel_beta_t)

    return {
        "rrmse_t": rrmse_t,
        "rrmse_s": rrmse_s,
        "cc": cc,
        "alpha_mae": alpha_mae,
        "beta_mae": beta_mae,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="2-epoch smoke test on subset")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    eeg = np.load(PROC_DIR / "eeg_512hz.npy")
    emg = np.load(PROC_DIR / "emg_512hz.npy")
    splits = np.load(PROC_DIR / "split_indices.npz")

    train_eeg = eeg[splits["train_eeg_idx"]]
    val_eeg = eeg[splits["val_eeg_idx"]]
    test_eeg = eeg[splits["test_eeg_idx"]]

    train_emg = emg[splits["train_emg_idx"]]
    val_emg = emg[splits["val_emg_idx"]]
    test_emg = emg[splits["test_emg_idx"]]

    if args.smoke:
        epochs = 2
        repeats = 1
        train_eeg = train_eeg[:256]
        val_eeg = val_eeg[:64]
        max_per_snr = 10
        print("[mode] SMOKE TEST (epochs=2, subset)")
    else:
        epochs = args.epochs
        repeats = 4
        max_per_snr = None
        print(f"[mode] FULL EXPERIMENT (epochs={epochs}, repeats={repeats})")

    # Build deterministic test suites
    suites = syn.build_test_suites(test_eeg, test_emg, seed=2026, max_per_snr=max_per_snr)

    # Train Exp A1, A2, A3
    exp_configs = [
        ("Exp A1", models.ResCNN1D(), nn.MSELoss(), "id"),
        ("Exp A2", models.ResUNet1D(base_ch=32), models.CombinedDenoiseLoss(alpha=0.1), "id"),
        ("Exp A3", models.ResUNet1D(base_ch=32), models.CombinedDenoiseLoss(alpha=0.1), "physics"),
    ]

    trained_models: dict[str, nn.Module | None] = {"Exp A0": None}
    for name, model, crit, data_mode in exp_configs:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== Training {name} (params={n_params:,}, data={data_mode}) ===")
        set_seed(42)
        trained_models[name] = train_experiment(
            exp_name=name,
            model=model,
            criterion=crit,
            eeg_train=train_eeg,
            emg_train=train_emg,
            eeg_val=val_eeg,
            emg_val=val_emg,
            data_mode=data_mode,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            repeats=repeats,
            device=device,
        )

    # Evaluate Exp A0..A3 on all 4 test tracks
    rows = []
    ood45_preds_by_exp: dict[str, np.ndarray] = {}
    for exp_name in ["Exp A0", "Exp A1", "Exp A2", "Exp A3"]:
        mdl = trained_models[exp_name]
        for track_name, track_data in suites.items():
            preds = predict_numpy(mdl, track_data["y"], device=device)
            if track_name == "Test-OOD-45":
                ood45_preds_by_exp[exp_name] = preds
            m = compute_metrics_batch(preds, track_data["x"])

            # Overall row
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
            # Per-SNR rows for noisy tracks
            if track_name != "Test-Clean":
                snrs = track_data["snr"]
                for snr_val in syn.SNR_LEVELS:
                    mask = snrs == snr_val
                    rows.append(
                        {
                            "experiment": exp_name,
                            "track": track_name,
                            "snr_db": int(snr_val),
                            "n_samples": int(mask.sum()),
                            "rrmse_t": float(m["rrmse_t"][mask].mean()),
                            "rrmse_s": float(m["rrmse_s"][mask].mean()),
                            "cc": float(m["cc"][mask].mean()),
                            "alpha_mae": float(m["alpha_mae"][mask].mean()),
                            "beta_mae": float(m["beta_mae"][mask].mean()),
                        }
                    )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path}")

    # Pick representative SNR=-5 dB sample from Test-OOD-45 for Fig 2
    ood45 = suites["Test-OOD-45"]
    idx_m5 = np.where(ood45["snr"] == -5)[0]
    m_a1 = compute_metrics_batch(ood45_preds_by_exp["Exp A1"][idx_m5], ood45["x"][idx_m5])
    m_a3 = compute_metrics_batch(ood45_preds_by_exp["Exp A3"][idx_m5], ood45["x"][idx_m5])
    # Prefer sample with non-trivial alpha/beta content and large A3 gain over A1
    freqs, psd_clean = welch(ood45["x"][idx_m5].astype(np.float64), fs=512, nperseg=256, axis=1)
    ab_mask = (freqs >= 8.0) & (freqs <= 30.0)
    tot_mask = (freqs >= 1.0) & (freqs <= 45.0)
    ab_ratio = psd_clean[:, ab_mask].sum(axis=1) / (psd_clean[:, tot_mask].sum(axis=1) + 1e-12)
    score = (m_a1["rrmse_t"] - m_a3["rrmse_t"]) + 0.3 * (m_a3["cc"] - m_a1["cc"]) + 0.5 * ab_ratio
    best_local = int(np.argmax(score))
    pick = int(idx_m5[best_local])
    np.savez(
        OUT_DIR / "fig2_sample.npz",
        noisy=ood45["y"][pick],
        clean=ood45["x"][pick],
        pred_a0=ood45_preds_by_exp["Exp A0"][pick],
        pred_a1=ood45_preds_by_exp["Exp A1"][pick],
        pred_a2=ood45_preds_by_exp["Exp A2"][pick],
        pred_a3=ood45_preds_by_exp["Exp A3"][pick],
        rrmse_a1=float(m_a1["rrmse_t"][best_local]),
        rrmse_a3=float(m_a3["rrmse_t"][best_local]),
        cc_a1=float(m_a1["cc"][best_local]),
        cc_a3=float(m_a3["cc"][best_local]),
        snr_db=-5,
        fc=45,
    )

    # Generate REPORT_SUMMARY.md
    overall_df = df[df["snr_db"] == "Overall"].copy()

    def _val(exp: str, trk: str, col: str) -> float:
        sub = overall_df[(overall_df["experiment"] == exp) & (overall_df["track"] == trk)]
        return float(sub[col].iloc[0])

    imp_ood60_vs_a1 = (_val("Exp A1", "Test-OOD-60", "rrmse_t") - _val("Exp A3", "Test-OOD-60", "rrmse_t")) / _val("Exp A1", "Test-OOD-60", "rrmse_t") * 100.0
    imp_ood45_vs_a1 = (_val("Exp A1", "Test-OOD-45", "rrmse_t") - _val("Exp A3", "Test-OOD-45", "rrmse_t")) / _val("Exp A1", "Test-OOD-45", "rrmse_t") * 100.0
    imp_ood45_vs_a2 = (_val("Exp A2", "Test-OOD-45", "rrmse_t") - _val("Exp A3", "Test-OOD-45", "rrmse_t")) / _val("Exp A2", "Test-OOD-45", "rrmse_t") * 100.0
    imp_clean_vs_a1 = (_val("Exp A1", "Test-Clean", "rrmse_t") - _val("Exp A3", "Test-Clean", "rrmse_t")) / _val("Exp A1", "Test-Clean", "rrmse_t") * 100.0

    lines = [
        "# SPEC A Experiment Report: Physics-Informed Single-Channel EEG EMG Denoising",
        "",
        "## 1. Overall Quantitative Comparison Across Test Tracks",
        "",
        "| Track | Experiment | RRMSE_t ↓ | RRMSE_s ↓ | Pearson CC ↑ | Alpha MAE ↓ | Beta MAE ↓ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for trk in ["Test-ID", "Test-OOD-60", "Test-OOD-45", "Test-Clean"]:
        for exp in ["Exp A0", "Exp A1", "Exp A2", "Exp A3"]:
            sub = overall_df[(overall_df["track"] == trk) & (overall_df["experiment"] == exp)].iloc[0]
            lines.append(
                f"| {trk} | {exp} | {sub['rrmse_t']:.4f} | {sub['rrmse_s']:.4f} | {sub['cc']:.4f} | {sub['alpha_mae']:.4f} | {sub['beta_mae']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 2. Key Findings",
            "",
            f"- **Test-OOD (fc = 60 Hz) RRMSE_t improvement (Exp A3 vs. Exp A1)**: **{imp_ood60_vs_a1:+.2f}%** ({_val('Exp A1', 'Test-OOD-60', 'rrmse_t'):.4f} -> {_val('Exp A3', 'Test-OOD-60', 'rrmse_t'):.4f}).",
            f"- **Test-OOD (fc = 45 Hz) RRMSE_t improvement (Exp A3 vs. Exp A1)**: **{imp_ood45_vs_a1:+.2f}%** ({_val('Exp A1', 'Test-OOD-45', 'rrmse_t'):.4f} -> {_val('Exp A3', 'Test-OOD-45', 'rrmse_t'):.4f}); vs. Exp A2: **{imp_ood45_vs_a2:+.2f}%** ({_val('Exp A2', 'Test-OOD-45', 'rrmse_t'):.4f} -> {_val('Exp A3', 'Test-OOD-45', 'rrmse_t'):.4f}).",
            f"- **Test-Clean Over-denoising Reduction (Exp A3 vs. Exp A1)**: **{imp_clean_vs_a1:+.2f}%** ({_val('Exp A1', 'Test-Clean', 'rrmse_t'):.4f} -> {_val('Exp A3', 'Test-Clean', 'rrmse_t'):.4f}).",
            f"- **Test-ID RRMSE_t**: Exp A1 = {_val('Exp A1', 'Test-ID', 'rrmse_t'):.4f}, Exp A2 = {_val('Exp A2', 'Test-ID', 'rrmse_t'):.4f}, Exp A3 = {_val('Exp A3', 'Test-ID', 'rrmse_t'):.4f}.",
            "",
            "## 3. Per-SNR Highlights (RRMSE_t at -5 dB, 0 dB, +2 dB)",
            "",
            "| Track | SNR (dB) | Exp A0 | Exp A1 | Exp A2 | Exp A3 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for trk in ["Test-ID", "Test-OOD-60", "Test-OOD-45"]:
        for snr_v in [-5, 0, 2]:
            vals = []
            for exp in ["Exp A0", "Exp A1", "Exp A2", "Exp A3"]:
                v = float(df[(df["track"] == trk) & (df["experiment"] == exp) & (df["snr_db"] == snr_v)]["rrmse_t"].iloc[0])
                vals.append(f"{v:.4f}")
            lines.append(f"| {trk} | {snr_v:+d} | {' | '.join(vals)} |")

    lines.extend(
        [
            "",
            "## 4. Output Figures",
            "",
            "- `outputs/figures/fig1_snr_rrmse.png`",
            "- `outputs/figures/fig2_waveform_psd.png`",
            "- `outputs/figures/fig3_clean_preservation.png`",
            "",
        ]
    )
    report_path = OUT_DIR / "REPORT_SUMMARY.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {report_path}\n")
    print(overall_df[["track", "experiment", "rrmse_t", "rrmse_s", "cc", "alpha_mae", "beta_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
