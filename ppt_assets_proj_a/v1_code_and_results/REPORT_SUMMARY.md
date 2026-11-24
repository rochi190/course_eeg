# SPEC A Experiment Report: Physics-Informed Single-Channel EEG EMG Denoising

## 1. Overall Quantitative Comparison Across Test Tracks

| Track | Experiment | RRMSE_t ↓ | RRMSE_s ↓ | Pearson CC ↑ | Alpha MAE ↓ | Beta MAE ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Test-ID | Exp A0 | 0.8059 | 0.5248 | 0.7507 | 0.0887 | 0.0510 |
| Test-ID | Exp A1 | 0.4699 | 0.2669 | 0.8738 | 0.0643 | 0.0286 |
| Test-ID | Exp A2 | 0.4100 | 0.2330 | 0.9027 | 0.0344 | 0.0223 |
| Test-ID | Exp A3 | 0.4625 | 0.2744 | 0.8790 | 0.0375 | 0.0261 |
| Test-OOD-60 | Exp A0 | 1.2138 | 1.5431 | 0.6099 | 0.1757 | 0.0994 |
| Test-OOD-60 | Exp A1 | 0.8373 | 0.6555 | 0.7213 | 0.0866 | 0.0525 |
| Test-OOD-60 | Exp A2 | 0.6343 | 0.3630 | 0.7972 | 0.0526 | 0.0379 |
| Test-OOD-60 | Exp A3 | 0.5244 | 0.3299 | 0.8388 | 0.0492 | 0.0313 |
| Test-OOD-45 | Exp A0 | 1.3354 | 1.9159 | 0.5757 | 0.1906 | 0.1164 |
| Test-OOD-45 | Exp A1 | 0.9369 | 0.7789 | 0.6788 | 0.1175 | 0.0887 |
| Test-OOD-45 | Exp A2 | 0.7269 | 0.4695 | 0.7567 | 0.0667 | 0.0528 |
| Test-OOD-45 | Exp A3 | 0.5388 | 0.3527 | 0.8283 | 0.0525 | 0.0332 |
| Test-Clean | Exp A0 | 0.4364 | 0.1123 | 0.8998 | 0.0170 | 0.0071 |
| Test-Clean | Exp A1 | 0.2427 | 0.3232 | 0.9820 | 0.0519 | 0.0333 |
| Test-Clean | Exp A2 | 0.2244 | 0.2521 | 0.9812 | 0.0487 | 0.0089 |
| Test-Clean | Exp A3 | 0.0325 | 0.0383 | 0.9991 | 0.0019 | 0.0017 |

## 2. Key Findings

- **Test-OOD (fc = 60 Hz) RRMSE_t improvement (Exp A3 vs. Exp A1)**: **+37.37%** (0.8373 -> 0.5244).
- **Test-OOD (fc = 45 Hz) RRMSE_t improvement (Exp A3 vs. Exp A1)**: **+42.48%** (0.9369 -> 0.5388); vs. Exp A2: **+25.87%** (0.7269 -> 0.5388).
- **Test-Clean Over-denoising Reduction (Exp A3 vs. Exp A1)**: **+86.62%** (0.2427 -> 0.0325).
- **Test-ID RRMSE_t**: Exp A1 = 0.4699, Exp A2 = 0.4100, Exp A3 = 0.4625.

## 3. Per-SNR Highlights (RRMSE_t at -5 dB, 0 dB, +2 dB)

| Track | SNR (dB) | Exp A0 | Exp A1 | Exp A2 | Exp A3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Test-ID | -5 | 0.9609 | 0.5405 | 0.4691 | 0.5231 |
| Test-ID | +0 | 0.6445 | 0.3880 | 0.3361 | 0.3839 |
| Test-ID | +2 | 0.5829 | 0.3478 | 0.3082 | 0.3503 |
| Test-OOD-60 | -5 | 1.4876 | 1.0278 | 0.7589 | 0.5721 |
| Test-OOD-60 | +0 | 0.9146 | 0.6288 | 0.4907 | 0.4558 |
| Test-OOD-60 | +2 | 0.7675 | 0.5303 | 0.4266 | 0.4216 |
| Test-OOD-45 | -5 | 1.6385 | 1.1489 | 0.8790 | 0.5835 |
| Test-OOD-45 | +0 | 0.9973 | 0.6979 | 0.5465 | 0.4728 |
| Test-OOD-45 | +2 | 0.8301 | 0.5880 | 0.4678 | 0.4374 |

## 4. Output Figures

- `outputs/figures/fig1_snr_rrmse.png`
- `outputs/figures/fig2_waveform_psd.png`
- `outputs/figures/fig3_clean_preservation.png`
