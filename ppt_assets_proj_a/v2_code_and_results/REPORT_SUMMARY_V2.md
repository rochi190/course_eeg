# SPEC A-V2 Report: Disentangled OOD Ablation, Cross-Filter Stress Tests & Real-EEG Zero-Shot

## 1. Disentangled Ablation & Cross-Filter Stress Tracks (Overall RRMSE_t ↓ / CC ↑)

| Track | Exp A0 | Exp A1 (ResCNN) | Exp A2 (UNet ID) | Exp A3-noLP (Burst+Clean) | **Exp A3 (Full Physics)** | A3 vs A3-noLP RRMSE_t |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Test-ID | 0.8280 (0.744) | 0.4714 (0.871) | 0.4142 (0.899) | 0.4606 (0.878) | 0.4664 (0.876) | **-1.3%** |
| Test-Pure-LP45 | 1.4521 (0.555) | 1.0171 (0.651) | 0.7882 (0.730) | 1.3354 (0.615) | 0.6261 (0.768) | **+53.1%** |
| Test-Pure-Burst | 0.7790 (0.761) | 0.4756 (0.878) | 0.4077 (0.905) | 0.3681 (0.924) | 0.3685 (0.924) | **-0.1%** |
| Test-ExpDecay | 1.1938 (0.619) | 0.8852 (0.702) | 0.6883 (0.770) | 0.8307 (0.723) | 0.6618 (0.763) | **+20.3%** |
| Test-RC1st | 1.4198 (0.565) | 0.9741 (0.668) | 0.7666 (0.736) | 1.0681 (0.669) | 0.7003 (0.745) | **+34.4%** |
| Test-OOD-45 | 1.3798 (0.568) | 0.9666 (0.671) | 0.7515 (0.747) | 1.1884 (0.657) | 0.5528 (0.818) | **+53.5%** |
| Test-Composite-Drift | 1.4646 (0.551) | 1.0563 (0.635) | 0.8038 (0.722) | 1.3869 (0.599) | 0.6380 (0.760) | **+54.0%** |
| Test-Clean | 0.4364 (0.900) | 0.2427 (0.982) | 0.2244 (0.981) | 0.0387 (1.000) | 0.0325 (0.999) | **+16.1%** |

*Note: `Test-Composite-Drift` adds unseen 0.5–1.5 Hz baseline drift (`RMS = 0.35 * RMS(clean_EEG)`) as an out-of-scope limitation stress test.*

## 2. Real Clinical EEG Zero-Shot Validation (`ds004504` Raw `sub-001` & `sub-037`, 512 Hz)

| Experiment | High-EMG 30–80 Hz Drop (dB) ↑ | High-EMG Alpha Retention | Occipital Alpha Power Ratio | Occipital PAF Shift (Hz) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| Exp A0 | 10.15 ± 0.94 | 1.000 | 1.000 | 0.003 |
| Exp A1 | 16.00 ± 3.30 | 0.801 | 0.768 | 0.043 |
| Exp A2 | 12.75 ± 1.28 | 0.640 | 0.809 | 0.011 |
| Exp A3-noLP | 13.87 ± 2.08 | 0.751 | 0.923 | 0.008 |
| Exp A3 | 13.47 ± 0.72 | 0.554 | 0.855 | 0.015 |

## 3. Single-Window (2s, 1,024 pts) Inference Latency & Parameters

| Experiment | Trainable Params | CPU Latency (ms, 4 threads) | GPU Latency (ms) |
| :--- | :---: | :---: | :---: |
| Exp A1 | 401,530 | 30.38 ± 2.20 | 3.16 ± 0.47 |
| Exp A2 | 4,108,545 | 45.31 ± 7.16 | 5.26 ± 0.80 |
| Exp A3-noLP | 4,108,545 | 45.67 ± 7.51 | 5.23 ± 0.69 |
| Exp A3 | 4,108,545 | 45.75 ± 7.42 | 5.20 ± 0.60 |

## 4. Output Figures

- `outputs/figures/fig4_disentangle_ablation.png`
- `outputs/figures/fig5_real_eeg_zeroshot.png`
