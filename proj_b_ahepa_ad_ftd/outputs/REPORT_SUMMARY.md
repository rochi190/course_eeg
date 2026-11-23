# SPEC B Experiment Report: Strict Subject-Independent AHEPA (ds004504) Classification

## 1. Subject-Level Performance (Soft Voting Across 4s Epochs, Mean ± Std)

| Task | Experiment | Macro-F1 (%) | Accuracy (%) | AUROC | Sens / Spec (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| AD_vs_CN | Exp B0 (Leaky ShallowConvNet) | 100.00 ± 0.00 | 100.00 ± 0.00 | 1.0000 ± 0.0000 | 100.0 / 100.0 |
| AD_vs_CN | Exp B1 (ShallowConvNet) | 80.18 ± 1.22 | 80.31 ± 1.29 | 0.8703 ± 0.0227 | 77.8 / 83.4 |
| AD_vs_CN | Exp B2a (LightGBM) | 78.68 ± 1.27 | 78.77 ± 1.29 | 0.8351 ± 0.0083 | 76.7 / 81.4 |
| AD_vs_CN | Exp B2b (Linear SVM) | 80.44 ± 2.70 | 80.62 ± 2.79 | 0.8872 ± 0.0269 | 81.1 / 80.0 |
| AD_vs_CN | Exp B3 (LightGAT) | 75.29 ± 1.91 | 75.38 ± 1.88 | 0.8272 ± 0.0110 | 73.3 / 77.9 |
| AD_vs_FTD | Exp B0 (Leaky ShallowConvNet) | 100.00 ± 0.00 | 100.00 ± 0.00 | 1.0000 ± 0.0000 | 100.0 / 100.0 |
| AD_vs_FTD | Exp B1 (ShallowConvNet) | 52.04 ± 2.66 | 54.92 ± 2.27 | 0.5580 ± 0.0411 | 65.0 / 39.1 |
| AD_vs_FTD | Exp B2a (LightGBM) | 48.99 ± 4.35 | 54.24 ± 3.79 | 0.4635 ± 0.0656 | 70.6 / 28.7 |
| AD_vs_FTD | Exp B2b (Linear SVM) | 56.19 ± 3.52 | 59.32 ± 3.79 | 0.5882 ± 0.0332 | 70.6 / 41.7 |
| AD_vs_FTD | Exp B3 (LightGAT) | 46.38 ± 4.92 | 50.51 ± 5.80 | 0.4374 ± 0.0487 | 63.3 / 30.4 |
| FTD_vs_CN | Exp B0 (Leaky ShallowConvNet) | 100.00 ± 0.00 | 100.00 ± 0.00 | 1.0000 ± 0.0000 | 100.0 / 100.0 |
| FTD_vs_CN | Exp B1 (ShallowConvNet) | 66.38 ± 2.85 | 67.31 ± 3.04 | 0.7073 ± 0.0353 | 57.4 / 75.2 |
| FTD_vs_CN | Exp B2a (LightGBM) | 77.69 ± 0.76 | 78.46 ± 0.86 | 0.8213 ± 0.0128 | 67.8 / 86.9 |
| FTD_vs_CN | Exp B2b (Linear SVM) | 74.37 ± 2.61 | 75.00 ± 2.36 | 0.7721 ± 0.0389 | 67.8 / 80.7 |
| FTD_vs_CN | Exp B3 (LightGAT) | 70.64 ± 5.67 | 71.54 ± 5.34 | 0.7328 ± 0.0182 | 61.7 / 79.3 |
| AD_vs_FTD_vs_CN | Exp B0 (Leaky ShallowConvNet) | 100.00 ± 0.00 | 100.00 ± 0.00 | 1.0000 ± 0.0000 | N/A (3-class) |
| AD_vs_FTD_vs_CN | Exp B1 (ShallowConvNet) | 56.14 ± 1.43 | 58.18 ± 1.48 | 0.7163 ± 0.0123 | N/A (3-class) |
| AD_vs_FTD_vs_CN | Exp B2a (LightGBM) | 55.79 ± 1.40 | 59.32 ± 0.51 | 0.7222 ± 0.0154 | N/A (3-class) |
| AD_vs_FTD_vs_CN | Exp B2b (Linear SVM) | 60.72 ± 3.21 | 64.09 ± 3.07 | 0.7723 ± 0.0108 | N/A (3-class) |
| AD_vs_FTD_vs_CN | Exp B3 (LightGAT) | 55.13 ± 4.73 | 58.86 ± 4.21 | 0.7220 ± 0.0233 | N/A (3-class) |

## 2. Segment-Level Performance (Mean ± Std)

| Task | Experiment | Macro-F1 (%) | Accuracy (%) | AUROC |
| :--- | :--- | :---: | :---: | :---: |
| AD_vs_CN | Exp B0 (Leaky ShallowConvNet) | 98.57 ± 0.03 | 98.59 ± 0.03 | 0.9988 ± 0.0001 |
| AD_vs_CN | Exp B1 (ShallowConvNet) | 75.19 ± 1.67 | 75.30 ± 1.69 | 0.8281 ± 0.0160 |
| AD_vs_CN | Exp B2a (LightGBM) | 70.27 ± 0.93 | 70.41 ± 0.96 | 0.7682 ± 0.0110 |
| AD_vs_CN | Exp B2b (Linear SVM) | 70.96 ± 1.69 | 71.11 ± 1.69 | 0.7884 ± 0.0204 |
| AD_vs_CN | Exp B3 (LightGAT) | 69.65 ± 0.62 | 69.77 ± 0.66 | 0.7622 ± 0.0065 |
| AD_vs_FTD | Exp B0 (Leaky ShallowConvNet) | 97.47 ± 0.16 | 97.66 ± 0.15 | 0.9968 ± 0.0004 |
| AD_vs_FTD | Exp B1 (ShallowConvNet) | 54.23 ± 1.12 | 56.84 ± 1.50 | 0.5561 ± 0.0266 |
| AD_vs_FTD | Exp B2a (LightGBM) | 48.76 ± 3.42 | 53.12 ± 3.22 | 0.4707 ± 0.0588 |
| AD_vs_FTD | Exp B2b (Linear SVM) | 53.86 ± 1.86 | 56.78 ± 1.95 | 0.5452 ± 0.0281 |
| AD_vs_FTD | Exp B3 (LightGAT) | 46.82 ± 2.13 | 50.42 ± 2.85 | 0.4520 ± 0.0263 |
| FTD_vs_CN | Exp B0 (Leaky ShallowConvNet) | 97.69 ± 0.28 | 97.77 ± 0.27 | 0.9976 ± 0.0004 |
| FTD_vs_CN | Exp B1 (ShallowConvNet) | 62.51 ± 2.26 | 63.97 ± 2.30 | 0.6685 ± 0.0310 |
| FTD_vs_CN | Exp B2a (LightGBM) | 70.11 ± 1.38 | 71.67 ± 1.13 | 0.7441 ± 0.0143 |
| FTD_vs_CN | Exp B2b (Linear SVM) | 65.14 ± 2.74 | 66.38 ± 2.75 | 0.6969 ± 0.0308 |
| FTD_vs_CN | Exp B3 (LightGAT) | 64.07 ± 1.46 | 65.22 ± 1.28 | 0.6805 ± 0.0115 |
| AD_vs_FTD_vs_CN | Exp B0 (Leaky ShallowConvNet) | 96.89 ± 0.12 | 97.02 ± 0.09 | 0.9974 ± 0.0001 |
| AD_vs_FTD_vs_CN | Exp B1 (ShallowConvNet) | 51.40 ± 1.13 | 53.88 ± 0.81 | 0.6861 ± 0.0078 |
| AD_vs_FTD_vs_CN | Exp B2a (LightGBM) | 48.60 ± 1.34 | 51.74 ± 1.15 | 0.6716 ± 0.0135 |
| AD_vs_FTD_vs_CN | Exp B2b (Linear SVM) | 50.39 ± 0.58 | 53.20 ± 0.66 | 0.6907 ± 0.0083 |
| AD_vs_FTD_vs_CN | Exp B3 (LightGAT) | 48.17 ± 2.46 | 51.62 ± 2.51 | 0.6672 ± 0.0169 |

## 3. Comparison vs. 2026 Literature Subject-Level Macro-F1 Benchmarks

- **AD_vs_CN** (2026 SOTA = 75.82%): B0 Leaky = 100.00% -> B1 Strict = 80.18% (leakage collapse = -19.82 pp) | B2a LightGBM = 78.68% | B2b Linear SVM = 80.44% | B3 LightGAT = 75.29% (-0.53 pp vs. 2026 SOTA).
- **AD_vs_FTD** (2026 SOTA = 70.65%): B0 Leaky = 100.00% -> B1 Strict = 52.04% (leakage collapse = -47.96 pp) | B2a LightGBM = 48.99% | B2b Linear SVM = 56.19% | B3 LightGAT = 46.38% (-24.27 pp vs. 2026 SOTA).
- **FTD_vs_CN** (2026 SOTA = 56.35%): B0 Leaky = 100.00% -> B1 Strict = 66.38% (leakage collapse = -33.62 pp) | B2a LightGBM = 77.69% | B2b Linear SVM = 74.37% | B3 LightGAT = 70.64% (+14.29 pp vs. 2026 SOTA).

## 4. Output Figures

- `outputs/figures/fig1_leakage_gap.png`
- `outputs/figures/fig2_topomap_plv.png`
- `outputs/figures/fig3_f1_comparison.png`
