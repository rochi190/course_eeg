# SPEC B-V2 Report: Rescuing Strict Subject-Independent AD vs. FTD Classification

## 1. V2 Subject-Level Performance (Strict 5-Fold StratifiedGroupKFold × 5 Seeds)

| Task | Experiment | Macro-F1 (%) | Accuracy (%) | AUROC | Sens (%) | Spec (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| AD_vs_CN | Exp B4a (Riemannian Epoch) | 79.26 ± 1.72 | 79.38 ± 1.75 | 0.8421 ± 0.0078 | 78.3 ± 3.6 | 80.7 ± 1.9 |
| AD_vs_CN | Exp B4b (Riemannian Subject) | 78.70 ± 1.69 | 78.77 ± 1.69 | 0.8611 ± 0.0095 | 75.0 ± 4.4 | 83.4 ± 4.5 |
| AD_vs_CN | Exp B5a (Neuro ElasticNet) | 72.81 ± 1.71 | 72.92 ± 1.75 | 0.7828 ± 0.0114 | 71.1 ± 3.7 | 75.2 ± 2.9 |
| AD_vs_CN | Exp B5b (Neuro LightGBM) | 69.94 ± 3.94 | 70.15 ± 3.86 | 0.7900 ± 0.0210 | 70.6 ± 3.2 | 69.7 ± 6.2 |
| AD_vs_CN | Exp B6 (Spectral ResGCN <25K) | 71.54 ± 4.19 | 71.69 ± 4.16 | 0.7845 ± 0.0378 | 69.4 ± 6.2 | 74.5 ± 8.3 |
| AD_vs_CN | Exp B7 (Multimodal Ensemble) | 75.55 ± 1.75 | 75.69 ± 1.69 | 0.8473 ± 0.0138 | 74.4 ± 3.6 | 77.2 ± 5.2 |
| AD_vs_FTD | Exp B4a (Riemannian Epoch) | 58.66 ± 2.88 | 60.68 ± 2.79 | 0.6278 ± 0.0507 | 67.8 ± 3.2 | 49.6 ± 3.9 |
| AD_vs_FTD | Exp B4b (Riemannian Subject) | 59.82 ± 2.74 | 62.37 ± 3.26 | 0.6174 ± 0.0680 | 71.7 ± 6.0 | 47.8 ± 3.1 |
| AD_vs_FTD | Exp B5a (Neuro ElasticNet) | 63.10 ± 6.00 | 64.41 ± 5.99 | 0.6766 ± 0.0797 | 67.8 ± 7.8 | 59.1 ± 8.5 |
| AD_vs_FTD | Exp B5b (Neuro LightGBM) | 59.30 ± 4.44 | 62.37 ± 3.87 | 0.6244 ± 0.0605 | 73.3 ± 4.2 | 45.2 ± 7.9 |
| AD_vs_FTD | Exp B6 (Spectral ResGCN <25K) | 51.37 ± 4.39 | 54.24 ± 5.22 | 0.5304 ± 0.0495 | 63.3 ± 11.4 | 40.0 ± 8.4 |
| AD_vs_FTD | Exp B7 (Multimodal Ensemble) | 62.81 ± 7.97 | 65.76 ± 8.59 | 0.6628 ± 0.0756 | 77.2 ± 12.2 | 47.8 ± 5.3 |
| FTD_vs_CN | Exp B4a (Riemannian Epoch) | 66.46 ± 4.38 | 66.92 ± 4.17 | 0.7523 ± 0.0256 | 63.5 ± 9.0 | 69.7 ± 2.9 |
| FTD_vs_CN | Exp B4b (Riemannian Subject) | 71.19 ± 4.50 | 71.54 ± 4.39 | 0.7712 ± 0.0191 | 70.4 ± 8.4 | 72.4 ± 7.3 |
| FTD_vs_CN | Exp B5a (Neuro ElasticNet) | 68.78 ± 3.87 | 69.23 ± 3.60 | 0.7481 ± 0.0391 | 68.7 ± 11.7 | 69.7 ± 8.2 |
| FTD_vs_CN | Exp B5b (Neuro LightGBM) | 65.36 ± 6.42 | 66.15 ± 6.61 | 0.6945 ± 0.0439 | 58.3 ± 6.6 | 72.4 ± 9.8 |
| FTD_vs_CN | Exp B6 (Spectral ResGCN <25K) | 60.16 ± 4.18 | 60.77 ± 3.99 | 0.7118 ± 0.0321 | 60.0 ± 14.2 | 61.4 ± 8.9 |
| FTD_vs_CN | Exp B7 (Multimodal Ensemble) | 72.04 ± 3.45 | 72.69 ± 3.16 | 0.7982 ± 0.0341 | 66.1 ± 8.4 | 77.9 ± 3.9 |
| AD_vs_FTD_vs_CN | Exp B4a (Riemannian Epoch) | 58.95 ± 4.71 | 60.23 ± 4.95 | 0.7450 ± 0.0212 | N/A | N/A |
| AD_vs_FTD_vs_CN | Exp B4b (Riemannian Subject) | 62.01 ± 2.77 | 63.86 ± 2.59 | 0.7695 ± 0.0176 | N/A | N/A |
| AD_vs_FTD_vs_CN | Exp B5a (Neuro ElasticNet) | 54.45 ± 1.41 | 55.00 ± 1.30 | 0.7198 ± 0.0091 | N/A | N/A |
| AD_vs_FTD_vs_CN | Exp B5b (Neuro LightGBM) | 47.43 ± 4.63 | 49.77 ± 4.21 | 0.6680 ± 0.0287 | N/A | N/A |
| AD_vs_FTD_vs_CN | Exp B6 (Spectral ResGCN <25K) | 52.39 ± 3.05 | 54.55 ± 2.41 | 0.6933 ± 0.0313 | N/A | N/A |
| AD_vs_FTD_vs_CN | Exp B7 (Multimodal Ensemble) | 59.67 ± 3.26 | 60.68 ± 2.96 | 0.7713 ± 0.0150 | N/A | N/A |

## 2. Direct V1 vs. V2 & 2026 SOTA Comparison (Subject Macro-F1 %)

- **AD_vs_CN** (2026 SOTA = 75.82%): V1 Best `Exp B2b (Linear SVM)` = **80.44%** -> V2 Best `Exp B4a (Riemannian Epoch)` = **79.26%** (**-1.17 pp vs. V1**).
- **AD_vs_FTD** (2026 SOTA = 70.65%): V1 Best `Exp B2b (Linear SVM)` = **56.19%** -> V2 Best `Exp B5a (Neuro ElasticNet)` = **63.10%** (**+6.91 pp vs. V1**).
- **FTD_vs_CN** (2026 SOTA = 56.35%): V1 Best `Exp B2a (LightGBM)` = **77.69%** -> V2 Best `Exp B7 (Multimodal Ensemble)` = **72.04%** (**-5.65 pp vs. V1**).
- **AD_vs_FTD_vs_CN**: V1 Best `Exp B2b (Linear SVM)` = **60.72%** -> V2 Best `Exp B4b (Riemannian Subject)` = **62.01%** (**+1.29 pp vs. V1**).

## 3. Output Figures

- `outputs/figures/fig4_ad_ftd_rescue.png`
- `outputs/figures/fig5_v2_all_tasks_vs_2026sota.png`
- `outputs/figures/fig6_neuro_biomarkers_boxplot.png`
