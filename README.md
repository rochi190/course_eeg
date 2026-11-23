# EEG Deep Learning Experiments (`course_eeg`)

本專案包含兩項完整的腦波（EEG）深度學習與神經科學先驗實驗（皆已於 Google Colab T4 實測執行並產出完整報表與 300 DPI 圖表）：

1. **SPEC A (`proj_a_eegdenoise/`)**：克服容積傳導頻譜偏移之單通道 EEG 肌電偽影（EMG）物理導向去噪（EEGdenoiseNet，512 Hz）。
2. **SPEC B (`proj_b_ahepa_ad_ftd/`)**：嚴格跨受試者驗證下之阿茲海默症（AD）與額顳葉失智症（FTD）腦功能連結圖神經網路分類（OpenNeuro `ds004504`，88 位受試者）。

---

## 目錄結構

```text
course_eeg/
├── a_eegdenoisenet.md              # SPEC A 實驗規格書
├── b_ahepa_ad_vs_ftd.md            # SPEC B 實驗規格書
├── proj_a_eegdenoise/
│   ├── src/                        # 01_download_data.py ~ 05_plot_figures.py, colab_run_all_a.py
│   └── outputs/
│       ├── metrics_summary.csv
│       ├── REPORT_SUMMARY.md
│       ├── fig2_sample.npz
│       └── figures/                # fig1_snr_rrmse.png, fig2_waveform_psd.png, fig3_clean_preservation.png
└── proj_b_ahepa_ad_ftd/
    ├── src/                        # 01_download_bids.py ~ 05_plot_figures.py, colab_run_all_b.py
    └── outputs/
        ├── metrics_summary.csv
        ├── metrics_per_seed.csv
        ├── REPORT_SUMMARY.md
        └── figures/                # fig1_leakage_gap.png, fig2_topomap_plv.png, fig3_f1_comparison.png
```

---

## SPEC A 核心結果（70 Epochs, Tesla T4）

- **報表與數據**：[proj_a_eegdenoise/outputs/REPORT_SUMMARY.md](proj_a_eegdenoise/outputs/REPORT_SUMMARY.md) ｜ [metrics_summary.csv](proj_a_eegdenoise/outputs/metrics_summary.csv)

| Track | Exp A0 (Bandpass 1–40 Hz) RRMSE_t ↓ | Exp A1 (1D-ResCNN ID) RRMSE_t ↓ | Exp A2 (1D-ResUNet ID) RRMSE_t ↓ | **Exp A3 (Physics-ResUNet)** RRMSE_t ↓ | Exp A3 CC ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test-ID** | 0.8059 | 0.4699 | **0.4100** | 0.4625 | 0.8790 |
| **Test-OOD-60** ($f_c = 60$ Hz) | 1.2138 | 0.8373 | 0.6343 | **0.5244** | **0.8388** |
| **Test-OOD-45** ($f_c = 45$ Hz) | 1.3354 | 0.9369 | 0.7269 | **0.5388** | **0.8283** |
| **Test-Clean** (Over-denoising) | 0.4364 | 0.2427 | 0.2244 | **0.0325** | **0.9991** |

- 相較於標準基準 `Exp A1`，物理導向增強模型 `Exp A3` 在 `Test-OOD-60` 降低 **37.37%** $\text{RRMSE}_t$、在 `Test-OOD-45` 降低 **42.48%**（相較 `Exp A2` 亦降低 **25.87%**），並在 `Test-Clean` 減少 **86.62%** 過度去噪失真。

### 重現指令（SPEC A）

```bash
python3 proj_a_eegdenoise/src/colab_run_all_a.py --epochs 70
```

---

## SPEC B 核心結果（5-Fold Subject-Independent CV × 5 Seeds, Tesla T4）

- **報表與數據**：[proj_b_ahepa_ad_ftd/outputs/REPORT_SUMMARY.md](proj_b_ahepa_ad_ftd/outputs/REPORT_SUMMARY.md) ｜ [metrics_summary.csv](proj_b_ahepa_ad_ftd/outputs/metrics_summary.csv)

### Subject-Level Macro-F1（Mean ± Std, %）

| Task | 2026 Lit. Ref | Exp B0 (Leaky Segment Split) | Exp B1 (ShallowConvNet) | Exp B2a (LightGBM) | Exp B2b (Linear SVM) | Exp B3 (LightGAT <50K) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AD vs. CN** | 75.82 | 100.00 ± 0.00 (Seg Acc 98.59%) | 80.18 ± 1.22 | 78.68 ± 1.27 | **80.44 ± 2.70** | 75.29 ± 1.91 |
| **AD vs. FTD** | 70.65 | 100.00 ± 0.00 (Seg Acc 97.66%) | 52.04 ± 2.66 | 48.99 ± 4.35 | **56.19 ± 3.52** | 46.38 ± 4.92 |
| **FTD vs. CN** | 56.35 | 100.00 ± 0.00 (Seg Acc 97.77%) | 66.38 ± 2.85 | **77.69 ± 0.76** | 74.37 ± 2.61 | 70.64 ± 5.67 |
| **AD vs. FTD vs. CN** | N/A | 100.00 ± 0.00 (Seg Acc 97.02%) | 56.14 ± 1.43 | 55.79 ± 1.40 | **60.72 ± 3.21** | 55.13 ± 4.73 |

- **切片洩漏對照**：隨機切片切分（`Exp B0`）在四項任務皆產生 `97.02%–98.59%` 切片準確率（受試者投票 100%），嚴格受試者隔離後立即暴跌 `19.8–48.0 pp`。
- **FTD vs. CN 突破**：結合 5 頻段相對 PSD 與 wPLI/PLV 先驗降維特徵後，在 2026 最難瓶頸任務 `FTD vs. CN` 上 `LightGBM` 達 **77.69%**、`Linear SVM` 達 **74.37%**、`LightGAT`（僅 3,634 參數）達 **70.64%**（超越 2026 基準 `56.35%` 達 **+14.29 pp**）。

### 重現指令（SPEC B）

```bash
pip install mne lightgbm boto3 pymatreader
python3 proj_b_ahepa_ad_ftd/src/colab_run_all_b.py --n-seeds 5
```
