# Project A 完整研究資料包（供 Claude 二次審計驗證 ＆ 製作 PPT）

本壓縮包包含 **Project A（克服容積傳導頻譜偏移之單通道 EEG 肌電去噪）** 的 **V1 主實驗** 與 **V2 進階補證測試** 全套原始程式碼、規格書、訓練日誌、完整 CSV 原始數據、12 張 300 DPI 圖表以及最新版簡報藍圖。

> **為何沒放 `.npy` / `.pt`？** 原始資料集（`EEG/EMG_all_epochs_512hz.npy`，共 80 MB）與 4 份 PyTorch 權重檔（共 49 MB）因體積過大未打包；其餘 **100% 原始程式碼、規格、訓練 log、明細 CSV 與 300 DPI 圖表** 皆完整收錄於本包，足以讓你獨立交叉核對每一筆數字、驗證無資料洩漏與審核所有結論。

---

## 一、資料夾結構導覽

```text
ppt_assets_proj_a/
├── README_FOR_CLAUDE.md              <-- 你正在讀的總入口與二次審計清單
├── ASSET_MANIFEST.md                 <-- 12 張圖表（01-07 波形視覺圖 + 08-12 統計圖）索引與配色表
├── ppt_eeg.md                        <-- 完整報告、Q&A 攻防備忘與 9 頁主簡報 + 3 頁 Backup 藍圖
├── figures/                          <-- 全部 12 張 300 DPI 高解析度圖片（直接插入 PPT）
├── specs_and_docs/
│   ├── README.md                     <-- 專案總覽
│   ├── a_eegdenoisenet.md            <-- V1 原始規格書
│   ├── spec_a_v2.md                  <-- V2 進階補證規格書（解釋為何多測 V2）
│   └── EXPLANATION_A_V2.md           <-- V2 技術解說報告
├── v1_code_and_results/              <-- [V1 主實驗] 完整程式碼、Colab 訓練日誌與 CSV 數據
│   ├── src/01_download_data.py       <-- 80/10/10 獨立防洩漏切分（EEG 與 EMG 各自切完再混音）
│   ├── src/02_synthesis.py           <-- 標準合成 vs. 物理導向合成（Lowpass 35-110Hz + Burst 30% + Clean 10%）
│   ├── src/03_models.py              <-- Exp A0 (帶通) / A1 (1D-ResCNN) / A2 & A3 (1D-ResUNet + MR-STFT Loss)
│   ├── src/04_train_eval.py          <-- 訓練、Early Stopping、全 10 級 SNR 評估管線
│   ├── src/05_plot_figures.py        <-- V1 fig1–fig3 繪圖腳本
│   ├── run_a.log                     <-- V1 實跑 Colab 完整訓練／驗證日誌
│   ├── metrics_summary.csv           <-- V1 全 10 級 SNR 原始數值（137 列）
│   └── REPORT_SUMMARY.md             <-- V1 摘要表
└── v2_code_and_results/              <-- [V2 進階補證] 完整程式碼、Colab 訓練日誌與 CSV 數據
    ├── src/06_ablation_stress_eval.py     <-- A3-noLP 訓練 + 8 條壓力軌 + ds004504 Zero-Shot + 延遲測試
    ├── src/07_plot_v2_figures.py          <-- V2 fig4–fig5 繪圖腳本
    ├── src/08_generate_ppt_visual_assets.py <-- 生成 figures/01–07 人眼直觀波形圖的完整腳本（嚴格只取 held-out test split）
    ├── run_a_v2.log                  <-- V2 實跑 Colab (T4 GPU) 完整訓練／驗證日誌
    ├── ablation_metrics_v2.csv       <-- V2 全 8 軌 × 5 模型 × 5 SNR 原始數值（215 列）
    ├── zeroshot_metrics_v2.csv       <-- V2 真實臨床 EEG Zero-Shot 數值
    ├── latency_benchmark_v2.csv      <-- V2 單窗推論延遲（CPU 4-thread / T4 GPU）與參數量
    └── REPORT_SUMMARY_V2.md          <-- V2 摘要表
```

---

## 二、請 Claude 優先執行的「二次驗證審計清單」

製作 PPT 前，歡迎你直接核對程式碼與原始 CSV，確認以下重點無誇大或錯誤：

1. **防資料洩漏查核**：檢查 `v1_code_and_results/src/01_download_data.py`（EEG 4,514 段與 EMG 5,598 段各自獨立 `default_rng(42)` 切分 80/10/10，且有 `assert isdisjoint`）與 `v2_code_and_results/src/08_generate_ppt_visual_assets.py`（所有波形圖均只從 `test_eeg` / `test_emg` 抽取）。
2. **數字溯源查核**：核對 `ppt_eeg.md` 引用的每個指標是否與 `v1_code_and_results/metrics_summary.csv`、`v2_code_and_results/ablation_metrics_v2.csv`、`zeroshot_metrics_v2.csv`、`latency_benchmark_v2.csv` 完全一致。
3. **V1 vs. V2 評估口徑差異核對**：
   - V1 使用 **10 級 SNR（`-7..+2 dB`）** 與 `(0.5, 1.2) s` Tukey burst；
   - V2 使用 **5 級代表 SNR（`[-7, -5, -3, 0, +2] dB`）** 與 `(0.3, 1.5) s` raised-cosine burst；
   - 因此同名軌道的 Overall RRMSE 會略有差異（例如 `Exp A3` 在 `Test-OOD-45`：V1 為 `0.5388`、V2 為 `0.5528`）。請在簡報清楚標示 `[V1 主實驗]` 與 `[V2 進階補證]`，避免混表。
4. **已知限制與誠實揭露事項（請勿過度宣稱）**：
   - **`figures/12_v2_backup_real_eeg_zeroshot_psd.png` 的 62 Hz 尖峰**：`06_ablation_stress_eval.py` 僅對 `ds004504` 做了 50 Hz notch，原始醫院錄影存在 ~62 Hz 窄頻儀器干擾；且真實高肌電段無 Ground Truth，因此該測試定位為 **Backup Pilot Check**，切勿放進主線強推。
   - **跨濾波器高 SNR 細節**：在 `Test-ExpDecay` 與 `Test-RC1st` 上，`Exp A3` 整體與中低 SNR（`-7` 到 `-3 dB`）勝出，但於 `+2 dB` 高 SNR 下 `Exp A2` 略優（因平緩滾降濾波器仍殘留高頻線索）。
   - **漂移邊界強度**：`Test-Composite-Drift` 的漂移振幅為 `0.35 × RMS(clean_EEG)`，屬輕度漂移（`A3`: `0.6261 -> 0.6380`），大振幅電極滑動仍須搭配前端高通濾波。
   - **訓練精度微差**：V1 模型為 FP32 訓練，V2 新訓的 `A3-noLP` 使用 AMP `float16`（loss 轉 `fp32` 計算）。

---

## 三、給 Claude 的 PPT 製作核心目標

- **目標聽眾與時間**：課程提案報告 **8–10 分鐘含 Q&A**（主簡報 9 頁口說約 **6 分 35 秒** ＋ Backup 彈藥庫 3 頁）。
- **視覺核心要求（「多看波形、少堆數字」）**：
  - 優先使用 `figures/01`–`07` 的**原始波形 vs. 模型濾波後波形對比圖**（搭配彩色殘差陰影與局部放大），讓人眼在 2 秒內直接看出誰清乾淨、誰過度去噪、誰誤放行低通肌電！
  - 清楚標註 **`[V1 主實驗]`** 與 **`[V2 進階補證]`** 徽章，並講清楚「為什麼要多做 V2（堵死因子混淆、濾波器過擬合、無法即時部署三大質疑）」。
  - 逐頁結構與講稿請參考同層級的 `ppt_eeg.md` 與 `ASSET_MANIFEST.md`，你也可在二次審計後主動提出更精煉的視覺排版優化！
