# SPEC A：克服容積傳導頻譜偏移之單通道 EEG 肌電偽影（EMG）物理導向去噪

> **任務性質**：深度學習實驗企劃書與自動化執行規格（供 AI Coding Agent 直接實作）
> **目標產出**：可重現之程式碼、定量評估報表（`metrics_summary.csv`、`REPORT_SUMMARY.md`）與 3 張 300 DPI 簡報專用高清圖表。

---

## 1. 簡報核心脈絡（對應課堂提案 2.1–2.5）

* **2.1 背景**：可攜式少通道腦波儀日益普及，但使用者咬牙、皺眉或轉頭產生的臉部肌電（EMG）頻帶極寬（20 至 200 Hz 以上），與腦波 Beta（13 至 30 Hz）及 Gamma（30 Hz 以上）頻段嚴重重疊。單通道情境無法使用傳統 ICA（獨立成分分析），傳統帶通濾波則會連真實腦波訊號一併抹除。
* **2.2 當前限制與物理漏洞**：現行深度學習去噪基準 **EEGdenoiseNet** 採用純量線性疊加模型：
  $$y(t) = x(t) + \lambda \cdot n(t)$$
  直接將「臉部雙極差分 sEMG」乘上常數 $\lambda$ 加到「頭皮單極 EEG」。此假設存在兩大生物物理破綻：
  1. **忽略容積傳導的空間低通效應**：雙極差分 sEMG 本質上偏向空間高通濾波器（高頻銳利），但真實肌肉電位經由頭皮與組織容積傳導至遠端 EEG 電極（如額葉、頂葉）時，空間電位梯度會被平滑化，等效為隨距離變化的低通捲積濾波 $y(t) = x(t) + (h_{\text{spatial}} * n)(t)$。
  2. **忽略肌電的局部突發性**：標準合成將整段 2 秒完全疊加雜訊，未考慮真實穿戴下短暫咬牙（0.3 至 1.5 秒）或完全乾淨片段，容易導致模型過度去噪（Over-denoising）。
  這使標準訓練模型過度擬合「臉部雙極 sEMG 的特定高頻紋理」，面對跨頻譜分佈（Out-of-Distribution, OOD）的頭皮肌電時效能顯著下降。
* **2.3 資料集與模型**：
  * **資料集**：EEGdenoiseNet（4,514 段乾淨 EEG、5,598 段真實臉部 EMG，統一 512 Hz、每段 2 秒共 1,024 個取樣點）。
  * **比較基準與模型**：傳統帶通濾波（1 至 40 Hz） vs. **1D-ResCNN（標準基準）** vs. **1D-ResUNet 搭配多解析度 STFT 頻域損失函數**。
* **2.4 預計解法**：
  1. **物理導向資料增強（Physics-informed Augmentation）**：訓練階段合成受污染訊號前，對純 EMG 隨機施加 4 階 Butterworth 低通濾波（隨機截止頻率 $f_c \in [35, 110]$ Hz，模擬不同電極距離的組織高頻衰減）；同時混入 30% **局部突發遮罩**（僅隨機 0.3 至 1.5 秒區間含 EMG）與 10% **純乾淨 EEG 樣本**（強迫模型學會無雜訊時原封不動保留腦波）。
  2. **時頻雙域約束損失函數**：
     $$\mathcal{L} = \mathcal{L}_{\text{MSE}}(x, \hat{x}) + \alpha \cdot \mathcal{L}_{\text{MR-STFT}}(x, \hat{x})$$
     透過多解析度短時傅立葉頻譜誤差，防止模型在低 SNR 下把 Alpha/Beta 腦波一起抹平。
* **2.5 目前進度（本次自動化執行目標）**：完成資料管線與防洩漏切分、實作標準合成 vs. 物理 OOD 壓力測試的 $2 \times 2$ 對照實驗，產出定量數據與簡報視覺化圖表。

---

## 2. 專案目錄架構

~~~text
proj_a_eegdenoise/
├── data/
│   ├── raw/                  # 存放 EEG_all_epochs*.npy 與 EMG_all_epochs*.npy
│   └── processed/            # 切分後之 train/val/test 索引與打包檔案（供 Colab A100 銜接用）
├── src/
│   ├── 01_download_data.py   # 自動拉取資料並強制校驗取樣率=512Hz (T=1024)
│   ├── 02_synthesis.py       # 標準線性疊加 + 物理低通 OOD / 局部突發遮罩合成模組
│   ├── 03_models.py          # 1D-ResCNN、1D-ResUNet 與 Multi-Resolution STFT Loss
│   ├── 04_train_eval.py      # 執行實驗矩陣、儲存 Checkpoint 與評估指標
│   └── 05_plot_figures.py    # 產出 300 DPI 簡報專用圖表
└── outputs/
    ├── checkpoints/
    ├── metrics_summary.csv
    ├── REPORT_SUMMARY.md
    └── figures/
        ├── fig1_snr_rrmse.png
        ├── fig2_waveform_psd.png
        └── fig3_clean_preservation.png
~~~

---

## 3. 資料取得與嚴格防呆規範

### 3.1 資料來源
優先由以下公開來源擇一下載 `.npy` 檔至 `data/raw/`：
1. 官方 GitHub Repo：`https://github.com/ncclabsustech/EEGdenoiseNet`（`data/` 目錄下之 `EEG_all_epochs_512hz.npy` 與 `EMG_all_epochs.npy`；若主庫為 Git LFS 指標檔，請改抓實際檔案或鏡像）。
2. HuggingFace 公開鏡像（如 `woldier/eeg_denoise_dataset` 或同源鏡像）。

### 3.2 關鍵防呆規則（必須寫入程式碼 `assert`）
* **取樣率與長度校驗**：部分鏡像中的 `EEG_all_epochs.npy` 為 256 Hz（形狀為 `(N, 512)`），而 `EMG_all_epochs.npy` 為 512 Hz（形狀為 `(M, 1024)`）。讀取後**必須檢查 `.shape[1]`**：
  * 若 EEG 的 `shape[1] == 512`，使用 `scipy.signal.resample(eeg, 1024, axis=1)` 重取樣至 512 Hz（`T = 1024`）。
  * 強制驗證 `assert eeg.shape[1] == 1024 and emg.shape[1] == 1024`。
* **防資料洩漏切分**：
  * 必須將乾淨 EEG（4,514 段）與純 EMG（5,598 段）**各自獨立**依 80% / 10% / 10% 切分為 Train / Val / Test（固定隨機種子 `seed=42`）。
  * **嚴禁先混音再切分**。必須加入斷言：`assert set(train_eeg_idx).isdisjoint(test_eeg_idx)` 與 `assert set(train_emg_idx).isdisjoint(test_emg_idx)`。

### 3.3 訊噪比（SNR）合成公式
給定單段乾淨 EEG $x$ 與單段 EMG $n$，依據均方根（RMS）比例控制目標 $\text{SNR}$（單位 dB，範圍 $-7$ dB 至 $+2$ dB，間隔 $1$ dB，共 10 個等級）：

$$\text{RMS}(s) = \sqrt{\frac{1}{T}\sum_{t=1}^{T} s[t]^2}, \quad \lambda = \frac{\text{RMS}(x)}{\text{RMS}(n) \cdot 10^{\text{SNR}/20}}$$

標準線性疊加訊號為 $y = x + \lambda \cdot n$。進入網路前將單筆輸入 $y$ 除以其標準差 $\sigma_y$ 正規化，對應之目標 $x$ 同步除以相同 $\sigma_y$（維持正確振幅比例）。

---

## 4. 實驗設計與對照矩陣

### 4.1 測試集定義（雙軌壓力測試）
1. **Test-ID（官方標準測試集）**：嚴格遵守 $y = x + \lambda \cdot n$，在 $\text{SNR} \in \{-7, -6, \dots, +2\}$ dB 下各生成固定配對測試樣本。
2. **Test-OOD（物理容積傳導偏移壓力測試集）**：在計算 RMS 與 $\lambda$ **之前**，對測試用 EMG 先套用 4 階零相位 Butterworth 低通濾波（`sosfiltfilt`），模擬遠端頭皮量測的高頻衰減（設兩組子情境：中度衰減 $f_c = 60$ Hz、重度衰減 $f_c = 45$ Hz），並於其中 50% 樣本施加隨機位置、長度 0.5 至 1.2 秒之平滑漢寧窗（Tukey window）局部突發遮罩。
3. **Test-Clean（純乾淨保真測試集）**：輸入 100% 無雜訊之測試集乾淨 EEG（$\lambda = 0$），檢驗模型是否出現過度去噪。

### 4.2 實驗組別

| 實驗編號 | 模型架構 | 訓練資料合成策略 | 損失函數 | 預期驗證目的 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp A0** | 傳統濾波（1 至 40 Hz 帶通） | 無需訓練 | N/A | 傳統 DSP 下限基準 |
| **Exp A1** | **1D-ResCNN** | 標準線性疊加（ID） | 純時域 MSE | 代表現有文獻標準做法，預期在 Test-OOD 與 Test-Clean 退化 |
| **Exp A2** | **1D-ResUNet** | 標準線性疊加（ID） | MSE + MR-STFT ($\alpha = 0.1$) | 拆解架構與頻域 Loss 本身的貢獻（消融對照） |
| **Exp A3** | **1D-ResUNet（本提案）** | **物理導向增強**（隨機低通 $f_c \in [35, 110]$ Hz + 30% 突發遮罩 + 10% 純乾淨） | MSE + MR-STFT ($\alpha = 0.1$) | 目標：Test-ID 持平或相當，**Test-OOD 與 Test-Clean 顯著擊敗 A1 與 A2** |

### 4.3 模型與訓練建議
* **1D-ResCNN**：5 層殘差卷積塊（Kernel sizes 採多尺度如 3、5、7，通道數 64 至 128），保持輸入輸出長度 1,024。
* **1D-ResUNet**：4 層 Encoder-Decoder 搭配 Skip Connections 與殘差塊（基礎通道數 32 -> 64 -> 128 -> 256），瓶頸層可選擇加入輕量 Multi-Head Self-Attention 或 Dilated Conv 擴增感受野。
* **MR-STFT Loss**：使用 `torch.stft` 在 3 組窗口尺寸（`n_fft` = 64, 128, 256）計算對數幅度譜 L1 誤差＋線性幅度譜收斂誤差。
* **訓練細節**：AdamW（`lr=1e-3`、`weight_decay=1e-4`）、CosineAnnealingLR、Batch Size 128、訓練 60 至 80 Epochs（開啟 Early Stopping）。
* **硬體調度備註**：若執行主機無獨立 GPU 或 VRAM 不足，請先以 CPU 跑完 `01_download_data.py`、`02_synthesis.py` 及 2 Epochs Smoke Test，並額外輸出單一自包含腳本 `colab_run_all_a.py` 以便直接移至 Colab A100 滿速訓練。

---

## 5. 評估指標與必出圖表

### 5.1 定量評估指標
對所有測試情境分 SNR 等級與整體平均計算以下 4 項指標，輸出至 `outputs/metrics_summary.csv`：
1. **時域相對均方根誤差（$\text{RRMSE}_t$）**：
   $$\text{RRMSE}_t = \frac{\text{RMS}(\hat{x} - x)}{\text{RMS}(x)}$$
2. **頻域相對均方根誤差（$\text{RRMSE}_s$）**：以 Welch 法計算 $\hat{x}$ 與 $x$ 之功率譜密度（PSD）後計算其相對 RMS 誤差。
3. **Pearson 相關係數（$\text{CC}$）**：$\hat{x}$ 與 $x$ 之線性相關度（越接近 1 越好）。
4. **頻段功率保留誤差**：計算 Alpha（8 至 13 Hz）與 Beta（13 至 30 Hz）相對功率的絕對誤差（MAE）。

### 5.2 必出 3 張簡報高清圖（300 DPI，字體大小 $\ge 12$ pt，英文標籤避免缺字）
1. `fig1_snr_rrmse.png`：左右雙子圖。左圖為 **Test-ID**、右圖為 **Test-OOD ($f_c = 45$ Hz)**；橫軸為 SNR（$-7$ 至 $+2$ dB），縱軸為 $\text{RRMSE}_t$（並可加副圖畫 $\text{CC}$），清楚標出 Exp A0 / A1 / A2 / A3 曲線差異。
2. `fig2_waveform_psd.png`：挑選一筆 $\text{SNR} = -5$ dB 之 Test-OOD 樣本。上子圖繪製時域波形（Noisy / Ground Truth / Exp A1 / Exp A3）；下子圖繪製對應的 Welch PSD 對數功率譜（1 至 80 Hz），標示 Alpha（8 至 13 Hz）與 Beta（13 至 30 Hz）陰影區。
3. `fig3_clean_preservation.png`：分組長條圖，對比 Exp A1 / A2 / A3 在 **Test-OOD 整體平均 $\text{RRMSE}_t$** 與 **Test-Clean 過度去噪失真率（$\text{RRMSE}_t$）** 的表現。

---

## 6. Agent 執行與驗收清單

1. **階段一（Smoke Test）**：先確認資料下載與形狀校驗通過、防洩漏 `assert` 通過，並以 2 Epochs 跑通全流程與產圖腳本（確保無 runtime error）。
2. **階段二（完整實驗）**：固定隨機種子，執行 Exp A0 至 A3 完整訓練與測試。
3. **階段三（自動彙整）**：自動產出 `outputs/REPORT_SUMMARY.md`，內含完整 Markdown 數據對比表、關鍵發現摘要（特別標註 A3 在 Test-OOD 相對 A1 的 $\text{RRMSE}_t$ 改善百分比），以及 3 張圖片的路徑確認。