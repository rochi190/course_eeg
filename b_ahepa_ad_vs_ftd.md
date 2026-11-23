# SPEC B：嚴格跨受試者驗證下之阿茲海默症（AD）與額顳葉失智症（FTD）腦功能連結圖神經網路分類

> **任務性質**：深度學習實驗企劃書與自動化執行規格（供 AI Coding Agent 直接實作）
> **目標產出**：可重現之程式碼、定量評估報表（`metrics_summary.csv`、`REPORT_SUMMARY.md`）與 3 張 300 DPI 簡報專用高清圖表。

---

## 1. 簡報核心脈絡（對應課堂提案 2.1–2.5）

* **2.1 背景**：阿茲海默症（AD）與額顳葉失智症（FTD）早期臨床症狀高度重疊，但後續藥物與治療策略截然不同。相較昂貴的 PET/MRI，19 通道臨床常規閉眼靜息態 EEG 低成本且易於普及，是極具潛力的早期鑑別篩檢工具。
* **2.2 當前限制與 2026 最新期刊基準痛點**：
  1. **資料洩漏導致虛高分數**：原資料集團隊於 2026 年 5 月在 *Cognitive Neurodynamics*（20:95）發表《*The AHEPA EEG benchmark*》，系統審查百餘篇論文並制定嚴格驗證分級，踢爆許多宣稱 95% 以上準確率的研究源於「切片層級隨機切分」，讓同一病患的相鄰片段同時出現在訓練與測試集。
  2. **嚴格隔離下 FTD 辨識瓶頸與大模型失效**：陽明交通大學團隊於 2026 年 8 月發表的 10 種深度學習 BCI 模型統一基準（bioRxiv `2026.08.12.743255`，嚴格 5-Fold 受試者隔離、重複 10 次）顯示，在 `ds004504`（ADFTD）上各二元任務的最佳受試者層級 F1 分數為：
     * **AD vs. HC**：75.82%（MSVTNet）
     * **FTD vs. AD**：70.65%（SzHNN）
     * **FTD vs. HC**：僅 **56.35%**（ShallowConvNet）
     且模型參數量越大反而越容易過擬合，大型 Transformer 跨任務平均表現墊底，反由極輕量的 ShallowConvNet 在 FTD vs. HC 居首。
* **2.3 資料集與模型**：
  * **資料集**：OpenNeuro `ds004504`（v1.0.9，CC0 公開授權，100% 醫院真實臨床錄製），共 **88 位受試者**（36 位 AD、23 位 FTD、29 位健康對照 CN），19 通道 10-20 系統閉眼靜息態 EEG，取樣率 500 Hz。
  * **比較基準與模型**：ShallowConvNet（端到端時域基準） vs. **傳統機器學習強基準（PSD+PLV 特徵 + LightGBM/Linear SVM）** vs. **輕量頻帶功能連結圖注意力網路（LightGAT）**。
* **2.4 預計解法**：在僅 88 位受試者（FTD 僅 23 人）的小樣本情境下，直接餵原始時域波形給高自由度網路極易背誦個體雜訊。本提案導入**神經科學先驗降維**：將 19 電極視為圖節點，萃取 5 頻段（Delta 0.5 至 4、Theta 4 至 8、Alpha 8 至 13、Beta 13 至 25、Gamma 25 至 45 Hz）之相對功率、頻譜熵與去偏相鎖指標（wPLI / 虛部相干性，用以抑制容積傳導假連結）建構功能連結圖，送入參數量小於 50K 的極輕量圖注意力網路（LightGAT），聚焦捕捉 AD（頂葉／後顳葉退化）與 FTD（額葉／前顳葉連結異常）的拓樸差異，並以受試者層級機率平均投票輸出最終診斷。
* **2.5 目前進度（本次自動化執行目標）**：完成 `ds004504` 載入與前處理、實證重現「隨機切片洩漏 vs. 嚴格受試者隔離」的效能崩塌現象、完成 ShallowConvNet 與 LightGAT 之跨受試者對照實驗，並產出腦地形圖與連結差異熱圖。

---

## 2. 專案目錄架構

~~~text
proj_b_ahepa_ad_ftd/
├── data/
│   ├── ds004504/             # OpenNeuro BIDS participants.tsv 與 derivatives/
│   └── cache/                # 快取切窗後之 Raw Epochs 與 PSD/wPLI/PLV 圖特徵 (.npz/.pt)
├── src/
│   ├── 01_download_bids.py   # 透過 AWS S3 (免認證) 或 openneuro-py 下載 derivatives
│   ├── 02_preprocess_feat.py # MNE 讀取 .set、切 4 秒窗、抽 5 頻段 PSD 與 wPLI/PLV 矩陣
│   ├── 03_models.py          # ShallowConvNet (Raw EEG) 與 LightGAT (Graph)
│   ├── 04_run_benchmarks.py  # 執行洩漏對照 (B0) 與嚴格 5-Fold 跨受試者實驗 (B1-B3)
│   └── 05_plot_figures.py    # 產出 300 DPI 簡報專用圖表
└── outputs/
    ├── metrics_summary.csv
    ├── REPORT_SUMMARY.md
    └── figures/
        ├── fig1_leakage_gap.png
        ├── fig2_topomap_plv.png
        └── fig3_f1_comparison.png
~~~

---

## 3. 資料取得與前處理規範

### 3.1 資料下載（優先採 AWS S3 公開桶直接拉取 `derivatives`，最省時且免 git-annex）
~~~bash
# 方法一（推薦）：使用 awscli 免認證下載
aws s3 sync --no-sign-request s3://openneuro.org/ds004504/derivatives ./data/ds004504/derivatives
aws s3 cp --no-sign-request s3://openneuro.org/ds004504/participants.tsv ./data/ds004504/participants.tsv

# 方法二（備案）：使用 Python openneuro-py 套件
# pip install openneuro-py
# openneuro-py download --dataset=ds004504 --include=participants.tsv --include=derivatives
~~~

### 3.2 標籤解析與資料切窗
* 讀取 `participants.tsv`（Tab 分隔）：`participant_id`（如 `sub-001`）對應 `Group` 欄位：
  * `A`：阿茲海默症（AD，36 人）
  * `F`：額顳葉失智症（FTD，23 人）
  * `C`：健康對照組（CN，29 人）
* 使用 `mne.io.read_raw_eeglab(set_file, preload=True)` 讀取 `derivatives/sub-XXX/eeg/*_eeg.set`。
* 檢查並統一 19 個標準 10-20 通道順序（Fp1, Fp2, F7, F3, Fz, F4, F8, T3/T7, C3, Cz, C4, T4/T8, T5/P7, P3, Pz, P4, T6/P8, O1, O2），必要時統一命名（如將舊命名 T3/T4/T5/T6 對應至標準 montage 供 MNE 繪製 Topomap）。
* 降採樣至 **250 Hz**（濾波帶寬 0.5 至 45 Hz），切成**不重疊之 4 秒窗口**（每窗形狀為 $19 \times 1000$）。剔除明顯異常振幅窗口（例如峰對峰值大於 $500\,\mu\text{V}$）。

### 3.3 特徵與腦圖建構（每個 4 秒窗口獨立計算，嚴禁跨窗口或跨受試者正規化）
* **節點特徵（Node Features，$19 \times F$）**：
  * 5 頻段（Delta, Theta, Alpha, Beta, Gamma）之**相對功率譜密度（Relative PSD）**與對數絕對功率。
  * 額外輔助特徵：頻譜熵（Spectral Entropy）、Theta/Alpha 功率比、Peak Alpha Frequency（PAF）。
* **邊權重矩陣（Edge / Adjacency，$19 \times 19$）**：
  * 針對 5 個頻段分別計算 **wPLI（Weighted Phase Lag Index，抑制容積傳導假性零延遲同步）** 與 **PLV（Phase Locking Value）**。
  * 每個頻段矩陣去除對角線並採 Top-K（如每節點保留前 30% 至 40% 最強連結）或軟權重保留，構成多通道邊特徵或頻段融合圖。

---

## 4. 嚴格防洩漏驗證與實驗矩陣

### 4.1 驗證協議（對標 2026 基準）
* 除刻意設計的洩漏對照組（Exp B0）外，所有實驗必須使用 `sklearn.model_selection.StratifiedGroupKFold(n_splits=5)`，嚴格以 `participant_id` 作為分組鍵。
* **強制防洩漏斷言**：每一折（Fold）訓練前必須執行：
  `assert set(train_subject_ids).isdisjoint(set(val_subject_ids))`
* **重複多次取穩定估計**：至少使用 5 個不同隨機種子（若計算時間充裕，拉滿 10 個種子以完全對標 2026 bioRxiv 基準），報告 **Mean $\pm$ Std**。
* **雙層級評分指標**：
  * **Segment-level**：所有測試窗口之 Macro-F1、Accuracy、AUROC。
  * **Subject-level（核心指標）**：將同一位受試者所有測試窗口的預測機率取平均（Soft Voting）後決定受試者類別，計算 **Subject-level Macro-F1**、Accuracy、AUROC 與 Sensitivity/Specificity。
* **所有標準化（StandardScaler）必須嚴格僅在該 Fold 的 Train Set 上 `fit`**，再 `transform` Val/Test Set。

### 4.2 評估任務（共 4 項任務）
1. **Task 1**：`AD vs. CN`（A vs. C）
2. **Task 2**：`AD vs. FTD`（A vs. F）
3. **Task 3**：`FTD vs. CN`（F vs. C，2026 基準最難瓶頸任務）
4. **Task 4（延伸）**：三分類 `AD vs. FTD vs. CN`（A vs. F vs. C）

### 4.3 實驗組別

| 實驗編號 | 切分方式 | 模型與輸入 | 預期驗證目的 |
| :--- | :--- | :--- | :--- |
| **Exp B0** | **切片隨機切分（故意洩漏）** | ShallowConvNet（Raw EEG） | 用數據實證踢爆「切片洩漏可輕易刷到 90% 至 95% 以上」的假象 |
| **Exp B1** | **嚴格 5-Fold 受試者隔離** | **ShallowConvNet**（Raw EEG） | 重現 2026 bioRxiv 基準（驗證 FTD vs. CN 落在約 55% 至 60% 瓶頸） |
| **Exp B2** | **嚴格 5-Fold 受試者隔離** | **LightGBM / Linear SVM**（PSD + wPLI 上三角向量） | 檢驗傳統強基準特徵表現，確保深度圖網路非無效堆疊 |
| **Exp B3** | **嚴格 5-Fold 受試者隔離** | **LightGAT（本提案）**（19 節點特徵 + 頻段 wPLI/PLV 圖） | 驗證極輕量圖網路（參數量 $< 50\text{K}$）對小樣本與 FTD 辨識的增益 |

### 4.4 模型與防過擬合設計
* **ShallowConvNet**：遵循經典 Schirrmeister et al. 架構（時域卷積 -> 空間卷積 -> 平方非線性 -> 平均池化 -> 對數非線性 -> 分類頭）。
* **LightGAT**：可採 PyTorch 原生實作（免去安裝 `torch_geometric` 之相依地雷，直接對 $19 \times 19$ 稠密鄰接矩陣做 Masked Multi-Head Graph Attention）：
  * 2 層 Graph Attention Layer（Hidden dim = 32, Heads = 4）搭配 LayerNorm、ELU 與 **強正則化（Dropout = 0.5、Weight Decay = $10^{-2}$、Label Smoothing = 0.05）**。
  * Global Mean + Max Pooling 聚合 19 節點表徵後接單層線性分類器。
  * 類別不平衡處理：於 CrossEntropyLoss 加入 `class_weight`（特別針對 `AD vs. FTD` 中 36 vs. 23 的比例不均）。
* **硬體調度備註**：88 位受試者特徵萃取後體積非常小，LightGBM 與 LightGAT 在 CPU 上亦可在數分鐘內完成；若 ShallowConvNet 在 CPU 訓練偏慢，可先將快取特徵與切窗打包並產生 `colab_run_b1.py` 移至 Colab A100 跑完。

---

## 5. 必出 3 張簡報高清圖（300 DPI，字體大小 $\ge 12$ pt，英文標籤）

1. `fig1_leakage_gap.png`：分組長條圖（含誤差線），並排展示 `AD vs. CN`、`AD vs. FTD`、`FTD vs. CN` 在 **Exp B0（隨機切片洩漏）** 與 **Exp B1（嚴格跨受試者隔離）** 下的 Accuracy / F1 斷崖式落差。
2. `fig2_topomap_plv.png`：上下複合圖。上排使用 `mne.viz.plot_topomap` 繪製 AD、FTD、CN 三組平均之 **Theta（4 至 8 Hz）** 與 **Alpha（8 至 13 Hz）** 相對功率腦地形圖（同頻段共用統一 Colorbar 刻度）；下排繪製 **AD 與 FTD 之 Alpha/Theta 頻段 wPLI 平均差異熱圖（$19 \times 19$）**，直觀標示額顳與頂葉連結差異。
3. `fig3_f1_comparison.png`：三大二元任務（`AD vs. CN`、`AD vs. FTD`、`FTD vs. CN`）之 Subject-level Macro-F1 對比圖，包含：
   * **2026 文獻基準線**（以水平虛線標註：75.82% / 70.65% / 56.35%）。
   * **Exp B1（ShallowConvNet）**、**Exp B2（Feature + ML）**、**Exp B3（LightGAT）** 之實測分數與標準差誤差線。

---

## 6. Agent 執行與驗收清單

1. **階段一（Smoke Test）**：先下載 `participants.tsv` 與前 10 位受試者 `.set` 檔跑通 MNE 載入、特徵萃取與圖網路前向傳播，確認防洩漏 `assert` 與 MNE Topomap 繪圖（含通道名稱對應）無誤。
2. **階段二（全量實驗）**：下載全部 88 位受試者資料，快取切窗與圖特徵至 `data/cache/`，完整執行 Exp B0 至 B3。
3. **階段三（自動彙整）**：輸出 `outputs/metrics_summary.csv` 與 `outputs/REPORT_SUMMARY.md`，自動整理 4 個任務的 Segment-level 與 Subject-level 表格，明確標註相對於 2026 基準線的勝負差距，並確認 3 張圖表皆已正確生成。