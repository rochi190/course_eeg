# Project A PPT 完整視覺素材清單（`ppt_assets_proj_a/`）

> **給下游 PPT 生成 AI（如 Claude）的使用指引**：
> 本資料夾已打包 Project A（EEG 肌電去噪）全部 **15 張 300 DPI 高解析度圖片** 與最新版簡報藍圖 [ppt_eeg.md](./ppt_eeg.md)（請先讀 [README_FOR_CLAUDE.md](./README_FOR_CLAUDE.md) 內的風格與禁止規則）。
> - **檔案編號 `00`、`00b`、`00c`**：**3 張學術研討會級概念與方法架構圖**（直接配給 Slide 2、4、5，徹底取代純文字與大表格！）。
> - **檔案編號 `01`–`07`**：**7 張人眼直觀波形對比 & 部署視覺圖**，直接從「原始含噪波形 → 各模型濾波後波形（含淺色殘差陰影）」秒懂效果。
> - **檔案編號 `08`–`11`**：V1 與 V2 原版**定量曲線／長條圖／頻譜圖**。
> - **檔案編號 `12`**：僅保留供內部參考，**請勿放進簡報投影片**。

---

## 全素材總覽與建議投影片配置表

| 檔名 | 類型 | 實驗階段 | 建議配置頁面 | 一句話看圖重點（人眼判讀焦點） |
| :--- | :---: | :---: | :---: | :--- |
| `figures/00_concept_wearable_ica_and_overlap.png` | **概念架構圖** | `[2.1 背景]` | **Slide 2（必放）** | 左：64 通道 ICA 可解 vs. 1–4 通道穿戴式欠定失效；右：EEG 生理頻段與寬頻 EMG 在 15–45 Hz 嚴重重疊、40 Hz 硬濾波切不開 |
| `figures/00b_protocol_and_ablation_design.png` | **實驗設計圖** | `[2.3 設計]` | **Slide 4（必放）** | 上：EEG/EMG 各自獨立 80/10/10 切分再混音（零洩漏）；下：A0 → A1 → A2 → A3-noLP → A3 五組遞進控制變因設計 |
| `figures/00c_method_physics_resunet_pipeline.png` | **方法架構圖** | `[2.4 解法]` | **Slide 5（必放）** | 左：物理導向合成三分支（隨機低通 75% ＋ 突發 30% ＋ 乾淨保真 10%）；右：4 層 1D-ResUNet（含 2× Bottleneck MHSA）＋ MSE + 0.1×MR-STFT 雙域 Loss |
| `figures/01_painpoint_shortcut_spectrum_waveform.png` | 波形＋頻譜 | `[2.2 痛點]` | **Slide 3（必放）** | 左欄比較乾淨 EEG、臉部高頻毛刺 EMG（`>80 Hz = 53.5%`）、與經 45 Hz 低通後「長得像腦波」的頭皮 EMG；右欄全域 PSD 標示紅區 `>80 Hz` 捷徑漏洞 |
| `figures/02_v1_waveform_stack_ood45_models.png` | **5 層波形瀑布圖** | `[V1 主實驗]` | **Slide 7（主推首選）** | 同一筆 `-5 dB` 突發低通樣本，由上而下排開 **Raw Input → A0 帶通 → A1 → A2 → A3** ＋ 右側局部放大；人眼直接看**彩色殘差陰影從超大縮到幾乎消失**（`1.778 -> 0.495`） |
| `figures/03_v1_clean_overdenoising_waveform.png` | **3 層乾淨保真波形** | `[V1 主實驗]` | **Slide 6 輔助 / 新增頁** | 100% 純乾淨腦波（`Test-Clean`）＋ 0.5 秒 Alpha 峰特寫：直接看見 **A1/A2 把正常高波峰削平（紅色／橘色失真陰影）**，而 **A3 藍線與黑線完全重合（`RRMSE=0.024, CC=0.9998`）** |
| `figures/04_v2_disentangle_waveform_a3_vs_nolp.png` | **雙欄對照波形** | `[V2 進階補證]` | **Slide 8（主推首選）** | 左欄 `Test-Pure-Burst`：`A3-noLP`（紫）與 `A3`（藍）都清得乾淨；右欄 `Test-Pure-LP45`：**抽掉低通的 `A3-noLP` 直接誤觸 Clean 開關把整段低通肌電原封不動放行（`1.628`）**，`A3` 穩健還原（`0.505`） |
| `figures/05_v2_crossfilter_waveforms_exp_rc1.png` | **雙欄跨濾波波形** | `[V2 進階補證]` | **Slide 8 延伸 / Backup** | 左欄未見平滑指數衰減 `ExpDecay`、右欄未見一階 `RC1st`（`-5 dB`），由上而下對比 Input → A2 → A3-noLP → A3，證明 A3 不是死背 4 階 Butterworth |
| `figures/06_v1_snr_progression_same_eeg.png` | **三級雜訊 Before/After** | `[V1 視覺展示]` | **Slide 1 封面示意 / Slide 6 延伸** | **同一段真實腦波**在 `-7 dB`（重度）、`-3 dB`（中度）、`+2 dB`（輕度）咬牙下的左欄輸入 vs. 右欄 `Exp A3` 還原（`CC = 0.89 / 0.92 / 0.95`），非專業聽眾最易秒懂 |
| `figures/07_v2_realtime_latency_benchmark.png` | 延遲視覺圖 | `[V2 進階補證]` | **Slide 8 右側 / Backup** | 左圖實測 CPU（`45.8 ms`）與 T4 GPU（`5.20 ms`）延遲與參數量；右圖對數尺度對比 `2,000 ms` 即時預算線（**43.7× / 384.6× 即時**） |
| `figures/08_v1_quant_snr_rrmse_curves.png` | 定量曲線圖 | `[V1 主實驗]` | **Slide 6（左圖）** | 原 V1 `fig1_snr_rrmse.png`：`Test-ID` vs. `Test-OOD (fc=45 Hz)` 全 10 級 SNR 之 RRMSE 與 CC 曲線 |
| `figures/09_v1_waveform_and_psd_single_case.png` | 疊加波形＋PSD | `[V1 主實驗]` | **Slide 7（備選或與 `02` 搭配）** | 原 V1 `fig2_waveform_psd.png`：與 `02` 為同一筆 `-5 dB` 樣本，上圖疊加波形、下圖 1–80 Hz Welch PSD 顯示 Alpha/Beta 頻譜保留 |
| `figures/10_v1_quant_ood_and_clean_bars.png` | 定量長條圖 | `[V1 主實驗]` | **Slide 6（右圖）** | 原 V1 `fig3_clean_preservation.png`：`Test-OOD-60 / OOD-45 / Test-Clean` 三組總體 RRMSE 長條對比 |
| `figures/11_v2_quant_disentangle_crossfilter_bars.png` | 定量長條圖 | `[V2 進階補證]` | **Slide 8（與 `04` 搭配）** | 原 V2 `fig4_disentangle_ablation.png`：4 條解耦與跨濾波器軌（`Pure-LP45 / Pure-Burst / ExpDecay / RC1st`）之總體 RRMSE 長條對比 |
| `figures/12_v2_backup_real_eeg_zeroshot_psd.png` | 臨床 Zero-Shot PSD | `[V2 備援 / Pilot]` | **Backup Slide B1（謹慎使用）** | 原 V2 `fig5_real_eeg_zeroshot.png`：⚠️ **注意左右圖 62 Hz 有原始醫院儀器窄頻尖峰**，建議定位為 Pilot Check 或僅於 Q&A 口頭說明 |

---

## 統一色彩規範（全 12 張圖一致，方便 PPT 配色呼應）

| 角色 | 顏色名稱 | Hex 色碼 | 說明 |
| :--- | :--- | :---: | :--- |
| Clean Ground Truth | 純黑 | `#111111` | 真實乾淨腦波基準線 |
| Raw Corrupted Input | 中灰 | `#7F8C8D` | 未處理之含噪輸入訊號 |
| `Exp A0`（1–40 Hz 帶通） | 深灰藍 | `#576574` | 傳統 DSP 下限 |
| `Exp A1`（1D-ResCNN ID） | 磚紅 | `#D62728` | 文獻標準基準（OOD 崩壞組） |
| `Exp A2`（1D-ResUNet ID） | 亮橘 | `#FF7F0E` | 架構升級但無物理增強對照組 |
| `Exp A3-noLP`（V2 消融） | 紫羅蘭 | `#8E44AD` | V2 新增：僅 Burst+Clean、移除低通 |
| **`Exp A3`（本提案方法）** | **科技藍** | **`#1F77B4`** | **完整物理導向增強＋時頻雙域重建** |

---

## 給 Claude 製作「重波形、少數字」版 PPT 的排版建議

1. **Slide 3（核心痛點）**：直接放 **`figures/01_painpoint_shortcut_spectrum_waveform.png`** 佔 65% 版面，文字只需 3 條短句解釋「高頻捷徑 53.5% vs 0.0%」。
2. **Slide 6（V1 定量＋防過度去噪）**：可用 **`figures/08_v1_quant_snr_rrmse_curves.png`**（或 **`figures/10_v1_quant_ood_and_clean_bars.png`**）搭配 **`figures/03_v1_clean_overdenoising_waveform.png`**，讓觀眾直接看到「過度去噪」是怎麼削平正常腦波峰值的！
3. **Slide 7（V1 原始波形 → 各模型濾波實戰）**：強烈推薦用 **`figures/02_v1_waveform_stack_ood45_models.png`** 作為主視覺（5 層排排站＋局部放大＋彩色殘差陰影，比單張疊圖更容易讓人眼一秒看懂！），若需補充頻譜再帶出 `figures/09_v1_waveform_and_psd_single_case.png` 下半部。
4. **Slide 8（V2 進階補證）**：主打 **`figures/04_v2_disentangle_waveform_a3_vs_nolp.png`**（波形鐵證：A3-noLP 把低通肌電直接放行）與 **`figures/11_v2_quant_disentangle_crossfilter_bars.png`**（全域統計），搭配 **`figures/07_v2_realtime_latency_benchmark.png`**（或濃縮為 2 顆 KPI 徽章：`CPU 45.8 ms (44×)` / `GPU 5.2 ms (385×)`）。
5. **延伸／備援頁**：把 **`figures/05_v2_crossfilter_waveforms_exp_rc1.png`**（跨濾波器波形）、**`figures/06_v1_snr_progression_same_eeg.png`**（同腦波三級雜訊 Before/After）與 **`figures/12_v2_backup_real_eeg_zeroshot_psd.png`**（真實臨床 Pilot Check）作為 Q&A 彈藥庫。
