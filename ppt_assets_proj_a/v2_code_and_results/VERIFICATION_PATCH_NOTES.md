# V2 修正與二次驗證補測紀錄（供更新 PPT 與講者備忘錄）

本文件彙整針對二次審計所完成的 **3 張圖檔標籤修正** 與 **2 項直接實測補證數據**（全部在 held-out 測試集 `N = 452` 筆腦波上實測，無資料洩漏）。

---

## 一、已修正的 3 張圖檔（位於 `figures/`）

1. **`figures/00c_method_physics_resunet_pipeline.png`**（Slide 5）
   - 修正左側 3 個物理合成方塊的百分比基準為統一全樣本分母：
     - `1. Random Lowpass (67.5% total)`（標註 `75% of noisy | 4th-ord filtfilt, fc~U[35,110] Hz`）
     - `2. Local Burst Mask (30% total)`
     - `3. Clean Identity (10% total)`
2. **`figures/05_v2_crossfilter_waveforms_exp_rc1.png`**（Backup B1）
   - 修正右欄標題由 `-20 dB/dec` 改為精確零相位規格：**`Zero-Phase 1st-Order RC (fc = 40 Hz, -40 dB/dec)`**（因 `sosfiltfilt` 雙向濾波使振幅響應平方，實際滾降為 `-40 dB/dec`、40 Hz 處衰減 `-6 dB`），並同步更新主標為 `Zero-Phase 4th-Order Butterworth`。
3. **`figures/11_v2_quant_disentangle_crossfilter_bars.png`**（Slide 8）
   - 修正 X 軸標籤：`Pure LP45 (Butterworth 4th, filtfilt)` 與 **`RC 1st-Order (filtfilt) (fc=40 Hz, -40 dB/dec)`**。

---

## 二、新增 2 項直接實測補證（供講者備忘錄 & Q&A 應答）

### 補證 A：直接證實 `Exp A3-noLP` 的「誤觸乾淨開關、原樣放行」機制
在 `Test-Pure-LP45`（`N = 452` 筆 held-out 測試對 / 每 SNR）上，直接計算模型輸出 $\hat{x}$ 與**含噪輸入 $y$** 的相似度（`CC(pred, noisy_input)` 越接近 `1.0`、`RRMSE(pred, noisy_input)` 越接近 `0.0`，代表模型越接近原樣放行）：

| 名義 SNR | `Exp A2` `CC(pred, in)` / `RRMSE(pred, in)` | **`Exp A3-noLP` `CC(pred, in)` / `RRMSE(pred, in)`** | `Exp A3` `CC(pred, in)` / `RRMSE(pred, in)` |
| :---: | :---: | :---: | :---: |
| **`-7 dB`** | 0.886 / 0.554 | **0.964 / 0.188** | 0.787 / 0.758 |
| **`-5 dB`** | 0.888 / 0.537 | **0.972 / 0.160** | 0.807 / 0.723 |
| **`0 dB`** | 0.910 / 0.464 | **0.991 / 0.090** | 0.879 / 0.580 |
| **`+2 dB`** | 0.924 / 0.424 | **0.996 / 0.069** | 0.901 / 0.527 |

- **結論（不再只是推論，已獲直接實測證實）**：在 `Test-Pure-LP45` 上，`Exp A3-noLP` 輸出與含噪輸入的相關係數高達 **`0.964–0.996`**（誤差僅 `0.069–0.188`），直接證實它在失去高頻毛刺時確實誤觸 Clean 恆等開關、幾乎原封不動把低通肌電放行；而 `Exp A3` 則積極濾除雜訊（`CC(pred, in) = 0.787–0.901`）。

### 補證 B：實測各測試軌之 1–45 Hz 頻帶內（In-Band）實際 SNR
在名義全頻段 $\text{SNR} = -5.00\text{ dB}$ 下（`N = 452` 筆 held-out 測試對平均）：
- `Test-ID (Raw EMG)`：1–45 Hz 頻帶內實際 SNR = **`+1.32 dB`**
- `Test-Pure-LP45`：1–45 Hz 頻帶內實際 SNR = **`-4.43 dB`**（比 `Test-ID` 頻帶內難 **`5.75 dB`**）
- `Test-ExpDecay`：1–45 Hz 頻帶內實際 SNR = **`-3.18 dB`**（比 `Test-ID` 頻帶內難 **`4.50 dB`**）
- `Test-RC1st`：1–45 Hz 頻帶內實際 SNR = **`-3.73 dB`**（比 `Test-ID` 頻帶內難 **`5.05 dB`**）

- **Q&A 標準說法（精準且專業）**：
  1. 承認因 `mix_snr` 在低通後依總功率縮放，低通軌的 1–45 Hz 頻帶內干擾確實比 `Test-ID` 強約 `4.5–5.8 dB`；依 `模型 ÷ A0` 比值看，`A1` 有顯著退步（`0.583 -> 0.702`），`A2` 相對平穩（`0.509 -> 0.544`），而 `A3` 在同一測試軌上仍比同架構 `A2` 優異 `25.9%`（`A3/A0 = 0.404`）。
  2. 另一方面，即使在輕度 `+2 dB` 的 `Test-Pure-LP45`（頻帶內 SNR 仍落在訓練範圍內），`A3-noLP` 輸出仍與含噪輸入高達 `CC = 0.996`、誤差 `0.767`（vs. `A2 = 0.471`、`A3 = 0.482`），證實高頻線索缺失誘發誤放行的機制獨立存在。
  3. 下一步將補做「固定 1–45 Hz 頻帶內 SNR」與「擴充 `A2` 低 SNR 訓練範圍」的嚴格對照實驗，進一步拆解兩者貢獻。
