# Project A 完整研究簡報素材包（給 Claude 製作研討會級提案 PPT）

本資料包包含 **Project A（克服容積傳導頻譜偏移之單通道 EEG 肌電去噪）** 的全部 **15 張 300 DPI 圖表**（含 3 張概念／方法架構圖 ＋ 7 張人眼直觀波形圖 ＋ 5 張定量圖）、完整原始程式碼、訓練日誌與 CSV 數據。

---

## 一、簡報風格與硬性規則（請務必嚴格遵守！）

### 1. 目標定位：嚴謹學術研討會口頭報告風（IEEE TBME / MICCAI / NeurIPS Oral Style）
- **語氣要求**：**自信、精準、有說服力的科學敘事（Storytelling）**。完整保留實驗設計與核心數據，講清楚「為何現行方法有物理盲點 → 我們如何對症下藥 → 波形與消融如何證明有效」。
- **結構與頁數**：嚴格維持 **9 頁主簡報（口說 6.5 分鐘）＋ 2 頁視覺 Backup（共 11 頁）**，請勿任意增生頁數。
- **視覺為主、文字為輔**：**Slide 2 至 Slide 8 每一頁都已配備專屬 300 DPI 架構圖或波形圖**，圖片佔版面 58%–75%，每頁文字精簡至 **2–3 個短重點**（字體 $\ge 16\text{–}18\text{ pt}$），**切忌把 Slide 2/4/5 做成純文字或大表格**。
- **標題與資訊整合**：請勿在封面自行編造學校或系所名稱（保留 `[報告人姓名] · [報告日期]` 即可）；也不要自己手繪假的示意波形，全部使用 `figures/` 內的真實資料圖。

### 2. ⛔ 絕對禁止事項（Hard Prohibitions — 切勿犯以下錯誤）
本包附上 `v1_code_and_results/` 與 `v2_code_and_results/` 是為了讓你 **「靜默交叉核對數字與方法正確性（確保零誇大、零幻覺）」**，**絕對不是要你把審計報告印到投影片上！**

1. **⛔ 禁止建立「審計檢討／自我拆台／限制盤點」頁面**（例如禁止製作任何類似「被追問時這些限制我們主動講」的 Backup 頁）。
2. **⛔ 禁止在投影片頁底或正文塞滿防禦性免責小字**：
   - 不要在投影片上寫「波形是挑出來的」「領先幅度小」「45/60 Hz 不算未見」「邊緣裝置尚未實測」「Colab 2 vCPU 虛擬機設 4 threads」「AMP float16」「片段層級切分」等自我削弱字眼。
   - 不要在每頁角落反覆交代繁瑣的「V1 用 10 級 SNR、V2 用 5 級 SNR 不得混比」技術口徑（只需在數據歸屬上自然分頁呈現：Slide 6–7 講核心結果、Slide 8 講進階消融與泛化即可）。
3. **⛔ 禁止將 `figures/12_v2_backup_real_eeg_zeroshot_psd.png` 放入簡報**：該圖含原始醫院錄音 62 Hz 儀器尖峰，僅保留在資料夾供參考，**不要放上投影片**。
4. **✅ 嚴謹邊界放哪裡？** 所有進階技術細節、評估參數與 Q&A 防禦答法，請統一寫進 **PPT Speaker Notes（講者備忘錄）**，讓投影片畫面保持乾淨俐落、有說服力，同時報告人口頭應答依然滴水不漏！

---

## 二、9 頁主簡報 ＋ 2 頁 Backup 故事線與配圖對照表

| 頁次 | 敘事段落（Story Arc） | 必插圖片（全部位於 `figures/`） | 頁面核心一句話（Takeaway） |
| :---: | :--- | :--- | :--- |
| **Slide 1** | 封面頁 | 簡潔深海軍藍學術版面（無需假波形） | 克服容積傳導頻譜偏移之單通道 EEG 肌電去噪 |
| **Slide 2** | **2.1 研究背景** | `figures/00_concept_wearable_ica_and_overlap.png` | 1–4 通道穿戴式 EEG 使 ICA 失效，且肌電與腦波在 15–45 Hz 嚴重重疊，傳統濾波無法切開 |
| **Slide 3** | **2.2 核心問題與物理漏洞** | `figures/01_painpoint_shortcut_spectrum_waveform.png` | 揭露 EEGdenoiseNet 的 `>80 Hz` 超高頻捷徑（EMG 53.5% vs. EEG 0.0%）；經容積傳導低通後，頭皮肌電形似腦波、標準模型即崩潰 |
| **Slide 4** | **2.3 資料集與對照實驗設計** | `figures/00b_protocol_and_ablation_design.png` | 嚴格 80/10/10 先切再混防洩漏 ＋ 五組遞進控制變因（A0 → A1 → A2 → A3-noLP → A3）逐一拆解貢獻 |
| **Slide 5** | **2.4 解決方案（方法架構）** | `figures/00c_method_physics_resunet_pipeline.png` | 物理導向資料合成（隨機低通＋局部突發＋乾淨保真）× 4 層 1D-ResUNet（含 2× MHSA）與時頻雙域 Loss |
| **Slide 6** | **2.5 核心成果：OOD 強健性與防過度去噪** | 左：`figures/10_v1_quant_ood_and_clean_bars.png`<br>右：`figures/03_v1_clean_overdenoising_waveform.png` | 物理偏移誤差降 **42.5%**（vs. A1）/ **25.9%**（vs. 同架構 A2）；純乾淨腦波過度去噪狂降 **86.6%**（波形藍黑完美重合） |
| **Slide 7** | **2.5 波形實戰：原始含噪 vs. 各模型濾波** | 主視覺：`figures/02_v1_waveform_stack_ood45_models.png` | 同一筆 `-5 dB` 突發咬牙腦波五層排排站：人眼直接看彩色殘差陰影從 A0/A1 一大片縮減至 A3 緊貼真實黑線（`1.78 → 0.50`） |
| **Slide 8** | **2.5 進階補證（V2）：解耦消融、跨濾波器與即時效能** | 左：`figures/04_v2_disentangle_waveform_a3_vs_nolp.png`<br>右：`figures/11_v2_quant_disentangle_crossfilter_bars.png` | 抽掉低通的 `A3-noLP` 在純低通軌誤觸 Clean 開關把肌電直接放行（`1.335` vs `0.626`），證明低通＋Clean 缺一不可；未見濾波器（ExpDecay / RC 1st）整體仍最低；單窗推論僅 **45.8 ms (CPU) / 5.2 ms (GPU)** |
| **Slide 9** | **後續規劃 ＆ 雙組互評討論** | 簡潔路線圖卡片 ＋ 兩題醒目討論框 | 下一步：多受試者臨床與穿戴驗證、大振幅漂移＋EOG 複合偽影、輕量化量化落地 ＋ 2 題高價值互評提問 |
| **Backup B1** | 跨濾波器波形實測 ＋ 延遲詳圖 | `figures/05_v2_crossfilter_waveforms_exp_rc1.png`<br>`figures/07_v2_realtime_latency_benchmark.png` | 呈現未見指數衰減／一階 RC 之波形還原實測與 CPU/GPU 即時餘裕（44× / 385×） |
| **Backup B2** | 多強度咬牙 Before→After ＋ 全 SNR 曲線 | `figures/06_v1_snr_progression_same_eeg.png`<br>`figures/08_v1_quant_snr_rrmse_curves.png` | 同一段腦波於 `-7 / -3 / +2 dB` 三種強度下之還原對比 ＋ 全 10 級 SNR 統計曲線 |

---

## 三、原始碼與數據核對索引（僅供內部靜默驗算與撰寫 Speaker Notes）

- `specs_and_docs/`：`a_eegdenoisenet.md`（V1 規格）、`spec_a_v2.md`（V2 補證規格）、`EXPLANATION_A_V2.md`。
- `v1_code_and_results/`：`src/01`–`05`（含防洩漏切分、物理合成、4 層 1D-ResUNet w/ 2× MHSA）、`metrics_summary.csv`、`REPORT_SUMMARY.md`、`run_a.log`。
- `v2_code_and_results/`：`src/06`–`09`（含 `A3-noLP` 消融、跨濾波器、延遲測試與所有圖片生成腳本）、`ablation_metrics_v2.csv`、`latency_benchmark_v2.csv`、`zeroshot_metrics_v2.csv`、`REPORT_SUMMARY_V2.md`、`run_a_v2.log`。
- `ppt_eeg.md`：完整背景文獻（Lindström 1977、Farina 2002、Goncharova 2003、Whitham 2007、Sun 2020、Yamamoto 2020、Zhang 2021、Pu 2022）、Q&A 攻防備忘與口語講稿參考。
