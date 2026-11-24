# SPEC B-V2: Rescuing Strict Subject-Independent AD vs. FTD Classification on AHEPA (ds004504)

## 1. Diagnosis & Rescue Objectives
In V1 (`proj_b_ahepa_ad_ftd`), all models collapsed on `AD_vs_FTD` (Subject Macro-F1: $46.4\%\text{--}56.2\%$, FTD Specificity: $28.7\%\text{--}41.7\%$), and `LightGAT` underperformed `Linear SVM` across tasks.
V2 fixes four root causes under **strict 5-Fold Subject-Independent CV (`StratifiedGroupKFold`, 5 seeds)**:
1. **Class Imbalance & Threshold Bias**: AD (36) vs. FTD (23) caused majority-class collapse. Apply inverse class weights + **strictly inner-CV threshold calibration** (never peek at test fold).
2. **Connectivity Variance**: Replace noisy $4\text{ s}$ wPLI windows with **$20\text{ s}$ windows (50% overlap)** and subject-level aggregated connectivity/covariance.
3. **AD vs. FTD Neurophysiological Biomarkers**: Add Anterior-Posterior (Frontal vs. Parietal-Occipital) power ratios, Regional Slowing ratios, Individual Alpha Frequency (IAF), and **$1/f$ Aperiodic Spectral Slope** (pure NumPy/SciPy robust log-log fit, avoiding `specparam` RC dependency issues).
4. **Small-Sample Geometry & Stabilized DL**: Introduce Multi-Band **Riemannian Tangent Space** (`pyriemann`), **Neuro-Feature Regularized Models**, and a **Stabilized Ultra-Light ChebGCN/ResGCN ($< 25\text{K}$ params)** with balanced loss + label smoothing.

## 2. Environment & Dependencies (Colab Ready)
~~~bash
pip install -q mne pyriemann lightgbm scikit-learn scipy torch
~~~
*(Note: `pyriemann>=0.9` / `0.12` is compatible with NumPy 2.x and scikit-learn pipelines.)*

## 3. Data Loading & Strict Anti-Leakage Protocol
- Reuse `data/ds004504/derivatives/` and `participants.tsv` (`Group`: A=36, F=23, C=29). If missing on fresh Colab, download via AWS S3 `--no-sign-request` (exact same command as `01_download_bids.py`).
- Resample to $250\text{ Hz}$, bandpass $0.5\text{--}45\text{ Hz}$.
- **Channel ROI Mapping** (verify exact channel names from `raw.ch_names`; handle `T3/T4/T5/T6` and `T7/T8/P7/P8` aliases automatically):
  - **Frontal (F)**: `Fp1, Fp2, F7, F3, Fz, F4, F8`
  - **Central (C)**: `C3, Cz, C4`
  - **Temporal (T)**: `T7, T8` (or `T3, T4`)
  - **Parietal-Occipital (PO)**: `P7, P3, Pz, P4, P8, O1, O2` (or `T5, P3, Pz, P4, T6, O1, O2`)
- **Validation Protocol**:
  - `StratifiedGroupKFold(n_splits=5)` grouped by `participant_id`, repeated across **5 fixed seeds** (`[42, 100, 2024, 2025, 2026]`) on all 3 binary tasks (`AD_vs_CN`, `AD_vs_FTD`, `FTD_vs_CN`) + 3-class (`AD_vs_FTD_vs_CN`).
  - **MANDATORY ASSERTION**: Every fold must execute `assert set(train_ids).isdisjoint(set(test_ids))`.
  - **Zero-Leakage Transformations**: ALL `StandardScaler`, `TangentSpace.fit()`, feature selectors (`SelectKBest`), and binary probability threshold tuners MUST be fit **strictly inside `train_ids`** (use inner 4-fold `StratifiedGroupKFold` on `train_ids` to tune decision threshold maximizing Subject Macro-F1, then apply fixed threshold to `test_ids`).

## 4. Feature Engineering (20s Windows, 10s Stride)
Slice each subject's recording into $20\text{ s}$ windows (5,000 samples @ $250\text{ Hz}$, 50% overlap; drop windows with extreme amplitude artifact $> 5 \times \text{IQR}$).

### Feature Set 1: Neuro-Biomarkers (`X_neuro`, computed per window + subject-level mean & IQR)
1. **Band Relative PSD** (Welch `nperseg=1000`, 50% overlap) per channel & per ROI (`F, C, T, PO`) across 5 bands: Delta ($0.5\text{--}4\text{ Hz}$), Theta ($4\text{--}8\text{ Hz}$), Alpha ($8\text{--}13\text{ Hz}$), Beta ($13\text{--}25\text{ Hz}$), Gamma ($25\text{--}45\text{ Hz}$).
2. **Anterior-Posterior (AP) & Frontal-Temporal Ratios** (key AD vs. FTD discriminator):
   - $\log(\text{Power}_F / \text{Power}_{PO})$ and $\log(\text{Power}_F / \text{Power}_T)$ for Delta, Theta, Alpha, Beta.
3. **Slowing & Peak Metrics** per ROI:
   - Slowing Ratio: $(\text{Delta} + \text{Theta}) / (\text{Alpha} + \text{Beta})$.
   - **Individual Alpha Frequency (IAF)**: Center-of-gravity frequency within $7\text{--}13\text{ Hz}$.
   - **Spectral Entropy** ($1\text{--}45\text{ Hz}$).
4. **Aperiodic $1/f$ Slope & Offset** per ROI (zero extra package):
   - Fit `HuberRegressor` on $\log_{10}(\text{freq})$ vs. $\log_{10}(\text{PSD})$ over $2\text{--}40\text{ Hz}$, **masking out $7\text{--}14\text{ Hz}$** to avoid alpha oscillation distortion. Extract slope & intercept.
5. **Stabilized Long-Window wPLI**:
   - Compute $19 \times 19$ debiased wPLI on $20\text{ s}$ windows in Theta ($4\text{--}8\text{ Hz}$) and Alpha ($8\text{--}13\text{ Hz}$), extract ROI-to-ROI mean connectivity ($10\text{ ROI pairs} \times 2\text{ bands} = 20$ compact features) + top-eigenvalue graph summary.

### Feature Set 2: Multi-Band Spatial Covariance Matrices (`X_cov`, for Riemannian Model)
- Filter each $20\text{ s}$ window into 5 bands: $1\text{--}4\text{ Hz}$, $4\text{--}8\text{ Hz}$, $8\text{--}13\text{ Hz}$, $13\text{--}25\text{ Hz}$, $1\text{--}40\text{ Hz}$.
- Estimate regularized $19 \times 19$ SPD covariance matrix per band via `pyriemann.estimation.Covariances(estimator='lwf')` (Ledoit-Wolf shrinkage guarantees SPD).
- Also compute **Subject-level Fréchet / Riemannian Mean Covariance** per band (`mean_riemann`) so each subject has one clean SPD matrix per band (dramatically reduces epoch noise).

## 5. Experimental Matrix (Compare Directly vs. V1 & 2026 SOTA)
Report **Subject-level Macro-F1, Accuracy, AUROC, Sensitivity, Specificity** ($\text{Mean} \pm \text{Std}$ across 5 seeds):

1. **`Exp B4a` (Riemannian Tangent Space — Epoch Soft Voting)**:
   - Per band: `TangentSpace(metric='riemann')` fit on train fold $\rightarrow$ concatenate 5 bands $\rightarrow$ `StandardScaler` $\rightarrow$ `LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)` $\rightarrow$ subject-level mean probability $\rightarrow$ inner-CV calibrated threshold.
2. **`Exp B4b` (Riemannian Tangent Space — Subject-Level Mean Covariance)**:
   - Directly classify the 88 subject-level Riemannian mean covariance matrices (fit `TangentSpace` strictly on train subjects) $\rightarrow$ `StandardScaler` $\rightarrow$ `LogisticRegression(C=0.1/0.5, class_weight='balanced')` + `LinearSVC` (calibrated).
3. **`Exp B5a` (Neuro-Biomarkers — Balanced ElasticNet LogReg)**:
   - Subject-level aggregated `X_neuro` (mean + IQR across $20\text{ s}$ windows) $\rightarrow$ `StandardScaler` $\rightarrow$ `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.2, class_weight='balanced')`.
4. **`Exp B5b` (Neuro-Biomarkers — Balanced Regularized LightGBM)**:
   - `LGBMClassifier(n_estimators=100, max_depth=3, num_leaves=7, min_child_samples=8, subsample=0.8, colsample_bytree=0.7, class_weight='balanced')`.
5. **`Exp B6` (Stabilized Deep Learning — Ultra-Light Spectral ResGCN, $< 25\text{K}$ Params)**:
   - Input: $20\text{ s}$ window 19-node graph (node features = 5-band RelPSD + IAF + $1/f$ slope + Spectral Entropy; adjacency = stabilized Theta/Alpha wPLI + identity self-loop, symmetric normalized $D^{-1/2} A D^{-1/2}$).
   - Architecture: 2-layer Residual GCN (`hidden=24`, LayerNorm, GELU, Dropout=0.5) + Global Mean/Max Pooling $\rightarrow$ Linear head (verify total params $< 25,000$).
   - Training: Inverse-frequency `CrossEntropyLoss(weight=..., label_smoothing=0.1)`, AdamW (`lr=1e-3, weight_decay=5e-2`), CosineAnnealingLR, early stopping on inner validation fold. Soft-vote probabilities per subject.
6. **`Exp B7` (Balanced Multimodal Ensemble — B4b + B5a + B6)**:
   - Weighted soft-voting average of predicted subject probabilities from `B4b (Riemannian)` + `B5a (Neuro-ElasticNet)` + `B6 (ResGCN)` (fixed weights `0.4 / 0.4 / 0.2`, no test leakage), with inner-CV threshold calibration.

## 6. Required New Scripts & Outputs
- `src/06_extract_v2_features.py`: Extracts $20\text{ s}$ `X_neuro` and multi-band SPD `X_cov` caches to `data/cache_v2/`.
- `src/07_run_rescue_benchmarks.py`: Executes `B4a`, `B4b`, `B5a`, `B5b`, `B6`, `B7` with strict zero-leakage assertions and saves `outputs/rescue_metrics_summary.csv` and `outputs/REPORT_SUMMARY_V2.md`.
- `src/08_plot_v2_figures.py`: Generates 300 DPI clean figures (**fix text overlap issue seen in V1 `fig3`** by placing reference line labels cleanly outside/above bars with white bbox):
  1. `outputs/figures/fig4_ad_ftd_rescue.png`: Side-by-side comparison of V1 (`ShallowConvNet`, `LightGAT`) vs. V2 (`B4b Riemannian`, `B5a Neuro-LogReg`, `B6 ResGCN`, `B7 Ensemble`) on `AD_vs_FTD` showing **Macro-F1, AD Sensitivity, and FTD Specificity** (highlighting recovery from $28\%\text{--}41\%$ specificity collapse).
  2. `outputs/figures/fig5_v2_all_tasks_vs_2026sota.png`: Clean updated version of `fig3` across `AD_vs_CN`, `AD_vs_FTD`, `FTD_vs_CN` vs. 2026 SOTA dashed lines ($75.82\%$, $70.65\%$, $56.35\%$) with zero overlapping labels.
  3. `outputs/figures/fig6_neuro_biomarkers_boxplot.png`: Boxplots + stripplots of `Frontal/PO Alpha Ratio`, `Slowing Ratio`, and `$1/f$ Aperiodic Slope` across `AD`, `FTD`, and `CN` (annotated with Mann-Whitney U $p$-values).