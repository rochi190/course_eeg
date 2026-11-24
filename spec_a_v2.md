# SPEC A-V2: Disentangled OOD Ablation, Cross-Filter Stress Test & Real-EEG Zero-Shot Validation

## 1. Objective
Extend `proj_a_eegdenoise` without breaking existing scripts. Answer 4 core reviewer questions:
1. **Disentanglement**: Separate the gains of *volume-conduction lowpass filtering* vs. *burst masking* on both test tracks and training ablations.
2. **Cross-Filter Generalization**: Prove `Exp A3` did not overfit to 4th-order Butterworth roll-off by evaluating on non-Butterworth physical attenuation curves.
3. **Real Clinical EEG Zero-Shot Check**: Evaluate trained models on real uncleaned clinical EEG segments (containing natural muscle activity) to verify high-frequency suppression and $8\text{--}13\text{ Hz}$ Alpha peak preservation.
4. **Deployment Efficiency**: Benchmark model parameters and single-window ($2\text{ s}$, 1,024 pts) inference latency on CPU and GPU.

## 2. Environment & Checkpoint Handling (Colab A100 Ready)
- Working directory: `proj_a_eegdenoise/`
- Reuse exact data download & strict 80/10/10 independent EEG/EMG split from `src/01_download_data.py` and `src/02_synthesis.py` (fixed seed, assert zero overlap between train/val/test raw indices).
- **Checkpoint Check**:
  - Check if trained weights (`A1.pt`, `A2.pt`, `A3.pt`) exist under `outputs/checkpoints/`.
  - If missing (e.g., fresh Colab runtime), run `04_train_eval.py` (or integrated runner) to train & save `A1` (1D-ResCNN, ID, MSE), `A2` (1D-ResUNet, ID, MSE+MR-STFT), and `A3` (1D-ResUNet, Physics LP+Burst+Clean, MSE+MR-STFT).
- **New Mandatory Ablation Model (`Exp A3-noLP`)**:
  - Architecture & Loss: Identical to `Exp A3` (1D-ResUNet + MSE + MR-STFT).
  - Training Augmentation: Keep **Burst Masking (30%)** and **Clean Preservation (10%)**, but **DISABLE random lowpass filtering** on EMG.
  - Purpose: Directly isolate the contribution of physics-informed lowpass augmentation (`A3` vs. `A3-noLP`).

## 3. New Evaluation Tracks (Zero Retraining Required for A1/A2/A3)
Evaluate `A0 (Bandpass 1-40Hz)`, `A1`, `A2`, `A3-noLP`, and `A3` across SNRs `[-7, -5, -3, 0, +2] dB` on 5 new tracks (all synthesized strictly from held-out Test EEG & Test EMG):

1. **`Test-Pure-LP45` (Lowpass Only, No Burst)**:
   - Apply 4th-order Butterworth lowpass ($f_c = 45\text{ Hz}$) to full $2\text{ s}$ EMG, then RMS-scale to target SNR and add to EEG.
2. **`Test-Pure-Burst` (Burst Only, No Lowpass)**:
   - Unfiltered full-band EMG masked to random active window $0.3\text{--}1.5\text{ s}$ (cosine taper $25\text{ ms}$ edges), RMS-scaled to target SNR.
3. **`Test-ExpDecay` (Non-Butterworth Volume-Conduction Filter)**:
   - Apply smooth exponential spectral attenuation in `rfft` domain to full $2\text{ s}$ EMG:
     $$H(f) = e^{-\alpha f}, \quad \alpha = 0.015$$
     (approx. $-6\text{ dB}$ at $46\text{ Hz}$, $-10.4\text{ dB}$ at $80\text{ Hz}$), then `irfft`, RMS-scale to target SNR.
4. **`Test-RC1st` (1st-Order RC Lowpass Filter)**:
   - Apply 1st-order Butterworth/RC lowpass (`order=1`, $f_c = 40\text{ Hz}$) to EMG (shallow $-20\text{ dB/decade}$ slope, very different from 4th-order Butterworth).
5. **`Test-Composite-Drift` (Limitation Stress Test)**:
   - `Test-Pure-LP45` EMG + synthetic electrode motion baseline drift (random-phase sinusoid $f \in [0.5, 1.5]\text{ Hz}$, $\text{RMS} = 0.35 \times \text{RMS}(\text{clean\_EEG})$). Log clearly as unseen out-of-scope drift stress test.

## 4. Real Clinical EEG Zero-Shot Test (No Ground Truth)
- Download 2 raw uncleaned clinical EEG files (not `derivatives/`) from OpenNeuro `ds004504` via AWS S3 (`aws s3 cp --no-sign-request s3://openneuro.org/ds004504/sub-001/eeg/ ...` and `sub-037` CN subject) or `openneuro-py`.
- Load via `mne.io.read_raw_eeglab()`, resample `500 Hz -> 512 Hz`, apply minimal $1\text{ Hz}$ highpass + $50\text{ Hz}$ notch, slice into $2\text{ s}$ (1,024-pt) single-channel windows (prioritize frontal/temporal channels `Fp1, Fp2, F7, F8, T7, T8` and occipital `O1, O2` for alpha check).
- Select top-20 windows with highest $30\text{--}80\text{ Hz}$ relative power (natural high-EMG segments) AND 20 occipital eyes-closed windows with strong $8\text{--}13\text{ Hz}$ Alpha peak.
- Normalize each window (zero-mean, unit-std), run `A1`, `A2`, `A3` inference, and compute:
  - **High-Freq Suppression ($30\text{--}80\text{ Hz}$ PSD drop in dB)** on high-EMG segments.
  - **Alpha Peak Preservation ($8\text{--}13\text{ Hz}$ PSD power ratio & peak frequency shift in Hz)** on occipital segments.

## 5. Deployment Latency Benchmark
- Measure for `A1`, `A2`, `A3`:
  - Total trainable parameters (`M` / `K`).
  - Single-window inference time (`batch_size=1, shape=[1, 1, 1024]`): 20 warmup runs + 200 timed runs (report $\text{mean} \pm \text{std}$ in `ms`) on both **CPU** (`torch.set_num_threads(4)`) and **GPU** (`torch.cuda.synchronize()`).

## 6. Required New Scripts & Outputs
- `src/06_ablation_stress_eval.py`: Runs `A3-noLP` training (if needed) + all 5 stress tracks + real-EEG zero-shot + latency benchmark.
- `src/07_plot_v2_figures.py`: Generates 300 DPI publication-grade figures (strictly check no overlapping text/legends):
  1. `outputs/figures/fig4_disentangle_ablation.png`: Grouped bar chart comparing `A1`, `A2`, `A3-noLP`, `A3` ($\text{RRMSE}_t$) across `Test-Pure-LP45`, `Test-Pure-Burst`, `Test-ExpDecay`, `Test-RC1st`.
  2. `outputs/figures/fig5_real_eeg_zeroshot.png`: 2-panel Welch PSD plot (Left: High-EMG frontal/temporal real segments showing $30\text{--}80\text{ Hz}$ suppression; Right: Occipital eyes-closed segments showing $8\text{--}13\text{ Hz}$ Alpha peak preservation across Raw / A1 / A3).
- `outputs/ablation_metrics_v2.csv`: Complete numerical table.
- `outputs/REPORT_SUMMARY_V2.md`: Clean Markdown tables ready for PPT/Q&A copy-paste.