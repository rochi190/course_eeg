#!/usr/bin/env bash
set -euo pipefail

SESSION="eeg-a-v2"
CFG="/tmp/colab_a_v2_sessions.json"
V1_DIR="/home/neil/course_eeg/proj_a_eegdenoise"
V2_DIR="/home/neil/course_eeg/proj_a_eegdenoise_v2"
RUNNER="/home/neil/course_eeg/run_remote_checked.py"
DONE_FILE="$V2_DIR/.DONE"
FAIL_FILE="$V2_DIR/.FAILED"

rm -f "$DONE_FILE" "$FAIL_FILE" "$CFG" "${CFG}.lock"

cleanup() {
    local rc=$?
    echo "[cleanup] Stopping Colab session '$SESSION' (exit=$rc)..."
    colab --config "$CFG" stop -s "$SESSION" >/dev/null 2>&1 || true
    rm -f "$CFG" "${CFG}.lock" /tmp/proj_a_v1_pack.tar.gz /tmp/proj_a_v2_src.tar.gz /tmp/proj_a_v2_outputs.tar.gz
    if [ "$rc" -eq 0 ] && [ -f "$DONE_FILE" ]; then
        echo "[status] SPEC A-V2 COMPLETED SUCCESSFULLY"
    else
        touch "$FAIL_FILE"
        echo "[status] SPEC A-V2 FAILED (rc=$rc)"
    fi
}
trap cleanup EXIT

echo "[start] SPEC A-V2 pipeline started at $(date -Is)"

GPU_TYPE="L4"
if ! timeout 90 colab --config "$CFG" new -s "$SESSION" --gpu L4; then
    echo "[warn] L4 allocation failed/timed out (400/503); cleaning partial state and falling back to T4..."
    colab --config "$CFG" stop -s "$SESSION" >/dev/null 2>&1 || true
    GPU_TYPE="T4"
    colab --config "$CFG" new -s "$SESSION" --gpu T4
fi
echo "[gpu] Allocated $GPU_TYPE for session '$SESSION'"

run_remote() {
    local cmd="$1"
    local attempt=1
    local tmp_out
    tmp_out=$(mktemp)
    while [ "$attempt" -le 5 ]; do
        : > "$tmp_out"
        if timeout 300 colab --config "$CFG" exec -s "$SESSION" --timeout 180 --env "CMD=$cmd" -f "$RUNNER" > "$tmp_out" 2>&1; then
            if grep -q "^__REMOTE_OK__$" "$tmp_out"; then
                grep -v "^__REMOTE_OK__$" "$tmp_out" || true
                rm -f "$tmp_out"
                return 0
            fi
        fi
        cat "$tmp_out" || true
        echo "[warn] run_remote attempt $attempt failed for: $cmd; retrying in 8s..." >&2
        attempt=$((attempt + 1))
        sleep 8
    done
    rm -f "$tmp_out"
    echo "[error] run_remote failed after 5 attempts: $cmd" >&2
    return 1
}

echo "[install] Installing mne, boto3, pymatreader on '$SESSION'..."
colab --config "$CFG" install -s "$SESSION" mne boto3 pymatreader
run_remote "nvidia-smi && nproc && free -h && python3 -c 'import mne, boto3, pymatreader, torch, scipy, pandas, matplotlib; print(\"A-V2 imports OK, cuda=\", torch.cuda.is_available())'"

echo "[upload] Packaging V1 scripts/checkpoints + V2 scripts..."
tar -czf /tmp/proj_a_v1_pack.tar.gz -C "$V1_DIR" src outputs/checkpoints/exp_a1_best.pt outputs/checkpoints/exp_a2_best.pt outputs/checkpoints/exp_a3_best.pt
tar -czf /tmp/proj_a_v2_src.tar.gz -C "$V2_DIR" src
run_remote "mkdir -p /content/proj_a_eegdenoise/data/raw /content/proj_a_eegdenoise_v2"
colab --config "$CFG" upload -s "$SESSION" /tmp/proj_a_v1_pack.tar.gz /content/proj_a_v1_pack.tar.gz
colab --config "$CFG" upload -s "$SESSION" /tmp/proj_a_v2_src.tar.gz /content/proj_a_v2_src.tar.gz
colab --config "$CFG" upload -s "$SESSION" "$V1_DIR/data/raw/EEG_all_epochs_512hz.npy" /content/proj_a_eegdenoise/data/raw/EEG_all_epochs_512hz.npy
colab --config "$CFG" upload -s "$SESSION" "$V1_DIR/data/raw/EMG_all_epochs_512hz.npy" /content/proj_a_eegdenoise/data/raw/EMG_all_epochs_512hz.npy
run_remote "tar -xzf /content/proj_a_v1_pack.tar.gz -C /content/proj_a_eegdenoise && tar -xzf /content/proj_a_v2_src.tar.gz -C /content/proj_a_eegdenoise_v2 && rm -f /content/proj_a_v1_pack.tar.gz /content/proj_a_v2_src.tar.gz"

echo "[step] Launching resilient background pipeline for SPEC A-V2..."
run_remote "rm -f /content/proj_a_v2_bg.log /content/proj_a_v2_bg.rc /content/proj_a_v2_outputs.tar.gz && cat <<'EOF' > /content/run_bg_a_v2.sh
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
python3 -u /content/proj_a_eegdenoise/src/01_download_data.py
python3 -u /content/proj_a_eegdenoise_v2/src/06_ablation_stress_eval.py --epochs 70
python3 -u /content/proj_a_eegdenoise_v2/src/07_plot_v2_figures.py
ls -lh /content/proj_a_eegdenoise_v2/outputs /content/proj_a_eegdenoise_v2/outputs/figures
tar -czf /content/proj_a_v2_outputs.tar.gz -C /content/proj_a_eegdenoise_v2 outputs
EOF
chmod +x /content/run_bg_a_v2.sh
nohup bash -c '/content/run_bg_a_v2.sh; echo \$? > /content/proj_a_v2_bg.rc' > /content/proj_a_v2_bg.log 2>&1 < /dev/null &"

printed_lines=0
while true; do
    sleep 20
    status_out=$(run_remote "python3 -c '
import os
log = \"/content/proj_a_v2_bg.log\"
rcf = \"/content/proj_a_v2_bg.rc\"
skip = $printed_lines
if os.path.exists(log):
    with open(log) as f:
        lines = f.readlines()
    for l in lines[skip:]:
        print(\"LOG:\" + l, end=\"\")
    print(f\"__LINES__={len(lines)}\")
else:
    print(\"__LINES__=0\")
if os.path.exists(rcf):
    print(\"__RC__=\" + open(rcf).read().strip())
'")
    while IFS= read -r line; do
        case "$line" in
            LOG:*)
                echo "${line#LOG:}"
                ;;
            __LINES__=*)
                printed_lines="${line#__LINES__=}"
                ;;
            __RC__=*)
                rc_val="${line#__RC__=}"
                echo "[remote-bg] Finished with exit code $rc_val"
                if [ "$rc_val" != "0" ]; then
                    exit "$rc_val"
                fi
                break 2
                ;;
        esac
    done <<< "$status_out"
done

echo "[download] Downloading SPEC A-V2 outputs..."
colab --config "$CFG" download -s "$SESSION" /content/proj_a_v2_outputs.tar.gz /tmp/proj_a_v2_outputs.tar.gz
tar -xzf /tmp/proj_a_v2_outputs.tar.gz -C "$V2_DIR"

test -s "$V2_DIR/outputs/ablation_metrics_v2.csv"
test -s "$V2_DIR/outputs/zeroshot_metrics_v2.csv"
test -s "$V2_DIR/outputs/latency_benchmark_v2.csv"
test -s "$V2_DIR/outputs/REPORT_SUMMARY_V2.md"
test -s "$V2_DIR/outputs/figures/fig4_disentangle_ablation.png"
test -s "$V2_DIR/outputs/figures/fig5_real_eeg_zeroshot.png"

echo "GPU=$GPU_TYPE" > "$DONE_FILE"
echo "[done] SPEC A-V2 finished at $(date -Is)"
