#!/usr/bin/env bash
set -euo pipefail

SESSION="eeg-b-v2"
CFG="/tmp/colab_b_v2_sessions.json"
V1_DIR="/home/neil/course_eeg/proj_b_ahepa_ad_ftd"
V2_DIR="/home/neil/course_eeg/proj_b_ahepa_ad_ftd_v2"
RUNNER="/home/neil/course_eeg/run_remote_checked.py"
DONE_FILE="$V2_DIR/.DONE"
FAIL_FILE="$V2_DIR/.FAILED"

rm -f "$DONE_FILE" "$FAIL_FILE" "$CFG" "${CFG}.lock"

cleanup() {
    local rc=$?
    echo "[cleanup] Stopping Colab session '$SESSION' (exit=$rc)..."
    colab --config "$CFG" stop -s "$SESSION" >/dev/null 2>&1 || true
    rm -f "$CFG" "${CFG}.lock" /tmp/proj_b_v1_metrics.tar.gz /tmp/proj_b_v2_src.tar.gz /tmp/proj_b_v2_outputs.tar.gz
    if [ "$rc" -eq 0 ] && [ -f "$DONE_FILE" ]; then
        echo "[status] SPEC B-V2 COMPLETED SUCCESSFULLY"
    else
        touch "$FAIL_FILE"
        echo "[status] SPEC B-V2 FAILED (rc=$rc)"
    fi
}
trap cleanup EXIT

echo "[start] SPEC B-V2 pipeline started at $(date -Is)"

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

echo "[install] Installing mne, pyriemann, lightgbm, boto3, pymatreader on '$SESSION'..."
colab --config "$CFG" install -s "$SESSION" mne pyriemann lightgbm boto3 pymatreader
run_remote "nvidia-smi && nproc && free -h && python3 -c 'import mne, pyriemann, lightgbm, boto3, pymatreader, torch, sklearn, scipy, matplotlib; from pyriemann.estimation import Covariances; from pyriemann.tangentspace import TangentSpace; from pyriemann.utils.mean import mean_riemann; print(\"B-V2 imports OK, pyriemann=\", pyriemann.__version__, \"cuda=\", torch.cuda.is_available())'"

echo "[upload] Uploading V1 summary CSV + V2 scripts..."
tar -czf /tmp/proj_b_v1_metrics.tar.gz -C "$V1_DIR" outputs/metrics_summary.csv
tar -czf /tmp/proj_b_v2_src.tar.gz -C "$V2_DIR" src
run_remote "mkdir -p /content/proj_b_ahepa_ad_ftd /content/proj_b_ahepa_ad_ftd_v2"
colab --config "$CFG" upload -s "$SESSION" /tmp/proj_b_v1_metrics.tar.gz /content/proj_b_v1_metrics.tar.gz
colab --config "$CFG" upload -s "$SESSION" /tmp/proj_b_v2_src.tar.gz /content/proj_b_v2_src.tar.gz
run_remote "tar -xzf /content/proj_b_v1_metrics.tar.gz -C /content/proj_b_ahepa_ad_ftd && tar -xzf /content/proj_b_v2_src.tar.gz -C /content/proj_b_ahepa_ad_ftd_v2 && rm -f /content/proj_b_v1_metrics.tar.gz /content/proj_b_v2_src.tar.gz"

echo "[step] Launching resilient background pipeline for SPEC B-V2..."
run_remote "rm -f /content/proj_b_v2_bg.log /content/proj_b_v2_bg.rc /content/proj_b_v2_outputs.tar.gz && cat <<'EOF' > /content/run_bg_b_v2.sh
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
python3 -u /content/proj_b_ahepa_ad_ftd_v2/src/06_extract_v2_features.py --cache-name ahepa_v2_features.npz
python3 -u /content/proj_b_ahepa_ad_ftd_v2/src/07_run_rescue_benchmarks.py --cache-name ahepa_v2_features.npz
python3 -u /content/proj_b_ahepa_ad_ftd_v2/src/08_plot_v2_figures.py
ls -lh /content/proj_b_ahepa_ad_ftd_v2/outputs /content/proj_b_ahepa_ad_ftd_v2/outputs/figures
tar -czf /content/proj_b_v2_outputs.tar.gz -C /content/proj_b_ahepa_ad_ftd_v2 outputs
EOF
chmod +x /content/run_bg_b_v2.sh
nohup bash -c '/content/run_bg_b_v2.sh; echo \$? > /content/proj_b_v2_bg.rc' > /content/proj_b_v2_bg.log 2>&1 < /dev/null &"

printed_lines=0
while true; do
    sleep 20
    status_out=$(run_remote "python3 -c '
import os
log = \"/content/proj_b_v2_bg.log\"
rcf = \"/content/proj_b_v2_bg.rc\"
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

echo "[download] Downloading SPEC B-V2 outputs..."
colab --config "$CFG" download -s "$SESSION" /content/proj_b_v2_outputs.tar.gz /tmp/proj_b_v2_outputs.tar.gz
tar -xzf /tmp/proj_b_v2_outputs.tar.gz -C "$V2_DIR"

test -s "$V2_DIR/outputs/rescue_metrics_summary.csv"
test -s "$V2_DIR/outputs/REPORT_SUMMARY_V2.md"
test -s "$V2_DIR/outputs/figures/fig4_ad_ftd_rescue.png"
test -s "$V2_DIR/outputs/figures/fig5_v2_all_tasks_vs_2026sota.png"
test -s "$V2_DIR/outputs/figures/fig6_neuro_biomarkers_boxplot.png"

python3 -c "
import csv, math
with open('$V2_DIR/outputs/rescue_metrics_summary.csv') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 24, f'Expected 24 rows, got {len(rows)}'
for r in rows:
    assert int(r['n_seeds']) == 5
    for k in ['macro_f1_mean', 'accuracy_mean', 'auroc_mean']:
        assert math.isfinite(float(r[k])), f'Non-finite {k} in {r}'
print('[verify] rescue_metrics_summary.csv verified: 24 rows, 5 seeds, zero NaNs')
"

echo "GPU=$GPU_TYPE" > "$DONE_FILE"
echo "[done] SPEC B-V2 finished at $(date -Is)"
