#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=convert-snr
#SBATCH --output=/iopsstor/scratch/cscs/%u/data-mix-small/Megatron-LM/logs/slurm/conversion/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/data-mix-small/Megatron-LM/logs/slurm/conversion/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#
# convert-snr.sh — Megatron -> HuggingFace conversion for the SNR pretraining sweep.
#
# Three modes, auto-detected from env / args:
#
#   1. PER-ITER (env: SIZE, FW_EDU_RATIO, FW2_RATIO, SEED, CKPT_STEP set; no PLAN_FILE):
#        Convert ONE iter (torch_dist -> torch -> HF). Runs inside the
#        training container.
#
#   2. SBATCH WRAPPER (env: SLURM_JOB_ID and PLAN_FILE both set):
#        Inside an sbatch job, srun the container, loop over plan-file lines,
#        invoke this script in per-iter mode for each (cell, iter).
#
#   3. LAUNCHER (no slurm, no per-iter env):
#        Walk the 36 cells under $ROOT, list each cell's valid canonical
#        iters, write per-size plan files, and (with --submit) sbatch one
#        job per size.
#
# Examples:
#   bash convert-snr.sh                              # dry-run launcher (all sizes)
#   bash convert-snr.sh --submit --sizes 175M       # submit just the 175M sweep
#   bash convert-snr.sh --submit --partition normal --time 06:00:00
#   SIZE=175M FW_EDU_RATIO=30 FW2_RATIO=70 SEED=1904 CKPT_STEP=50000 bash convert-snr.sh
#
# Per-iter env (optional):
#   MEGATRON_LM_DIR     defaults /iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM
#   PRETRAIN_LOGS_DIR   defaults $MEGATRON_LM_DIR/logs/Meg-Runs/data-mix-small
#   STAGING_BASE        defaults /iopsstor/scratch/cscs/$USER/snr-hf-checkpoints
#   HF_TOKENIZER        defaults alehc/swissai-tokenizer
#   TEST_LOGITS=1       enable --test-logits
#   SKIP_PIP=1          skip pip install transformers (already in env)
#   KEEP_TMP_TORCH=1    keep intermediate torch ckpt after HF save

set -euo pipefail
SCRIPT_PATH="$(realpath "$0")"

# ============================================================================
# Mode 1: per-iter conversion (auto-detected via per-iter env vars)
# ============================================================================
if [[ -n "${SIZE:-}" && -n "${FW_EDU_RATIO:-}" && -n "${FW2_RATIO:-}" \
        && -n "${SEED:-}" && -n "${CKPT_STEP:-}" && -z "${PLAN_FILE:-}" ]]; then

    MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM}"
    PRETRAIN_LOGS_DIR="${PRETRAIN_LOGS_DIR:-$MEGATRON_LM_DIR/logs/Meg-Runs/data-mix-small}"
    STAGING_BASE="${STAGING_BASE:-/iopsstor/scratch/cscs/$USER/snr-hf-checkpoints}"
    TMP_TORCH_BASE="${TMP_TORCH_BASE:-$STAGING_BASE/_tmp_torch}"
    HF_TOKENIZER="${HF_TOKENIZER:-alehc/swissai-tokenizer}"

    EXP_NAME="apertus-${SIZE}-fwEdu${FW_EDU_RATIO}-fw2${FW2_RATIO}-seed${SEED}"
    CKPT_PATH="$PRETRAIN_LOGS_DIR/$EXP_NAME/checkpoints"
    ITER_DIR_NAME=$(printf "iter_%07d" "$CKPT_STEP")
    SRC_ITER_DIR="$CKPT_PATH/$ITER_DIR_NAME"

    if [[ ! -d "$SRC_ITER_DIR" ]]; then
        echo "[convert-snr] ERROR: source iter dir not found: $SRC_ITER_DIR" >&2
        exit 2
    fi
    # Skip empty/cleaned iter dirs (the May-3 sweep emptied non-canonical iters
    # — and many seed1797/seed28 canonical iters too). A torch_dist ckpt is only
    # loadable when the dir has both .metadata AND ≥1 .distcp shard.
    if [[ ! -f "$SRC_ITER_DIR/.metadata" ]] || ! compgen -G "$SRC_ITER_DIR/*.distcp" >/dev/null; then
        echo "[convert-snr] SKIP: $SRC_ITER_DIR has no .metadata or .distcp shards"
        exit 0
    fi

    TORCH_CKPT_SAVE_PATH="$TMP_TORCH_BASE/$EXP_NAME/$ITER_DIR_NAME"
    SAVE_DIR="$STAGING_BASE/$EXP_NAME/$ITER_DIR_NAME"
    mkdir -p "$TORCH_CKPT_SAVE_PATH" "$(dirname "$SAVE_DIR")"

    if [[ -f "$SAVE_DIR/config.json" ]] && \
       ( [[ -f "$SAVE_DIR/model.safetensors.index.json" ]] || [[ -f "$SAVE_DIR/model.safetensors" ]] ); then
        echo "[convert-snr] SKIP $EXP_NAME iter $CKPT_STEP: SAVE_DIR already populated"
        exit 0
    fi

    echo "[convert-snr] $EXP_NAME iter $CKPT_STEP -> $SAVE_DIR"
    [[ "${SKIP_PIP:-0}" == "1" ]] || pip install --quiet transformers==4.57.6
    export PYTHONPATH="$MEGATRON_LM_DIR:${PYTHONPATH:-}"
    cd "$MEGATRON_LM_DIR"

    extra=()
    [[ "${TEST_LOGITS:-0}" == "1" ]] && extra+=("--test-logits")

    echo "[convert-snr] Step 1/2: torch_dist -> torch"
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc-per-node=1 \
        "$MEGATRON_LM_DIR/scripts/conversion/torchdist_2_torch.py" \
        --bf16 \
        --load "$CKPT_PATH" \
        --ckpt-step "$CKPT_STEP" \
        --ckpt-convert-save "$TORCH_CKPT_SAVE_PATH"

    echo "[convert-snr] Step 2/2: core (torch) -> HF (swissai_hf)"
    python "$MEGATRON_LM_DIR/tools/checkpoint/convert.py" \
        --model-type GPT \
        --loader core \
        --saver swissai_hf \
        --load-dir "$TORCH_CKPT_SAVE_PATH/torch" \
        --save-dir "$SAVE_DIR" \
        --hf-tokenizer "$HF_TOKENIZER" \
        "${extra[@]}"

    [[ "${KEEP_TMP_TORCH:-0}" == "1" ]] || rm -rf "$TORCH_CKPT_SAVE_PATH"
    echo "[convert-snr] DONE $SAVE_DIR"
    exit 0
fi

# ============================================================================
# Mode 2: sbatch wrapper (running in slurm + PLAN_FILE) — srun container,
# loop the plan
# ============================================================================
if [[ -n "${SLURM_JOB_ID:-}" && -n "${PLAN_FILE:-}" ]]; then
    [[ -f "$PLAN_FILE" ]] || { echo "[convert-snr/sbatch] PLAN_FILE not found: $PLAN_FILE" >&2; exit 2; }

    MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM}"
    CONTAINER_TOML="${CONTAINER_TOML:-/capstor/store/cscs/swissai/a139/containers/ngc_25-11-nemo-alps3.toml}"
    mkdir -p /iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM/logs/slurm/conversion

    echo "[$(date)] convert-snr/sbatch starting, plan=$PLAN_FILE"
    cat "$PLAN_FILE"
    echo "---"

    # Forward HF cache vars + force offline so per-iter tokenizer loads hit local
    # cache (the saver and Megatron's HuggingFaceTokenizer both fetch the
    # `alehc/swissai-tokenizer` repo, and the `/api/models/` freshness check
    # will burn the team-plan rate-limit on a 9-cell × ~13-iter sweep).
    INNER_EXPORTS="export MEGATRON_LM_DIR='$MEGATRON_LM_DIR' PLAN_FILE='$PLAN_FILE' \
HF_HOME='${HF_HOME:-/iopsstor/scratch/cscs/$USER/hf_home}' \
HF_HUB_CACHE='${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/users/$USER/hf_models}' \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

    srun --mpi=pmix \
        --network=disable_rdzv_get \
        --environment="$CONTAINER_TOML" \
        --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        -lu bash -c "
$INNER_EXPORTS
total_ok=0
total_fail=0
failed_list=()
unset PLAN_FILE  # so the per-iter recursion takes the per-iter mode branch
while IFS= read -r line || [[ -n \"\$line\" ]]; do
    line=\${line%%#*}
    line=\$(echo \"\$line\" | xargs)
    [[ -z \"\$line\" ]] && continue
    read -ra parts <<< \"\$line\"
    export SIZE=\${parts[0]} FW_EDU_RATIO=\${parts[1]} FW2_RATIO=\${parts[2]} SEED=\${parts[3]}
    for it in \"\${parts[@]:4}\"; do
        export CKPT_STEP=\"\$it\"
        echo \"[\$(date)] === \$SIZE fwEdu\$FW_EDU_RATIO seed\$SEED iter \$it ===\"
        if bash $SCRIPT_PATH; then
            total_ok=\$((total_ok+1))
        else
            rc=\$?
            echo \"[\$(date)] FAILED rc=\$rc for \$SIZE fwEdu\$FW_EDU_RATIO seed\$SEED iter \$it\"
            total_fail=\$((total_fail+1))
            failed_list+=(\"\$SIZE/fwEdu\$FW_EDU_RATIO/seed\$SEED/iter\$it\")
        fi
    done
done < '$PLAN_FILE'
echo \"[\$(date)] convert-snr/sbatch finished: ok=\$total_ok fail=\$total_fail\"
if (( total_fail > 0 )); then
    echo 'failed:'
    printf '  - %s\\n' \"\${failed_list[@]}\"
fi
"

    # Refresh the pretraining progress plot — newly-staged HF dirs now show
    # up as yellow via the filesystem check. Runs on the slurm host (outside
    # the container) with the user's snr conda env. Best-effort.
    PRETRAIN_PROGRESS="$(dirname "$SCRIPT_PATH")/../pretrain_progress.py"
    SNR_PY="$HOME/miniconda3/envs/snr/bin/python"
    if [[ -f "$PRETRAIN_PROGRESS" && -x "$SNR_PY" ]]; then
        "$SNR_PY" "$PRETRAIN_PROGRESS" >/dev/null \
            || echo "[convert-snr/sbatch] warning: progress plot refresh failed"
    fi
    exit 0
fi

# ============================================================================
# Mode 3: launcher — walk cells, write plan files, optionally sbatch
# ============================================================================
ROOT="/iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small"
PLAN_DIR="/iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM/logs/conversion-plans"
# Per-seed iter policy mirrors snr_progress.py (ITERS_SEED1904 / ITERS_OTHER).
# Update both lists together when the policy changes.
#   seed1904       → 9 picks from the canonical 13-iter set
#   seed28/seed1797 → 10000-stepped grid (10k/20k/30k iters NOT canonical)
ITERS_SEED1904=(6000 12000 22000 28000 42000 44000 46000 48000 50000)
ITERS_OTHER=(6000 10000 20000 30000 42000 44000 46000 48000 50000)
SEEDS=(1904 1797 28)
MIXES=("30 70" "60 40" "90 10")
DEFAULT_SIZES=(175M 350M 600M 1B)

SIZES=()
SUBMIT=0
PARTITION=""
TIME=""
RESERVATION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --sizes) shift; while [[ $# -gt 0 && "$1" != --* ]]; do SIZES+=("$1"); shift; done ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --reservation) RESERVATION="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --plan-dir) PLAN_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '1,40p' "$SCRIPT_PATH"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ ${#SIZES[@]} -eq 0 ]] && SIZES=("${DEFAULT_SIZES[@]}")
mkdir -p "$PLAN_DIR"
TS=$(date +%Y%m%d-%H%M%S)

valid_iters_for_cell() {
    local cell=$1 seed=$2 d out=() iters
    if [[ "$seed" == "1904" ]]; then
        iters=("${ITERS_SEED1904[@]}")
    else
        iters=("${ITERS_OTHER[@]}")
    fi
    for it in "${iters[@]}"; do
        d="$ROOT/$cell/checkpoints/iter_$(printf '%07d' "$it")"
        [[ -d "$d" ]] || continue
        [[ -f "$d/.metadata" ]] || continue
        compgen -G "$d/*.distcp" >/dev/null || continue
        out+=("$it")
    done
    echo "${out[*]}"
}

# To launch in debug mode, update time to 01:30:00
partition_for_size() { case "$1" in 175M) echo normal ;; *) echo normal ;; esac; }
# Per-iter wall is ~35-40s across ALL sizes (sacct on 12 jobs with done>=5,
# 2026-04..05): 175M=35s, 350M=36s, 600M=38s, 1B=38s. The cost is dominated
# by torch_dist->torch shard load + safetensors write, NOT by GPU compute,
# so size barely matters. Full 9-cell × 9-iter per-size sweep = 81 iters ×
# 40s ≈ 54 min wall + ~5 min pip+container overhead. 02:00:00 gives a
# generous safety margin (slow node / cold cache / outliers up to 47s/iter).
time_for_size() {
    case "$1" in
        175M|350M|600M|1B) echo "02:00:00" ;;
    esac
}

for SIZE in "${SIZES[@]}"; do
    plan="$PLAN_DIR/plan-${SIZE}-${TS}.txt"
    {
        echo "# Auto-generated by convert-snr.sh launcher on $(date)"
        echo "# Format: SIZE FW_EDU_RATIO FW2_RATIO SEED ITER1 ITER2 ..."
        for SEED in "${SEEDS[@]}"; do
            for mix in "${MIXES[@]}"; do
                read -r FW_EDU_RATIO FW2_RATIO <<< "$mix"
                cell="apertus-${SIZE}-fwEdu${FW_EDU_RATIO}-fw2${FW2_RATIO}-seed${SEED}"
                [[ -d "$ROOT/$cell/checkpoints" ]] || { echo "# SKIP $cell (no checkpoints dir)"; continue; }
                iters=$(valid_iters_for_cell "$cell" "$SEED")
                if [[ -z "$iters" ]]; then
                    echo "# SKIP $cell (no valid iters on disk for the per-seed policy)"
                    continue
                fi
                echo "$SIZE $FW_EDU_RATIO $FW2_RATIO $SEED $iters"
            done
        done
    } > "$plan"

    cell_count=$(grep -vc '^#' "$plan" || true)
    iter_count=$(grep -v '^#' "$plan" | awk '{n+=NF-4} END {print n+0}')
    echo "=== $SIZE plan ($cell_count cells, $iter_count iters): $plan ==="
    cat "$plan"

    if [[ $cell_count -eq 0 ]]; then
        echo "[convert-snr/launcher] $SIZE: nothing to convert, skipping"
        continue
    fi

    part="${PARTITION:-$(partition_for_size "$SIZE")}"
    walltime="${TIME:-$(time_for_size "$SIZE")}"

    sbatch_args=(--partition="$part" --time="$walltime"
                 --job-name="convert-snr-$SIZE"
                 --export=ALL,PLAN_FILE="$plan")
    [[ -n "$RESERVATION" ]] && sbatch_args+=(--reservation="$RESERVATION")

    if [[ $SUBMIT -eq 1 ]]; then
        sbatch "${sbatch_args[@]}" "$SCRIPT_PATH"
    else
        echo "[convert-snr/launcher] $SIZE: would sbatch ${sbatch_args[*]} $SCRIPT_PATH"
        echo "  pass --submit to actually submit"
    fi
done
