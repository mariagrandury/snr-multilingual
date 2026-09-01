#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=convert-snr
#SBATCH --output=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/conversion/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/conversion/%x-%j.err
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
#   1. PER-ITER (env: CKPT_STEP set, no PLAN_FILE, plus EITHER an explicit
#      (EXP_NAME, CKPT_PATH) pair OR the custom-sweep quad SIZE/FW_EDU_RATIO/
#      FW2_RATIO/SEED):
#        Convert ONE iter (torch_dist -> torch -> HF). Runs inside the
#        training container. The explicit (EXP_NAME, CKPT_PATH) form is what
#        the MODEL plan-line / `--models` launcher uses, so any model in
#        configs/models.json (e.g. the a06 main runs) converts the same way.
#
#   2. SBATCH WRAPPER (env: SLURM_JOB_ID and PLAN_FILE both set):
#        Inside an sbatch job, srun the container, loop over plan-file lines,
#        invoke this script in per-iter mode for each (cell, iter). Plan
#        lines come in two formats (see Mode 2 below).
#
#   3. LAUNCHER (no slurm, no per-iter env):
#        --models <names>: models.json-driven — resolve each model's megatron
#          checkpoint dir from configs/models.json `backends.megatron`, write a
#          plan of MODEL lines, (with --submit) sbatch one job. Handles any
#          model in models.json, including non-custom ones like a06.
#        otherwise: walk the 36 custom cells under $ROOT, list each cell's
#          valid canonical iters, write per-size plan files, and (with
#          --submit) sbatch one job per size.
#
# Examples:
#   bash convert-snr.sh                              # dry-run launcher (all sizes)
#   bash convert-snr.sh --submit --sizes 175M       # submit just the 175M sweep
#   bash convert-snr.sh --submit --partition normal --time 06:00:00
#   bash convert-snr.sh --models apertus3-1b-21-nodes --iters 20000 --submit
#   SIZE=175M FW_EDU_RATIO=30 FW2_RATIO=70 SEED=1904 CKPT_STEP=50000 bash convert-snr.sh
#
# Per-iter env (optional):
#   MEGATRON_LM_DIR     defaults /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM
#   PRETRAIN_LOGS_DIR   defaults $MEGATRON_LM_DIR/logs/Meg-Runs/data-mix-small
#   STAGING_BASE        defaults /iopsstor/scratch/cscs/$USER/snr-hf-checkpoints
#   HF_TOKENIZER        defaults alehc/swissai-tokenizer
#   EXP_NAME, CKPT_PATH explicit checkpoint dir + name (overrides the
#                       SIZE/FW_EDU_RATIO/FW2_RATIO/SEED reconstruction)
#   TEST_LOGITS=1       enable --test-logits
#   SKIP_PIP=1          skip pip install transformers (already in env)
#   KEEP_TMP_TORCH=1    keep intermediate torch ckpt after HF save

set -euo pipefail
SCRIPT_PATH="$(realpath "$0")"

# ============================================================================
# Mode 1: per-iter conversion (auto-detected via per-iter env vars).
# Fires when CKPT_STEP is set, no PLAN_FILE, and the checkpoint location is
# known either explicitly (EXP_NAME + CKPT_PATH) or via the custom-sweep quad.
# ============================================================================
if [[ -n "${CKPT_STEP:-}" && -z "${PLAN_FILE:-}" ]] && \
   { { [[ -n "${EXP_NAME:-}" && -n "${CKPT_PATH:-}" ]]; } || \
     { [[ -n "${SIZE:-}" && -n "${FW_EDU_RATIO:-}" \
          && -n "${FW2_RATIO:-}" && -n "${SEED:-}" ]]; }; }; then

    MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM}"
    PRETRAIN_LOGS_DIR="${PRETRAIN_LOGS_DIR:-$MEGATRON_LM_DIR/logs/Meg-Runs/data-mix-small}"
    STAGING_BASE="${STAGING_BASE:-/iopsstor/scratch/cscs/$USER/snr-hf-checkpoints}"
    TMP_TORCH_BASE="${TMP_TORCH_BASE:-$STAGING_BASE/_tmp_torch}"
    HF_TOKENIZER="${HF_TOKENIZER:-alehc/swissai-tokenizer}"

    # Checkpoint location: an explicit (EXP_NAME, CKPT_PATH) pair — e.g. from a
    # models.json-driven MODEL plan line — takes precedence; otherwise
    # reconstruct the custom-sweep layout from SIZE/FW_EDU_RATIO/FW2_RATIO/SEED.
    if [[ -n "${EXP_NAME:-}" && -n "${CKPT_PATH:-}" ]]; then
        CKPT_PATH="${CKPT_PATH%/}"
    else
        EXP_NAME="apertus-${SIZE}-fwEdu${FW_EDU_RATIO}-fw2${FW2_RATIO}-seed${SEED}"
        CKPT_PATH="$PRETRAIN_LOGS_DIR/$EXP_NAME/checkpoints"
    fi
    ITER_DIR_NAME=$(printf "iter_%07d" "$CKPT_STEP")
    SRC_ITER_DIR="$CKPT_PATH/$ITER_DIR_NAME"

    SAVE_DIR="$STAGING_BASE/$EXP_NAME/$ITER_DIR_NAME"

    # Completed output short-circuits everything below: a finished snapshot
    # needs no source, and the Megatron iter may legitimately be gone by now
    # (iopsstor scratch is auto-purged) — a manual Mode-1 run over such an
    # iter is the marker-backfill recovery path.
    if [[ -f "$SAVE_DIR/.hf_complete" ]]; then
        echo "[convert-snr] SKIP $EXP_NAME iter $CKPT_STEP: already converted (.hf_complete)"
        exit 0
    fi
    if [[ -f "$SAVE_DIR/config.json" ]] && \
       ( [[ -f "$SAVE_DIR/model.safetensors.index.json" ]] || [[ -f "$SAVE_DIR/model.safetensors" ]] ); then
        # Backfill the completion marker for snapshots converted before it
        # existed — but only after validating the weights: a job killed
        # mid-save_pretrained leaves config.json + a truncated (in-place,
        # unsharded at these sizes) model.safetensors, and stamping that
        # would poison the snapshot forever. safe_open rejects a truncated
        # file (header/offset checks). Invalid (rc=1) -> fall through and
        # re-convert (save_pretrained overwrites in place); rc>=2 = the
        # validator couldn't run — don't guess either way.
        rc=0
        python3 - "$SAVE_DIR" <<'VEOF' || rc=$?
import json, sys
from pathlib import Path
try:
    from safetensors import safe_open
except ImportError as e:
    print(f"safetensors unavailable: {e}", file=sys.stderr)
    sys.exit(3)
d = Path(sys.argv[1])
idx = d / "model.safetensors.index.json"
files = ({d / f for f in json.loads(idx.read_text())["weight_map"].values()}
         if idx.is_file() else {d / "model.safetensors"})
try:
    for f in files:
        with safe_open(f, framework="pt"):
            pass
except ImportError as e:
    print(f"cannot validate ({e})", file=sys.stderr)
    sys.exit(3)
except Exception as e:
    sys.exit(f"invalid weights: {e}")
VEOF
        if (( rc == 0 )); then
            touch "$SAVE_DIR/.hf_complete"
            echo "[convert-snr] SKIP $EXP_NAME iter $CKPT_STEP: SAVE_DIR already populated"
            exit 0
        elif (( rc >= 2 )); then
            echo "[convert-snr] ERROR: weight validation could not run (rc=$rc)" >&2
            exit "$rc"
        fi
        echo "[convert-snr] $EXP_NAME iter $CKPT_STEP: SAVE_DIR populated but weights invalid — re-converting"
    fi

    if [[ ! -d "$SRC_ITER_DIR" ]]; then
        echo "[convert-snr] ERROR: source iter dir not found: $SRC_ITER_DIR" >&2
        exit 2
    fi
    # Skip empty/cleaned iter dirs (the May-3 sweep emptied non-canonical iters
    # — and many seed1797/seed28 canonical iters too). Delegates to the single
    # source-of-truth helper in pretrain_progress.py (--is-valid CLI). python3
    # (not python3.11) because we're inside the eval container.
    # Inside sbatch context $0 (and thus SCRIPT_PATH) is the slurm_script copy
    # under /var/spool/slurmd/, so the script-relative path doesn't resolve.
    # Prefer env override (set by Mode 2 INNER_EXPORTS), then script-relative
    # (works for standalone direct invocations), with an absolute fallback.
    PROGRESS_PY="${PROGRESS_PY:-$(dirname "$SCRIPT_PATH")/../pretrain_progress.py}"
    [[ -f "$PROGRESS_PY" ]] || \
        PROGRESS_PY="/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/pretrain_progress.py"
    rc=0
    python3 "$PROGRESS_PY" --is-valid "$SRC_ITER_DIR" || rc=$?
    if (( rc >= 2 )); then
        # rc=1 means "invalid checkpoint"; rc>=2 means the helper itself
        # didn't run (path doesn't resolve, python can't open it) — treating
        # that as SKIP would silently no-op the whole sweep.
        echo "[convert-snr] ERROR: --is-valid helper failed (rc=$rc, PROGRESS_PY=$PROGRESS_PY)" >&2
        exit "$rc"
    elif (( rc != 0 )); then
        echo "[convert-snr] SKIP: $SRC_ITER_DIR is not a valid checkpoint"
        exit 0
    fi

    TORCH_CKPT_SAVE_PATH="$TMP_TORCH_BASE/$EXP_NAME/$ITER_DIR_NAME"
    mkdir -p "$TORCH_CKPT_SAVE_PATH" "$(dirname "$SAVE_DIR")"

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
    # Written LAST: the watchers treat a snapshot as staged only once this
    # exists, so a half-written save_pretrained is never evaluated.
    touch "$SAVE_DIR/.hf_complete"
    echo "[convert-snr] DONE $SAVE_DIR"
    exit 0
fi

# ============================================================================
# Mode 2: sbatch wrapper (running in slurm + PLAN_FILE) — srun container,
# loop the plan
# ============================================================================
if [[ -n "${SLURM_JOB_ID:-}" && -n "${PLAN_FILE:-}" ]]; then
    [[ -f "$PLAN_FILE" ]] || { echo "[convert-snr/sbatch] PLAN_FILE not found: $PLAN_FILE" >&2; exit 2; }

    MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM}"
    CONTAINER_TOML="${CONTAINER_TOML:-/capstor/store/cscs/swissai/a139/containers/ngc_25-11-nemo-alps3.toml}"
    mkdir -p /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/conversion

    echo "[$(date)] convert-snr/sbatch starting, plan=$PLAN_FILE"
    cat "$PLAN_FILE"
    echo "---"

    # Forward HF cache vars + force offline so per-iter tokenizer loads hit local
    # cache (the saver and Megatron's HuggingFaceTokenizer both fetch the
    # tokenizer repo, and the `/api/models/` freshness check will burn the
    # team-plan rate-limit on a 9-cell × ~13-iter sweep). HF_TOKENIZER is
    # forwarded so predictivity conversions embed the sweep's tokenizer
    # (swiss-ai/Apertus-70B-2509) instead of the old-sweep default —
    # pre-warm it into the cache first (offline mode can't download it).
    # PROGRESS_PY and TMP_TORCH_BASE must cross the pyxis boundary too:
    # without them the per-iter mode falls back to a hardcoded checkout path
    # for --is-valid (silently SKIPping every iter when it doesn't resolve)
    # and to $STAGING_BASE/_tmp_torch for the multi-GB torch intermediate
    # (churning capstor now that STAGING_BASE points at the store).
    # The two HF paths are hard-set, not ${VAR:-...}: --export=ALL plus the
    # ~/.bashrc HF_HOME=.../$USER/hf_home means a default loses to whichever
    # user submitted, and everything here runs HF_HUB_OFFLINE (bug 12) against
    # the tokenizer pre-cached in mariagrandury's hf_models.
    INNER_EXPORTS="export MEGATRON_LM_DIR='$MEGATRON_LM_DIR' PLAN_FILE='$PLAN_FILE' \
HF_TOKENIZER='${HF_TOKENIZER:-alehc/swissai-tokenizer}' \
STAGING_BASE='${STAGING_BASE:-/iopsstor/scratch/cscs/$USER/snr-hf-checkpoints}' \
TMP_TORCH_BASE='${TMP_TORCH_BASE:-/iopsstor/scratch/cscs/$USER/snr-hf-checkpoints/_tmp_torch}' \
PROGRESS_PY='${PROGRESS_PY:-}' \
HF_HOME='/iopsstor/scratch/cscs/mariagrandury/hf_home' \
HF_HUB_CACHE='/capstor/store/cscs/swissai/infra01/users/mariagrandury/hf_models' \
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
    # Two plan-line formats:
    #   MODEL <exp_name> <ckpt_path> ITER...   (models.json-driven, any model)
    #   <SIZE> <FW_EDU> <FW2> <SEED> ITER...   (custom-sweep layout)
    if [[ \"\${parts[0]}\" == MODEL ]]; then
        export EXP_NAME=\${parts[1]} CKPT_PATH=\${parts[2]}
        unset SIZE FW_EDU_RATIO FW2_RATIO SEED
        label=\"\$EXP_NAME\"; iter_start=3
    else
        export SIZE=\${parts[0]} FW_EDU_RATIO=\${parts[1]} FW2_RATIO=\${parts[2]} SEED=\${parts[3]}
        unset EXP_NAME CKPT_PATH
        label=\"\$SIZE fwEdu\$FW_EDU_RATIO seed\$SEED\"; iter_start=4
    fi
    for it in \"\${parts[@]:\$iter_start}\"; do
        export CKPT_STEP=\"\$it\"
        echo \"[\$(date)] === \$label iter \$it ===\"
        if bash $SCRIPT_PATH; then
            total_ok=\$((total_ok+1))
        else
            rc=\$?
            echo \"[\$(date)] FAILED rc=\$rc for \$label iter \$it\"
            total_fail=\$((total_fail+1))
            failed_list+=(\"\$label/iter\$it\")
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
    # SCRIPT_PATH inside sbatch is the slurm_script copy; same fallback as Mode 1.
    PRETRAIN_PROGRESS="$(dirname "$SCRIPT_PATH")/../pretrain_progress.py"
    [[ -f "$PRETRAIN_PROGRESS" ]] || \
        PRETRAIN_PROGRESS="/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/pretrain_progress.py"
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
ROOT="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small"
PLAN_DIR="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/conversion-plans"
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
MODELS=""
ITERS=""
SUBMIT=0
PARTITION=""
TIME=""
RESERVATION=""
ALL_ITERS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --sizes) shift; while [[ $# -gt 0 && "$1" != --* ]]; do SIZES+=("$1"); shift; done ;;
    --models) MODELS="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --all-iters) ALL_ITERS=1; shift ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --reservation) RESERVATION="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --plan-dir) PLAN_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '1,48p' "$SCRIPT_PATH"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ ${#SIZES[@]} -eq 0 ]] && SIZES=("${DEFAULT_SIZES[@]}")
mkdir -p "$PLAN_DIR"
TS=$(date +%Y%m%d-%H%M%S)

# Resolve the --is-valid helper from THIS checkout while $0 is still the
# real script path (inside sbatch it's the /var/spool/slurmd copy);
# --export=ALL carries it into the job env, INNER_EXPORTS into the
# container. Common to BOTH launcher branches — without it every iter of a
# submitted job silently SKIPs when the hardcoded fallback doesn't resolve.
export PROGRESS_PY="${PROGRESS_PY:-$(dirname "$SCRIPT_PATH")/../pretrain_progress.py}"

# --models <name,name,...>: models.json-driven plan. Resolve each model's
# megatron checkpoint dir from configs/models.json `backends.megatron`, keep
# the iters that are valid on disk (.metadata + >=1 .distcp shard), and write
# one MODEL line per model. --iters <n,n,...> restricts to specific iters
# (otherwise the model's `checkpoints.all` from models.json is used). This is
# the path for non-custom-sweep models such as the a06 main runs.
if [[ -n "$MODELS" ]]; then
    SRC_DIR=$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)   # .../src
    # Include the models list in the filename so simultaneous submissions for
    # different models don't collide on the plan path (which is captured in
    # the sbatch --export and read by the job at run time).
    _models_slug=${MODELS//,/_}
    plan="$PLAN_DIR/plan-models-${_models_slug}-${TS}.txt"
    python3.11 - "$SRC_DIR" "$MODELS" "$ITERS" > "$plan" <<'PYEOF'
import os, sys
from pathlib import Path
src_dir, models_arg, iters_arg = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(src_dir, "evals", "scripts", "utils"))
sys.path.insert(0, os.path.join(src_dir, "pretrain"))
from configs import get_model, iters_for
from pretrain_progress import is_valid_iter_dir   # canonical validity helper

print("# Auto-generated by convert-snr.sh --models")
print("# Format: MODEL <exp_name> <ckpt_path> ITER1 ITER2 ...")
want = [int(x) for x in iters_arg.split(",")] if iters_arg else None
for model in (m.strip() for m in models_arg.split(",")):
    if not model:
        continue
    ckpt = (get_model(model).get("backends") or {}).get("megatron")
    if not ckpt:
        print(f"# SKIP {model} (no megatron backend in models.json)", file=sys.stderr)
        continue
    ckpt = ckpt.rstrip("/")
    iters = want if want is not None else iters_for(model, subset="all")
    valid = [it for it in iters if is_valid_iter_dir(Path(f"{ckpt}/iter_{it:07d}"))]
    if not valid:
        print(f"# SKIP {model} (no valid iters on disk among {iters})", file=sys.stderr)
        continue
    print(f"MODEL {model} {ckpt} " + " ".join(str(i) for i in valid))
PYEOF

    line_count=$(grep -vc '^#' "$plan" || true)
    echo "=== models plan ($line_count model line(s)): $plan ==="
    cat "$plan"
    if [[ $line_count -eq 0 ]]; then
        echo "[convert-snr/launcher] --models: nothing to convert, skipping"
        exit 0
    fi

    part="${PARTITION:-normal}"
    walltime="${TIME:-02:00:00}"
    sbatch_args=(--partition="$part" --time="$walltime"
                 --job-name="convert-snr-models"
                 --export=ALL,PLAN_FILE="$plan")
    [[ -n "$RESERVATION" ]] && sbatch_args+=(--reservation="$RESERVATION")

    if [[ $SUBMIT -eq 1 ]]; then
        sbatch "${sbatch_args[@]}" "$SCRIPT_PATH"
    else
        echo "[convert-snr/launcher] --models: would sbatch ${sbatch_args[*]} $SCRIPT_PATH"
        echo "  pass --submit to actually submit"
    fi
    exit 0
fi

valid_iters_for_cell() {
    local cell=$1 seed=$2 d out=() iters it
    # Validity check delegates to the canonical helper in pretrain_progress.py
    # (--is-valid CLI). Mode 3 runs on the login node → python3.11.
    local progress_py="$(dirname "$SCRIPT_PATH")/../pretrain_progress.py"
    if [[ $ALL_ITERS -eq 1 ]]; then
        # Every iter dir on disk that lands on a 2000-step boundary.
        # Excludes SIGUSR2-triggered intermediate exit dirs (e.g. iter_0049180,
        # iter_0032619) — those are training-grace artifacts, not canonical
        # eval/inspection points. --is-valid filters async-save shells.
        for d in "$ROOT/$cell/checkpoints"/iter_*/; do
            [[ -d "$d" ]] || continue
            it=$(basename "$d"); it=${it#iter_}; it=$((10#$it))
            (( it % 2000 == 0 )) || continue
            python3.11 "$progress_py" --is-valid "$ROOT/$cell/checkpoints/iter_$(printf '%07d' "$it")" 2>/dev/null && out+=("$it")
        done
    else
        if [[ "$seed" == "1904" ]]; then
            iters=("${ITERS_SEED1904[@]}")
        else
            iters=("${ITERS_OTHER[@]}")
        fi
        for it in "${iters[@]}"; do
            d="$ROOT/$cell/checkpoints/iter_$(printf '%07d' "$it")"
            python3.11 "$progress_py" --is-valid "$d" 2>/dev/null && out+=("$it")
        done
    fi
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
