#!/bin/bash
# score_bpb.sbatch — per-language BPB + perplexity for every due checkpoint of
# one cell, the predictivity study's outcome metric (see scripts/score_bpb.py).
#
# One job per CELL, not per checkpoint: the model load is seconds against
# minutes of scoring, so paying the container start and pip install once for
# ~10 checkpoints is most of the wall-clock saving available here.
#
#   sbatch --job-name=bpb-<cell> scripts/score_bpb.sbatch <cell> [iters...]
#   sbatch scripts/score_bpb.sbatch lm-90M-L2-deep-seed1904 450 900
#
# With no iters it scores every checkpoint converted for the cell.
#
#SBATCH --account=infra01
#SBATCH --job-name=score-bpb
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200000
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail

CELL=${1:?usage: sbatch score_bpb.sbatch <cell> [iters...]}
shift || true
ITERS=("$@")

REPO_DIR=$PWD
STAGING=${STAGING:-/capstor/store/cscs/swissai/infra01/msnr-hf-models}
# Results live beside the harness results, under the same cell-iter name, so
# the eval tooling and the analysis find them in one place.
OUT_ROOT=${OUT_ROOT:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/msnr}
DATA_DIR=${DATA_DIR:-/iopsstor/scratch/cscs/mariagrandury/data}
MAX_TOKENS=${MAX_TOKENS:-1000000}

if [ ${#ITERS[@]} -eq 0 ]; then
    # .hf_complete is convert-snr.sh's last write, so its presence means the
    # snapshot is whole — config.json alone can match a half-written save.
    mapfile -t ITERS < <(find "$STAGING/$CELL" -name .hf_complete -printf '%h\n' 2>/dev/null \
        | xargs -r -n1 basename | sed 's/^iter_0*//' | sort -n)
fi
echo "cell=$CELL  iters=${ITERS[*]:-none}  max_tokens=$MAX_TOKENS"
[ ${#ITERS[@]} -eq 0 ] && { echo "no converted checkpoints for $CELL"; exit 1; }

# Same container as the harness: it already carries torch + transformers, and
# reusing it keeps the numerics identical to what lm_eval sees.
srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml \
     --export=ALL,HF_TOKEN,HF_HOME,HF_HUB_CACHE bash -lc "
    cd '$REPO_DIR'
    set -uo pipefail
    unset SSL_CERT_FILE
    export HF_HOME='${HF_HOME:-/iopsstor/scratch/cscs/mariagrandury/hf_home}'
    # Compute nodes have no internet. Without these, from_pretrained on a LOCAL
    # path still calls the Hub to check for a newer revision and blocks on the
    # connect timeout — minutes per checkpoint, for nothing (evals CLAUDE.md
    # bug 12, the same fix convert-snr.sh needed).
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
    rc=0
    for it in ${ITERS[*]}; do
        padded=\$(printf 'iter_%07d' \$it)
        model='$STAGING/$CELL'/\$padded
        dest='$OUT_ROOT'/'$CELL'-iter\$it/bpb
        mkdir -p \"\$dest\"
        # Idempotent, like every other stage: a re-run after a walltime kill
        # skips what already landed instead of rescoring it.
        if [ -s \"\$dest/bpb.json\" ]; then
            echo \"== \$padded already scored, skipping\"; continue
        fi
        echo \"== scoring \$padded\"
        python3 scripts/score_bpb.py --model \"\$model\" \
            --data-dir '$DATA_DIR' --max-tokens $MAX_TOKENS \
            --out \"\$dest/bpb.json\" || rc=1
    done
    exit \$rc
"
echo "END: $(date)"
