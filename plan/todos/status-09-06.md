Evals:
- ✅ Remove afrimmmlu and afrixnli (for now) -> review list of languages and available benchmarks, bloks L100 analysis
- ✅ Job 3311744: eval-175M-L50-deep-seed1904-iter3416 COMPLETED with batch=1
- sbatch scripts/mirror_eval_logs.sbatch -> now it mirrors evals AND touches megatron ckpt files (Job ID 3312266)

BPB:
- ✅ bash evals/scripts/launch_bpb.sh --filter '1B'
- bash evals/scripts/launch_bpb.sh --filter 'deep' (1.7B + 90M L8)
- bash evals/scripts/launch_bpb.sh --filter '600M' (40 ckpts, all shallow)
- loss/BPB scaling fits

1B & 1.7B:
- ✅ convert new checkpoints: python3.11 pretrain/auto_evals_cscs.py --convert-only
- fit 1B and 1.7B estimates
- resume training of 1.7B models
- launch 1Bs again? -> maybe on Wednesday better

SNR:
- Update the SNR module to new naming
- Start writing on the ≤600M ladder. Signal, noise, decision accuracy and scaling-law error are all computable on 90M–600M × 6 language settings

L100:
- [Discuss] Update the list of available high-quality benchmarks for low resource languages.
- Launch the creation of the L100 data mixture with the new list of languages (1 day). Blocks pretraining of all L100 models. Blocked by update of available benchmarks for low resource languages.
- [Discuss] Update the model grid plan so each size-languages cell has at least 3 models (we need 3 to calculate DA).
- The 4 small sizes of L100 models (classic: deep, A).


Merge changes in evaluate.sbatch (if needed after other eval optimization fixes):

from

LM_EVAL_HARNESS_PIP_SPEC="git+https://github.com/swiss-ai/lm-evaluation-harness.git"
if [[ -n "$LM_EVAL_HARNESS_BRANCH" ]]; then
    LM_EVAL_HARNESS_PIP_SPEC="${LM_EVAL_HARNESS_PIP_SPEC}@${LM_EVAL_HARNESS_BRANCH}"
fi

to

# Install the harness from a SHARED CHECKOUT, not from GitHub.
#
# Every job used to run `pip install git+https://github.com/swiss-ai/...`,
# i.e. one fresh clone per job. That is fine for a handful of jobs and fails
# as a fleet: with ~120 evals queued on 2026-09-02 the partial clone's
# promisor fetch started coming back `HTTP 401`, pip aborted, and the job died
# in 35 s on `lm_eval: command not found` — 90 failures in a row, none of
# which reached a single dataset. The clone is also the slowest part of a
# short eval, and it silently tracked whatever HEAD was at submission time,
# so two checkpoints of the same cell could be scored by different harness
# versions. A pinned local checkout fixes all three.
#
# Refresh it deliberately (login node, then note the commit in the eval log):
#   git -C $HARNESS_SRC pull
HARNESS_SRC=${HARNESS_SRC:-/capstor/store/cscs/swissai/infra01/msnr-harness/lm-evaluation-harness}
# A PREBUILT WHEEL is the first choice, because `pip install <dir>` builds
# in-tree: setuptools writes build/ INSIDE the source directory, so every
# concurrent job shares one build tree and they delete each other's files
# mid-copy — `error: [Errno 2] No such file or directory`, then
# `lm_eval: command not found`. Two of six jobs died that way on 2026-09-03;
# the ones that ran alone were fine, which is exactly the signature of a race.
# Installing an artifact writes nothing shared and skips the ~15 min build.
#
# Rebuild it after refreshing the checkout (login node, minutes):
#   git -C $HARNESS_SRC pull
#   rsync -a --exclude .git --exclude build --exclude '*.egg-info' \
#         $HARNESS_SRC/ /iopsstor/scratch/cscs/$USER/tmp-harness-build/src/
#   pip wheel --no-deps --no-build-isolation \
#         -w $(dirname $HARNESS_SRC)/wheels /iopsstor/.../tmp-harness-build/src
HARNESS_WHEEL=$(ls -t "$(dirname "$HARNESS_SRC")"/wheels/lm_eval-*.whl 2>/dev/null | head -1)
if [[ -z "$LM_EVAL_HARNESS_BRANCH" && -n "$HARNESS_WHEEL" ]]; then
    LM_EVAL_HARNESS_PIP_SPEC="$HARNESS_WHEEL"
    echo "Harness: $(basename "$HARNESS_WHEEL") (prebuilt wheel)"
elif [[ -z "$LM_EVAL_HARNESS_BRANCH" && -d "$HARNESS_SRC" ]]; then
    # No wheel yet: build from the checkout. Correct, but serialize the jobs —
    # see the race above.
    LM_EVAL_HARNESS_PIP_SPEC="$HARNESS_SRC"
    echo "Harness: $HARNESS_SRC @ $(git -C "$HARNESS_SRC" rev-parse --short HEAD 2>/dev/null || echo '?') (in-tree build — no wheel found)"
else
    # Explicit branch, or the shared checkout is missing: fall back to GitHub.
    LM_EVAL_HARNESS_PIP_SPEC="git+https://github.com/swiss-ai/lm-evaluation-harness.git"
    if [[ -n "$LM_EVAL_HARNESS_BRANCH" ]]; then
        LM_EVAL_HARNESS_PIP_SPEC="${LM_EVAL_HARNESS_PIP_SPEC}@${LM_EVAL_HARNESS_BRANCH}"
    fi
    echo "Harness: cloning $LM_EVAL_HARNESS_PIP_SPEC (no shared checkout at $HARNESS_SRC)"
fi