#!/bin/bash
# Is every data build alive, finished, or stopped? One line per build.
#
# data_progress.py answers "which languages, how many tokens" — it is a
# coverage heatmap and it has nothing to say about an English build, which has
# no per-language structure. This answers the other question, the operational
# one: did the chain die, and do I resubmit?
#
#   bash build_status.sh
#
# States, and what to do about each:
#   DONE      built and staged. For an English build it also checks the
#             realized size against the largest run's draw — NOTHING else does.
#             fineweb_source() short-circuits at L=1, so undersized_build never
#             runs on an English build and a short one repeats silently.
#   building  a job holds it; the token count is the .bin as it grows.
#   STALLED   incomplete and no job left. Prints the one sbatch to resubmit,
#             fully expanded: an unset BUILD_OUT once sent 23 attempts to /BT3.
#   pending   nothing built yet and a job is queued (it has not started).
#   -         not built and not queued; nothing has ever asked for it.
#
# The 92B/165B rebuild roots are deliberately out of scope: they are separate
# roots with their own job names (-92b/-165b) and their own stage trees.
set -uo pipefail
REPO=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
OUT=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data
STAGE=/iopsstor/scratch/cscs/mariagrandury/data

# The registry owns which builds exist and how big each one is meant to be —
# read it rather than restating it, exactly as launch_builds.sh does.
PLAN=$(python3.11 - <<'PY'
import sys
sys.path.insert(0, "/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain")
sys.path.insert(0, "/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data")
from launch_trainings import DATA_SCHEMES, SEQ_LEN, scheme_sizes
from build_data_mixtures import (SIZE_BUDGET_B, english_target_tokens,
                                 fineweb_target_tokens)

# stage : scheme : subdir : name : target tokens : what the largest run draws
for name, v in DATA_SCHEMES.items():
    sub = v["subdir"]
    if name == "A" or v.get("english"):          # A's is the shared build
        draw = int(SIZE_BUDGET_B[scheme_sizes(name, 1)[-1]] * 1e9) if 1 in v["langs"] else 0
        print(f"english:{name}:{sub}:english_dclm:{english_target_tokens()}:{draw}")
    for L in sorted(v["langs"]):
        if L == 1:
            continue
        draw = int(SIZE_BUDGET_B[scheme_sizes(name, L)[-1]] * 1e9) // 2   # 50% ml half
        print(f"fineweb:{name}:{sub}:fineweb_L{L}:{fineweb_target_tokens(L, name)}:{draw}")
PY
) || { echo "cannot read the scheme registry" >&2; exit 1; }

printf '%-22s %-9s %s\n' BUILD STATE DETAIL
while IFS=: read -r stage scheme sub name target draw; do
  pre=$OUT${sub:+/$sub}/$name
  if [ "$stage" = english ]; then
    job=build-en${sub:+-$(echo "$scheme" | tr 'A-Z' 'a-z')}
  else
    job=build-$(echo "$scheme" | tr 'A-Z' 'a-z')-${name#fineweb_}
  fi
  n=$(squeue -u "$USER" -h -n "$job" 2>/dev/null | wc -l)
  tok=0; [ -f "$pre.bin" ] && tok=$(( $(stat -Lc%s "$pre.bin") / 4 ))
  label=$scheme/$name

  if [ -f "$pre.idx" ] && [ ! -f "$pre.checkpoint.json" ]; then
    st=$([ -f "$STAGE${sub:+/$sub}/$name.idx" ] && echo staged || echo "NOT staged")
    # Only an English build needs this said out loud; a fineweb one is checked
    # by undersized_build at launch time.
    ep=""
    if [ "$stage" = english ] && [ "$draw" -gt 0 ]; then
      ep=$(awk -v t="$tok" -v d="$draw" 'BEGIN{printf "%.2f epochs", d/t}')
      [ "$tok" -lt "$draw" ] && ep="$ep — TOO SMALL, the largest run repeats"
    fi
    printf '%-22s %-9s %s\n' "$label" DONE \
      "$(awk -v t="$tok" 'BEGIN{printf "%.1fB tokens", t/1e9}'), $st${ep:+, $ep}"
  elif [ "$n" -gt 0 ] && [ "$tok" -gt 0 ]; then
    printf '%-22s %-9s %s\n' "$label" building \
      "$(awk -v t="$tok" -v g="$target" 'BEGIN{printf "%.1fB / %.1fB (%d%%)", t/1e9, g/1e9, 100*t/g}'), $n job(s)"
  elif [ "$n" -gt 0 ]; then
    printf '%-22s %-9s %s\n' "$label" pending "$n job(s) queued, nothing written yet"
  elif [ "$tok" -gt 0 ]; then
    printf '%-22s %-9s %s\n' "$label" STALLED \
      "$(awk -v t="$tok" 'BEGIN{printf "%.1fB", t/1e9}') written, NO job left — resubmit:"
    printf '    sbatch --job-name=%s --dependency=singleton --time=23:59:00 --exclusive \\\n' "$job"
    printf '      --partition=preemptable --export=ALL,BUILD_SCHEME=%s,BUILD_STAGE=%s%s,BUILD_OUT=%s \\\n' \
      "$scheme" "$stage" "$([ "$stage" = fineweb ] && echo ",BUILD_SETTING=${name#fineweb_L}")" "$OUT${sub:+/$sub}"
    printf '      %s/src/pretrain/data/submit_build_one.sh\n' "$REPO"
  else
    printf '%-22s %-9s %s\n' "$label" - "not built, nothing queued"
  fi
done <<< "$PLAN"
