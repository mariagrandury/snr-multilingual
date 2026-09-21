#!/usr/bin/env bash
# Bring every derived artefact up to date with the published ladder report.
#
# Nothing under documents/ is hand-maintained except prose: the CSVs, the
# figures, the deck's generated blocks, the report PDF and the compendium all
# derive from one file, and this script regenerates the lot in dependency
# order. Run it whenever new eval results land; a plot or table can then never
# be older than the data.
#
#   bash scripts/refresh_analysis.sh              # fetch the report, then rebuild
#   bash scripts/refresh_analysis.sh --no-fetch   # rebuild from the local copy
#   bash scripts/refresh_analysis.sh --no-deck    # skip the slidev build
#   bash scripts/refresh_analysis.sh --curves     # also redraw rq00's ~140
#                                                 # acc-vs-FLOPs grids (+1 h)
#
# Two things this cannot guess, both of which silently produce stale numbers:
#
#   FORCE=1 — the fetched report carries its COMMIT time, so a report
#   published at noon and an analysis re-run that evening leave every cached
#   table "newer than the report" and the pipeline reuses them. Pass FORCE=1
#   whenever the report was regenerated since the last analysis run, which is
#   the normal case; without it only the figures downstream are redrawn.
#
#   The run takes hours, so it belongs in a Slurm allocation, not on the
#   login node (src/pretrain/CLAUDE.md).
#
# Prose is the one thing it cannot fix. The last step writes
# documents/ladder-facts.json and prints every headline number that moved, so
# the sentences quoting them can be found instead of quietly going stale.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
PY=${PY:-python3}
# Fixed PDF creation date: an unchanged figure re-renders to the same bytes,
# so git sees no diff (PNGs are already deterministic).
export SOURCE_DATE_EPOCH=0
FETCH=1; DECK=1
CURVES=${CURVES:-0}
for a in "$@"; do
  case "$a" in
    --no-fetch) FETCH=0 ;;
    --no-deck)  DECK=0 ;;
    --curves)   CURVES=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done
FAILED=()
step() { echo; echo "=== $* ==="; }
run()  { "$@"; [ $? -eq 0 ] || FAILED+=("$*"); }

# 1. The ladder report. It is published to an orphan branch of this repo as
#    well as to the Hub, and the branch is what a fresh clone can reach.
if [ "$FETCH" = 1 ]; then
  step "fetch the ladder report"
  if git fetch origin data/ladder-report 2>&1 | tail -2; then
    for d in "$REPO/src/signal-and-noise/data/ladder-report" "$REPO/data/ladder-report"; do
      mkdir -p "$d" && git archive origin/data/ladder-report | tar -x -C "$d"
    done
    echo "installed $(git log -1 --format='%h %s' origin/data/ladder-report)"
  else
    echo "WARNING: fetch failed, continuing with the local copy" >&2
  fi
fi
LADDER=$REPO/src/signal-and-noise/data/ladder-report/ladder_report.csv
[ -f "$LADDER" ] || { echo "no ladder report at $LADDER" >&2; exit 1; }
echo "report: $(wc -l < "$LADDER") rows, $(date -r "$LADDER" '+%Y-%m-%d %H:%M')"

# 2. The analysis. Its own cache re-runs whatever is older than the report, so
#    a refreshed report invalidates every table below it.
step "analysis pipeline"
( cd src/signal-and-noise && HF_HUB_OFFLINE=1 CURVES=$CURVES bash run_all_predictivity.sh ) \
  || FAILED+=("run_all_predictivity.sh")

# 2b. The paper's figures: every one is written by an rqNN script above and
#     only copied here, so the paper can never be newer than the tables.
step "paper figures"
( cd documents/paper/figures && $PY make_rq_figures.py ) || FAILED+=("make_rq_figures.py")

# 2c. The paper's audit tables: a reshape of the same rqNN tables, so the
#     numbers section 4 quotes cannot drift from the ones the figures show.
step "paper tables"
( cd documents/paper/sections && $PY verify_paper_results.py ) || FAILED+=("verify_paper_results.py")

# 3. The deck figures, then the two report figures the deck reuses.
step "figures"
for f in fig_setup fig_languages fig_benchmarks fig_predictivity fig_rq6_sketch \
         fig_benchmark_predictivity fig_rqs fig_appendix fig_from_analysis; do
  echo "-- $f"
  ( cd documents/figures && HF_HUB_OFFLINE=1 $PY "$f.py" ) 2>&1 | tail -3
  [ ${PIPESTATUS[0]} -eq 0 ] || FAILED+=("$f.py")
done

# 3b. The scaling-fit panel. ladder_report.py draws it on the cluster as part of
#     --plot, but this one reads the published CSV alone, so it refreshes here.
step "scaling figure"
$PY - <<'EOF' || FAILED+=("plot_scaling")
import sys
from pathlib import Path
R = Path.cwd()
for d in (R / "src", R / "src" / "pretrain", R / "src" / "signal-and-noise"):
    sys.path.insert(0, str(d))
from snr.download.ladder import ladder_dir
import ladder_report as LR
print("wrote", LR.plot_scaling(ladder_dir() / "ladder_report.csv",
                               R / "documents" / "public" / "ladder"))
EOF

# 4. The report PDF.
step "report pdf"
run $PY documents/build_report.py

# 5. The compendium. Its source keeps relative paths so the diff stays
#    readable; the published copy needs them inlined, because an artifact can
#    load images only from a data URI.
step "compendium"
run $PY scripts/inline_artifact.py

# 6. Checks that catch a stale or missing figure before anyone presents it.
step "checks"
$PY - "$LADDER" <<'EOF' || FAILED+=("figure check")
import re, pathlib, sys
root = pathlib.Path("documents")
pub = root / "public"
report_mtime = pathlib.Path(sys.argv[1]).stat().st_mtime

slides = (root / "slides.md").read_text()
html = (root / "research-questions.html").read_text()
md = (root / "report" / "snr_predictivity_report.md").read_text()
refs = ({m.group(1).lstrip("/") for m in re.finditer(r"^image:\s*(\S+)", slides, flags=re.M)}
        | {p[len("public/"):] for p in re.findall(r'src="(public/[^"]+)"', html)}
        | {m.rsplit("/ladder/", 1)[1] and "ladder/" + m.rsplit("/ladder/", 1)[1]
           for m in re.findall(r"\]\((\.\./public/ladder/[^)]+)\)", md)})

# Drawn on the cluster from the training logs, which this machine cannot read.
CLUSTER = {"ladder/ladder_report_loss.png", "ladder/pretrain_progress_plan.png",
           "ladder/pretrain_progress_simple.png", "ladder/pretrain_progress_detailed.png"}
# Only public/ladder/ holds figures derived from the report. Anything else is a
# static asset (a paper scan, a screenshot) and has no data to go stale against.
derived = {r for r in refs if r.startswith("ladder/")}
miss = sorted(r for r in refs if not (pub / r).is_file())
stale = sorted(r for r in derived - CLUSTER if (pub / r).is_file()
               and (pub / r).stat().st_mtime < report_mtime)
cluster = sorted(r for r in derived & CLUSTER if (pub / r).is_file()
                 and (pub / r).stat().st_mtime < report_mtime)
used = {"/" + r for r in refs}
allp = {f"/ladder/{q.relative_to(pub / 'ladder')}" for q in (pub / "ladder").rglob("*.png")}
print(f"slides: {len(re.split(r'^---$', slides, flags=re.M))} chunks; "
      f"{len(refs)} figures referenced ({len(derived)} derived from the report), "
      f"{len(miss)} missing, {len(stale)} older than the report, "
      f"{len(allp - used)} unreferenced")
for m in miss:
    print("  MISSING", m)
for m in stale:
    print("  STALE  ", m, "- no generator ran for it in this refresh")
for m in cluster:
    print("  CLUSTER", m, "- redraw with ladder_report.py --plot / pretrain_progress.py --plot")
sys.exit(1 if miss or stale else 0)
EOF

if [ "$DECK" = 1 ]; then
  step "slidev build"
  # The grep keeps the output to one line, but it also swallowed the reason the
  # step failed: on a node without npx the only message is "npx: command not
  # found", which matches neither pattern, so the step failed in silence. Say
  # what is missing, and treat "no node here" as a skip rather than a failure —
  # the deck builds where node is installed, and --no-deck silences the notice.
  if ! command -v npx >/dev/null 2>&1; then
    echo "  npx not found on this node — deck not built (install node, or pass --no-deck)"
  else
    ( cd documents && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npx slidev build --out /tmp/slidev-refresh \
        2>&1 | grep -E "✓ built|error|not found|No such file" ) || FAILED+=("slidev build")
  fi
fi

# 7. The numbers the prose quotes.
step "headline numbers"
( cd documents/figures && HF_HUB_OFFLINE=1 $PY facts.py ) || FAILED+=("facts.py")

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "############ FAILED ############"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "############ ALL REFRESHED ############"
