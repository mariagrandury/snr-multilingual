#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=refetch-fineweb
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --output=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/logs/%x-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/logs/%x-%j.out
#SBATCH --no-requeue

# Re-fetch the TRUNCATED parquet files of the shared FineWeb mirror.
#
# Why this exists (2026-09-23). The FWEB English build stalled at file 54 with
#   pyarrow.lib.ArrowInvalid: Parquet magic bytes not found in footer
# and then burned its whole 25-attempt chain budget failing the same way in
# ~60s a time. The cause is not the build: the shared mirror at
# $FINEWEB is incomplete for the 2019+ crawls. Of the first 450 files in the
# build's read order, 2013-2018 are 100% intact and 2019-2022 are 15-38%
# intact, and every bad file's size is an exact MiB multiple — an interrupted
# parallel download, not disk rot.
#
# Skipping the bad files was the alternative, and it is worse: FWEB reads
# `crawls <= 2022` precisely so it matches DCLM's Common Crawl window and
# CORPUS is the only difference from DCLMP. Dropping the unreadable files
# would collapse the realized mix onto 2013-2018 and put the recency confound
# back in, invisibly.
#
# What it does, per file, only for files that DO NOT already open as parquet:
# download from the Hub to a staging dir on the same filesystem, verify it
# (exact Hub size, PAR1 footer, and pyarrow reports rows), and only then
# os.replace() it over the stub — atomic, so an interrupted run can never
# leave a half-written file where a good one was. A file that already passes
# is never opened for writing, so this cannot damage an intact mirror and is
# safe to re-run.
#
#   ./refetch_fineweb.sh --dry-run      # list the work, download nothing
#   sbatch refetch_fineweb.sh           # do it
#
# Env overrides: TARGET_TOKENS, FINEWEB, STAGING, WORKERS.
set -euo pipefail
source ~/.bashrc
conda activate snr

FINEWEB=${FINEWEB:-/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb/data}
# Same /capstor/store/cscs filesystem as $FINEWEB, so the final rename is a
# true atomic replace rather than a copy across mounts.
STAGING=${STAGING:-/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/.fineweb-refetch}
# 220e9, not the build's 184e9 target: the 20% margin costs nothing (184B needs
# 66 repairs, 200B and 220B both need 81) and keeps the build from stalling on
# a fresh stub if the 0.3452 tokens/byte estimate undershoots.
TARGET_TOKENS=${TARGET_TOKENS:-220e9}
WORKERS=${WORKERS:-4}
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

export FINEWEB STAGING TARGET_TOKENS WORKERS DRY_RUN
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data

python3.11 - <<'REFETCH_PY'
import os, stat, subprocess, sys, json
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from create_data_mixture import discover_dclm_files

REPO, REPO_TYPE = "HuggingFaceFW/fineweb", "dataset"
FINEWEB = os.environ["FINEWEB"]
STAGING = os.environ["STAGING"]
TARGET = float(os.environ["TARGET_TOKENS"])
WORKERS = int(os.environ["WORKERS"])
DRY_RUN = os.environ["DRY_RUN"] == "1"
# Measured on the 54 files the stalled build did read: 116.0 GB -> 40.0243B.
TOKENS_PER_BYTE = 0.3452


def repo_path(local: str) -> str:
    """The Hub key for a local mirror path: data/<crawl>/<file>.parquet."""
    return "data/" + os.path.relpath(local, FINEWEB)


def readable(path: str) -> bool:
    """Does this open as parquet? The footer magic is the cheap half (a
    truncated download keeps PAR1 at the head and loses it at the tail); the
    metadata read is the half that catches a file corrupt further in."""
    try:
        with open(path, "rb") as fh:
            fh.seek(-4, 2)
            if fh.read(4) != b"PAR1":
                return False
        return pq.ParquetFile(path).metadata.num_rows > 0
    except Exception:
        return False


print(f"Hub sizes for {REPO} ...", flush=True)
sizes = {s.rfilename: s.size
         for s in HfApi().repo_info(REPO, repo_type=REPO_TYPE,
                                    files_metadata=True).siblings
         if s.rfilename.endswith(".parquet")}

# Exactly the order the build reads, so "the files the build needs" is the
# prefix of this list that reaches the target — nothing else is our problem.
files = discover_dclm_files(FINEWEB, max_year=2022)
todo, need, est = [], 0, 0.0
for local in files:
    if est >= TARGET:
        break
    size = sizes.get(repo_path(local))
    if size is None:          # in the mirror, not in the repo: not ours to fix
        print(f"  NOT IN REPO, skipped: {repo_path(local)}", flush=True)
        continue
    need += 1
    est += size * TOKENS_PER_BYTE
    if not readable(local):
        todo.append((local, size))

print(f"\n{need} files reach {est/1e9:.1f}B tokens; {len(todo)} need re-fetching"
      f" ({sum(s for _, s in todo)/1e9:.1f} GB)\n", flush=True)
for local, size in todo:
    print(f"  {repo_path(local):55s} {os.path.getsize(local)/1e6:9.1f} MB"
          f" -> {size/1e6:9.1f} MB")
if not todo:
    print("\nNothing to do — every file the build needs already opens.")
    sys.exit(0)
if DRY_RUN:
    print("\n--dry-run: nothing downloaded.")
    sys.exit(0)

os.makedirs(STAGING, exist_ok=True)
failed = []


def inherit_sharing(src: str, model: str) -> None:
    """Give `src` the mode and ACL of the file it is about to replace.

    os.replace MOVES AN INODE. The file that lands in the shared mirror keeps
    the STAGING file's owner, mode and ACL — it does NOT pick up the target
    directory's setgid bit or default ACL. STAGING is private (other::---, no
    infra01 entry) while the mirror is group:infra01:rwx / other::r-x, so
    replacing without this strips every other infra01 and csstaff user's read
    access to each repaired file, and only this user or root could give it
    back. Copying the VICTIM's own metadata is what keeps the mirror exactly
    as shared as it was; getfacl/setfacl carry the named-group entries that a
    chmod alone would drop.
    """
    os.chmod(src, stat.S_IMODE(os.stat(model).st_mode))
    acl = subprocess.run(["getfacl", "-p", model],
                         capture_output=True, text=True)
    if acl.returncode == 0:
        subprocess.run(["setfacl", "--set-file=-", src],
                       input=acl.stdout, text=True, check=False)


def fetch(item) -> None:
    local, size = item
    key = repo_path(local)
    try:
        tmp = hf_hub_download(REPO, key, repo_type=REPO_TYPE, local_dir=STAGING)
        got = os.path.getsize(tmp)
        if got != size:
            failed.append(f"{key}: got {got} bytes, Hub says {size}")
            return
        if not readable(tmp):
            failed.append(f"{key}: downloaded file still does not open as parquet")
            return
        # Re-check the target here, not just in the planning pass: never replace a
        # file that has become readable since (another repair run, or a restored
        # mirror). Only a file that is still broken is ever overwritten.
        if readable(local):
            print(f"  skip (now intact): {key}", flush=True)
            return
        inherit_sharing(tmp, local)
        os.replace(tmp, local)          # same filesystem => atomic
        print(f"  ok {key} ({size/1e6:.1f} MB)", flush=True)
    except Exception as e:
        # One network error must not abort the pool before the verification
        # pass and the per-file summary below ever print.
        failed.append(f"{key}: {type(e).__name__}: {e}")


with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    list(pool.map(fetch, todo))

still_bad = [repo_path(p) for p, _ in todo if not readable(p)]
print(f"\nre-fetched {len(todo) - len(still_bad)}/{len(todo)}")
for f in failed:
    print(f"  FAILED {f}")
for f in still_bad:
    print(f"  STILL BROKEN {f}")
if failed or still_bad:
    sys.exit(1)
print("\nEvery file the FWEB build needs now opens as parquet.")
print("Spot-check the sharing of a repaired file before trusting the rest:")
print(f"  getfacl -p {todo[0][0]}")
REFETCH_PY
