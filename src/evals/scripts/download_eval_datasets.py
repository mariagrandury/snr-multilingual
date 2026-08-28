#!/usr/bin/env python3
"""Build every eval dataset into the iopsstor HF cache, all configs.

Run on the LOGIN node — compute nodes have no internet, and the harness runs
all tasks in one batched call, so a single missing dataset aborts the whole
eval.

Uses load_dataset (not just snapshot_download): the harness needs the BUILT
arrow cache under $HF_HOME/datasets/, which only load_dataset produces.
Configs are enumerated from the Hub, so a repo's new configs are picked up
automatically — nothing to hand-list.

Idempotent: already-built configs are a no-op, so re-run any time. Adding a
benchmark = add its repo to configs/eval_datasets.txt.

    python3.11 scripts/download_eval_datasets.py
"""
import os
from pathlib import Path

# Datasets live on iopsstor: the sweep touches them on every eval, so the
# access-based cleaning policy keeps them alive. Set before importing datasets.
os.environ["HF_HOME"] = f"/iopsstor/scratch/cscs/{os.environ['USER']}/hf_home"
os.environ["HF_DATASETS_CACHE"] = f"{os.environ['HF_HOME']}/datasets"

from datasets import get_dataset_config_names, load_dataset  # noqa: E402
from huggingface_hub.utils import disable_progress_bars  # noqa: E402

disable_progress_bars()   # keep the log to one line per config

import sys  # noqa: E402

# Optional argv[1]: an alternative manifest (e.g. a priority subset to build first).
MANIFEST = (Path(sys.argv[1]) if len(sys.argv) > 1
            else Path(__file__).resolve().parents[1] / "configs" / "eval_datasets.txt")
repos = [ln.strip() for ln in MANIFEST.read_text().splitlines()
         if ln.strip() and not ln.startswith("#")]
print(f"{len(repos)} dataset repos -> {os.environ['HF_DATASETS_CACHE']}")

failed, built = [], 0
for i, repo in enumerate(repos, 1):
    try:
        configs = get_dataset_config_names(repo) or [None]
    except Exception as e:
        failed.append((repo, str(e).split("\n")[0][:110]))
        print(f"[{i}/{len(repos)}] FAIL {repo} (configs): {failed[-1][1]}")
        continue
    for cfg in configs:
        label = f"{repo}:{cfg}" if cfg else repo
        try:
            load_dataset(repo, cfg)
            built += 1
            print(f"[{i}/{len(repos)}] ok   {label}")
        except Exception as e:      # one bad config must not stop the rest
            failed.append((label, str(e).split("\n")[0][:110]))
            print(f"[{i}/{len(repos)}] FAIL {label}: {failed[-1][1]}")

print(f"\ndone: {built} configs built, {len(failed)} failed")
for label, err in failed:
    print(f"  {label}: {err}")
raise SystemExit(1 if failed else 0)
