#!/usr/bin/env python3
"""Push converted HF checkpoints to the snr-models HF org, one branch per iter.

Layout:
  staging  <STAGING_BASE>/apertus-<SIZE>-fwEdu<FW_EDU_RATIO>-fw2<FW2_RATIO>-seed<SEED>/iter_<NNNNNNN>/
  hub      snr-models/<exp_name>:branch=stage1-step-<N>  (and 'main' = last iter)

Run from the login node — only needs HF_TOKEN and outbound network.

Examples:
  # push all iters of one cell, mirroring the highest iter to main
  python push-snr.py --size 175M --fw-edu 30 --seed 1904

  # push a specific subset
  python push-snr.py --size 175M --fw-edu 30 --seed 1904 --iters 2000 50000

  # dry run
  python push-snr.py --size 175M --fw-edu 30 --seed 1904 --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import time

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Per-seed org to spread HF private storage quota across three accounts:
#   snr-models-1904, snr-models-1797, snr-models-28
# (the original `snr-models` was renamed to snr-models-1904 by the user
# after hitting the private storage limit on a single org around 2026-05-04.)
def org_for_seed(seed: int) -> str:
    return f"snr-models-{seed}"


DEFAULT_STAGING = Path("/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints")

# Files that must exist for a converted iter dir to be considered ready.
REQUIRED_FILES = ("config.json",)
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")

ITER_RE = re.compile(r"^iter_(\d+)$")


def fw2_for(fw_edu: int) -> int:
    if fw_edu + (300 - fw_edu) == 300:  # always true; just being explicit
        pass
    return 100 - fw_edu  # mix is fwEdu / fw2 split out of 100


def exp_name(size: str, fw_edu: int, seed: int) -> str:
    return f"apertus-{size}-fwEdu{fw_edu}-fw2{fw2_for(fw_edu)}-seed{seed}"


def iter_dirs(staging: Path, name: str, requested: list[int] | None) -> list[tuple[int, Path]]:
    base = staging / name
    if not base.is_dir():
        sys.exit(f"[push-snr] no staging dir: {base}")
    out: list[tuple[int, Path]] = []
    for d in sorted(base.iterdir()):
        m = ITER_RE.match(d.name)
        if not m:
            continue
        step = int(m.group(1))
        if requested is not None and step not in requested:
            continue
        if not (d / "config.json").is_file():
            print(f"[push-snr] skip {d}: missing config.json")
            continue
        if not any((d / w).exists() for w in WEIGHT_FILES):
            print(f"[push-snr] skip {d}: no weights")
            continue
        out.append((step, d))
    return out


def _retry_on_429(fn, *args, max_attempts: int = 20, **kwargs):
    """Run fn(*args, **kwargs); on HTTP 429, sleep the server-suggested
    Retry-After (or exponential backoff) and try again.

    The team plan caps API at 3000 req / 5 min, and bursty pushes can hit
    that even with small file counts when other agents share the token."""
    delay = 10.0
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None) if e.response is not None else None
            if status != 429:
                raise
            last_exc = e
            wait = None
            if e.response is not None:
                ra = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
                if ra:
                    try:
                        wait = float(ra)
                    except ValueError:
                        pass
            if wait is None:
                # parse "Retry after N seconds" from the message body
                msg = str(e)
                m = re.search(r"Retry after (\d+) seconds", msg)
                if m:
                    wait = float(m.group(1))
            if wait is None:
                wait = delay
                delay *= 2
            wait += 5  # cushion for clock skew
            print(f"[push-snr] 429 rate-limited (attempt {attempt}/{max_attempts}); sleeping {wait:.0f}s")
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def ensure_repo(api: HfApi, repo_id: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[push-snr] (dry-run) would ensure repo: {repo_id} (private)")
        return
    _retry_on_429(api.create_repo, repo_id, private=True, exist_ok=True, repo_type="model")


def existing_branches(api: HfApi, repo_id: str) -> set[str]:
    try:
        refs = api.list_repo_refs(repo_id, repo_type="model")
        return {b.name for b in refs.branches}
    except HfHubHTTPError:
        return set()


def main_is_empty_stub(api: HfApi, repo_id: str) -> bool:
    """create_repo seeds main with just .gitattributes (no model weights or
    config). Once we mirror an iter to main, those files appear. This lets
    us mirror only when main is still the empty stub, avoiding wasteful
    re-uploads on every re-run."""
    try:
        files = api.list_repo_files(repo_id, revision="main")
    except HfHubHTTPError:
        return False
    return not any(
        f in files for f in ("config.json", "model.safetensors", "model.safetensors.index.json")
    )


def push_iter(api: HfApi, repo_id: str, src: Path, branch: str, dry_run: bool, force: bool) -> bool:
    """Returns True iff the branch is confirmed live on the hub after this
    call (newly uploaded, or already present). False on dry-run."""
    branches = existing_branches(api, repo_id)
    if branch in branches and not force:
        print(f"[push-snr] {repo_id}@{branch}: already exists, skipping (use --force to re-push)")
        return True
    if dry_run:
        print(f"[push-snr] (dry-run) would upload {src} -> {repo_id}@{branch}")
        return False
    if branch != "main" and branch not in branches:
        _retry_on_429(api.create_branch, repo_id, branch=branch, exist_ok=True)
    _retry_on_429(
        api.upload_folder,
        repo_id=repo_id,
        folder_path=str(src),
        revision=branch,
        commit_message=f"Upload converted Megatron checkpoint ({branch})",
        repo_type="model",
    )
    print(f"[push-snr] pushed {repo_id}@{branch}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True, choices=["175M", "350M", "600M", "1B"])
    ap.add_argument("--fw-edu", required=True, type=int, choices=[30, 60, 90])
    ap.add_argument("--seed", required=True, type=int, choices=[28, 1797, 1904])
    ap.add_argument("--iters", type=int, nargs="*", default=None,
                    help="Restrict to these iter steps; default = all under staging.")
    ap.add_argument("--no-main-mirror", action="store_true",
                    help="Skip mirroring the highest iter to the main branch.")
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-push even if a branch already exists on the hub.")
    args = ap.parse_args()

    name = exp_name(args.size, args.fw_edu, args.seed)
    repo_id = f"{org_for_seed(args.seed)}/{name}"
    print(f"[push-snr] cell: {name}")
    print(f"[push-snr] repo: {repo_id}")
    print(f"[push-snr] staging: {args.staging / name}")

    pairs = iter_dirs(args.staging, name, args.iters)
    if not pairs:
        sys.exit(f"[push-snr] nothing to push for {name}")

    print(f"[push-snr] iters to push: {[s for s, _ in pairs]}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    ensure_repo(api, repo_id, args.dry_run)

    confirmed_steps: list[int] = []
    for i, (step, src) in enumerate(pairs):
        # 5-digit zero-pad so branch names sort lexicographically
        # (stage1-step-02000 < stage1-step-50000). Original training caps at
        # 50000 iters, so 5 digits is enough.
        branch = f"stage1-step-{step:05d}"
        if push_iter(api, repo_id, src, branch, args.dry_run, args.force):
            confirmed_steps.append(step)
        # Spread API calls across the team-plan rate-limit window (3000
        # req / 5 min). Without this, a cell with many real uploads bursts
        # through the budget and the next cell starts in a saturated state.
        # Skip the sleep on the last iter and on dry-runs.
        if not args.dry_run and i < len(pairs) - 1:
            time.sleep(15)

    if not args.no_main_mirror:
        # The highest iter present gets mirrored to main. Only push if main
        # is still the empty auto-created stub from create_repo, OR the user
        # passed --force. Skips redundant re-uploads on subsequent runs.
        latest_step, latest_src = max(pairs, key=lambda p: p[0])
        if args.dry_run or args.force or main_is_empty_stub(api, repo_id):
            push_iter(api, repo_id, latest_src, "main", args.dry_run, force=True)
        else:
            print(f"[push-snr] {repo_id}@main: already populated, skipping (use --force to re-mirror)")

    # (The old 36-sweep progress plot used to track pushed-to-hub state here;
    # pretrain_progress.py now covers the predictivity sweep and doesn't.)

    print(f"[push-snr] done: {repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
