#!/usr/bin/env python3
"""Push converted predictivity HF checkpoints to the public `msnr` HF org.

Run MANUALLY from the login node (needs HF_TOKEN + outbound network). The
auto-eval watcher converts every checkpoint to HF and persists it on capstor
(`/capstor/store/cscs/swissai/infra01/msnr-hf-models/`); this script mirrors
that capstor tree to the Hub.

Layout:
  capstor  <STAGING_BASE>/<cell>/iter_<NNNNNNN>/          (cell = the run name)
  hub      msnr/<cell>:branch=step-<N>   (and 'main' = the highest iter)

`msnr` repos are PUBLIC (no private-storage quota), one repo per cell, one
branch per iter, main mirrors the latest iter.

Examples:
  python push-snr.py                                    # push every cell in staging
  python push-snr.py --name lm-90M-L2-deep-seed1904     # one cell
  python push-snr.py --name lm-90M-L2-deep-seed1904 --iters 225 4500
  python push-snr.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

ORG = "msnr"  # single PUBLIC org — public repos have no storage limit
DEFAULT_STAGING = Path("/capstor/store/cscs/swissai/infra01/msnr-hf-models")

# A converted iter dir is ready once it has the .hf_complete marker the
# convert scripts touch LAST (it implies config + weights; those alone can
# match a half-written save_pretrained, and a partial push is permanent via
# the branch-exists skip). Marker-less pre-marker snapshots get backfilled
# by the next convert job over the cell.
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")  # main_is_empty_stub
ITER_RE = re.compile(r"^iter_(\d+)$")
CELL_PREFIX = "lm-"


def iter_dirs(staging: Path, cell: str, requested: list[int] | None) -> list[tuple[int, Path]]:
    base = staging / cell
    if not base.is_dir():
        print(f"[push-snr] no staging dir: {base}")
        return []
    out: list[tuple[int, Path]] = []
    for d in sorted(base.iterdir()):
        m = ITER_RE.match(d.name)
        if not m:
            continue
        step = int(m.group(1))
        if requested is not None and step not in requested:
            continue
        if not (d / ".hf_complete").is_file():
            print(f"[push-snr] skip {d}: no .hf_complete (conversion "
                  f"incomplete or in flight — a convert-snr pass backfills it)")
            continue
        out.append((step, d))
    return out


def _retry_on_429(fn, *args, max_attempts: int = 20, **kwargs):
    """Run fn; on HTTP 429, sleep the server-suggested Retry-After (or
    exponential backoff) and try again. The team plan caps API at 3000 req /
    5 min, and bursty pushes can hit that when other agents share the token."""
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
                m = re.search(r"Retry after (\d+) seconds", str(e))
                if m:
                    wait = float(m.group(1))
            if wait is None:
                wait = delay
                delay *= 2
            wait += 5
            print(f"[push-snr] 429 rate-limited (attempt {attempt}/{max_attempts}); sleeping {wait:.0f}s")
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def ensure_repo(api: HfApi, repo_id: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[push-snr] (dry-run) would ensure repo: {repo_id} (public)")
        return
    _retry_on_429(api.create_repo, repo_id, private=False, exist_ok=True, repo_type="model")


def existing_branches(api: HfApi, repo_id: str) -> set[str]:
    try:
        refs = api.list_repo_refs(repo_id, repo_type="model")
        return {b.name for b in refs.branches}
    except HfHubHTTPError:
        return set()


def main_is_empty_stub(api: HfApi, repo_id: str) -> bool:
    """create_repo seeds main with just .gitattributes. Mirror to main only
    while it's still that empty stub (or on --force), avoiding re-uploads."""
    try:
        files = api.list_repo_files(repo_id, revision="main")
    except HfHubHTTPError:
        return False
    return not any(f in files for f in ("config.json", *WEIGHT_FILES))


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
        api.upload_folder, repo_id=repo_id, folder_path=str(src), revision=branch,
        commit_message=f"Upload converted Megatron checkpoint ({branch})", repo_type="model",
        ignore_patterns=[".hf_complete"],  # convert-snr.sh's completion marker
    )
    print(f"[push-snr] pushed {repo_id}@{branch}")
    return True


def push_cell(api: HfApi, staging: Path, cell: str, args) -> None:
    pairs = iter_dirs(staging, cell, args.iters)
    if not pairs:
        print(f"[push-snr] nothing to push for {cell}")
        return
    repo_id = f"{ORG}/{cell}"
    print(f"[push-snr] {cell}: iters {[s for s, _ in pairs]} -> {repo_id}")
    ensure_repo(api, repo_id, args.dry_run)
    for i, (step, src) in enumerate(pairs):
        # zero-pad so branches sort lexicographically (step-002000 < step-081000)
        push_iter(api, repo_id, src, f"step-{step:06d}", args.dry_run, args.force)
        if not args.dry_run and i < len(pairs) - 1:
            time.sleep(15)  # spread across the team-plan rate-limit window
    if not args.no_main_mirror:
        latest_step, latest_src = max(pairs, key=lambda p: p[0])
        if args.dry_run or args.force or main_is_empty_stub(api, repo_id):
            push_iter(api, repo_id, latest_src, "main", args.dry_run, force=True)
        else:
            print(f"[push-snr] {repo_id}@main: already populated, skipping (use --force to re-mirror)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="restrict to one cell (default: every cell in staging)")
    ap.add_argument("--iters", type=int, nargs="*", default=None,
                    help="Restrict to these iter steps; default = all under the cell.")
    ap.add_argument("--no-main-mirror", action="store_true",
                    help="Skip mirroring the highest iter to the main branch.")
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-push even if a branch already exists on the hub.")
    args = ap.parse_args()

    if not args.staging.is_dir():
        sys.exit(f"[push-snr] no staging dir: {args.staging}")
    if args.name:
        cells = [args.name]
    else:
        cells = sorted(d.name for d in args.staging.iterdir()
                       if d.is_dir() and d.name.startswith(CELL_PREFIX))
    if not cells:
        sys.exit(f"[push-snr] no cells found under {args.staging}")
    print(f"[push-snr] staging: {args.staging}")
    print(f"[push-snr] cells:   {len(cells)}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for cell in cells:
        push_cell(api, args.staging, cell, args)
    print("[push-snr] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
 