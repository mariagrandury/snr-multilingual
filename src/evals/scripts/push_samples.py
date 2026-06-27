#!/usr/bin/env python3
"""Push surviving samples_*.jsonl to multilingual-snr/samples.

Companion to build_hf_dataset.py: that publishes the aggregate parquets,
this publishes the per-instance lm_eval samples. Two changes from raw:

  1. Per-record field stripping. `doc` and `arguments` together account for
     ~79% of the bytes (44.7% + 34.4%; sampled 2026-06-01). Both are
     re-derivable from (task, doc_id) by re-running lm_eval setup, so we
     drop them at upload time. ~608 GB raw -> ~127 GB on the hub.
     Kept: doc_id, target, resps, filtered_resps, filter, metrics, acc,
     acc_norm, acc_bytes, exact_match, doc_hash, prompt_hash, target_hash.

  2. ADDITIVE-ONLY uploads. The hub copy is the canonical archive — once
     a (NAME, task, eval_TS).jsonl is published, it's never overwritten.
     `api.list_repo_files()` is fetched once per run to build the set of
     existing in-repo paths; local files whose target path is already
     present are skipped. Re-runs after new evals land = idempotent
     append-only.

Canonical in-repo layout (matches the eval_logs filename convention
1:1 — every NAME/task pair has exactly one eval_TS per the audit on
2026-06-01):

    samples/<NAME>/<task>/<eval_TS>.jsonl

Wire-up: chunked create_commit (default 500 files per commit) with
back-off on HTTP 429 (HF team plan caps API at 3000 req / 5 min;
see eval CLAUDE.md bug #12).

Usage:
    # dry-run: list what would be uploaded
    python scripts/push_samples.py --dry-run

    # test small batch first
    python scripts/push_samples.py --limit-files 100

    # full push (long-running; ~7h for 266k files / 127 GB)
    python scripts/push_samples.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.utils import HfHubHTTPError

EVAL_LOGS_DEFAULT = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/snr-experiments"
)
REPO_ID = "multilingual-snr/samples"

# Fields dropped from each jsonl record before upload (derivable from
# (task, doc_id) via lm_eval; bulk of the bytes).
DROP_FIELDS = ("doc", "arguments")

# lm_eval samples filename: samples_<task>_<YYYY-MM-DDTHH-MM-SS.NNNNNN>.jsonl
_FN_RE = re.compile(
    r"^samples_(?P<task>.+)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl$"
)

DEFAULT_BATCH = 100
DEFAULT_SLEEP_S = 15


def canonical_path(name: str, task: str, ts: str) -> str:
    return f"samples/{name}/{task}/{ts}.jsonl"


def stripped_bytes(src: Path) -> bytes:
    """Read src jsonl, drop DROP_FIELDS from each line, return new bytes."""
    out_lines = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # Preserve malformed lines verbatim — loader can deal.
                out_lines.append(line)
                continue
            for k in DROP_FIELDS:
                r.pop(k, None)
            out_lines.append(json.dumps(r, ensure_ascii=False) + "\n")
    return "".join(out_lines).encode("utf-8")


def plan(eval_logs: Path) -> list[tuple[Path, str]]:
    """Walk eval_logs and return [(local_src, canonical_hub_path), ...]."""
    out: list[tuple[Path, str]] = []
    for nd in sorted(eval_logs.iterdir()):
        if not nd.is_dir():
            continue
        for p in nd.rglob("samples_*.jsonl"):
            m = _FN_RE.match(p.name)
            if not m:
                continue
            out.append((p, canonical_path(nd.name, m.group("task"), m.group("ts"))))
    return out


def existing_hub_paths(api: HfApi, repo_id: str) -> set[str]:
    """All files currently in the repo, for additive-only enforcement.

    Best-effort: the listing endpoint paginates and HF Hub returns 504
    (gateway timeout) on repos with 100K+ files. If listing fails after
    a small in-house retry, return an empty set — the caller will rely
    on the local manifest cache (see manifest_paths()) for the additive
    check instead.
    """
    delay = 30
    for attempt in range(1, 4):
        try:
            return set(api.list_repo_files(repo_id, repo_type="dataset"))
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 404:
                return set()  # fresh repo
            print(f"[hub] list_repo_files attempt {attempt}/3: HTTP {status}; "
                  f"sleeping {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
        except Exception as e:
            print(f"[hub] list_repo_files attempt {attempt}/3: "
                  f"{type(e).__name__}: {e!s:.120s}; sleeping {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    print("[hub] list_repo_files exhausted retries; falling back to manifest-only "
          "additive check (may re-attempt some uploads, but commit-overwrite is idempotent).",
          flush=True)
    return set()


def manifest_paths(manifest: Path) -> set[str]:
    """Locally-cached set of in-repo paths we've already successfully pushed.

    Written incrementally by push_batch_with_manifest after every successful
    commit. Survives across process restarts so resume doesn't depend on
    list_repo_files succeeding (which fails on big repos with HTTP 504)."""
    if not manifest.is_file():
        return set()
    with manifest.open() as f:
        return {line.strip() for line in f if line.strip()}


def append_manifest(manifest: Path, paths: list[str]) -> None:
    """Atomically append paths to the manifest, one per line."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as f:
        for p in paths:
            f.write(p + "\n")


def push_batch(api: HfApi, repo_id: str, batch: list[tuple[Path, str]],
               batch_label: str, max_attempts: int = 12) -> None:
    """Build CommitOperationAdd for each (src, dst) — bytes are stripped in
    memory, no temp files — and create_commit with broad retry coverage.

    Retries on HTTP 429 (honouring Retry-After) and on any transient
    transport-level failure (ConnectionError, broken pipe, timeout) with
    exponential back-off. The first hour-long push attempt died mid-batch
    after batch 53 with no logged exception — almost certainly a silent
    connection reset on a 500+ MB upload — so we now catch the broad base
    Exception class for transport-layer issues. Hub-side state is
    transactional (a commit either fully lands or doesn't), so retrying a
    failed batch is safe.
    """
    ops = [
        CommitOperationAdd(path_in_repo=dst, path_or_fileobj=stripped_bytes(src))
        for src, dst in batch
    ]
    attempt = 1
    delay = 30
    last_exc: BaseException | None = None
    while attempt <= max_attempts:
        try:
            commit = api.create_commit(
                repo_id=repo_id, repo_type="dataset", operations=ops,
                commit_message=f"Add {len(batch)} samples files ({batch_label})",
            )
            print(f"[push] {batch_label}: {len(batch)} files -> {commit.commit_url}", flush=True)
            return
        except HfHubHTTPError as e:
            last_exc = e
            status = getattr(e.response, "status_code", None) if e.response is not None else None
            if status == 429:
                # Honour Retry-After if the server sent one.
                ra = None
                if e.response is not None:
                    ra = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
                try:
                    wait = float(ra) if ra else delay
                except ValueError:
                    wait = delay
                wait += 5
            elif status in (502, 503, 504):
                wait = delay
            else:
                # Permanent 4xx/5xx — bubble up.
                raise
            print(f"[push] HTTP {status} on {batch_label} (attempt {attempt}/{max_attempts}); sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
        except (ConnectionError, TimeoutError, OSError) as e:
            # Transport-layer disconnects (ConnectionResetError, BrokenPipeError,
            # socket timeout, etc.). Retry with back-off.
            last_exc = e
            print(f"[push] transport error on {batch_label} ({type(e).__name__}: {e!s:.80s}); "
                  f"attempt {attempt}/{max_attempts}; sleeping {delay}s", flush=True)
            time.sleep(delay)
        except Exception as e:
            # Unknown exception class — log loudly and retry once or twice
            # before giving up. Better than silent death.
            last_exc = e
            print(f"[push] unexpected error on {batch_label} ({type(e).__name__}: {e!s:.80s}); "
                  f"attempt {attempt}/{max_attempts}; sleeping {delay}s", flush=True)
            time.sleep(delay)
        attempt += 1
        delay = min(delay * 2, 600)
    # Exhausted: re-raise the last exception so the loop in main() sees it.
    assert last_exc is not None
    raise last_exc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval-logs", default=str(EVAL_LOGS_DEFAULT), type=Path)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                    help=f"Files per create_commit call (default {DEFAULT_BATCH})")
    ap.add_argument("--inter-batch-sleep", type=float, default=DEFAULT_SLEEP_S,
                    help=f"Seconds between commits to spread API calls (default {DEFAULT_SLEEP_S})")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Stop after N batches (smoke-test).")
    ap.add_argument("--limit-files", type=int, default=None,
                    help="Cap total files to upload (smoke-test).")
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).resolve().parent / "push-samples-pushed.txt",
                    help="Local cache file listing in-repo paths we've already pushed. "
                         "Written incrementally; survives restarts so resume doesn't "
                         "depend on hub-side list_repo_files (which 504s on big repos).")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))

    if not args.dry_run:
        try:
            # Always public — at ~127 GB this exceeds the team-plan private
            # storage quota, so the dataset must be public to fit.
            api.create_repo(args.repo_id, repo_type="dataset",
                            exist_ok=True, private=False)
        except HfHubHTTPError as e:
            sys.exit(f"create_repo failed: {e}")

    pairs = plan(args.eval_logs)
    print(f"[plan] {len(pairs):,} local samples.jsonl under {args.eval_logs}", flush=True)

    # Additive-only check: union of hub list (best-effort; can fail) and the
    # local manifest cache (authoritative for our own past successes).
    manifest_set = manifest_paths(args.manifest)
    print(f"[manifest] {len(manifest_set):,} paths already pushed (from {args.manifest})", flush=True)

    print("[hub] listing existing files (best-effort cross-check) ...", flush=True)
    hub_set = existing_hub_paths(api, args.repo_id)
    print(f"[hub] {len(hub_set):,} files reported by hub list", flush=True)

    # First time the hub list succeeds, seed the manifest so future restarts
    # can resume without depending on the hub listing endpoint (which 504s
    # once a repo has ~100K+ files).
    new_for_manifest = sorted(hub_set - manifest_set)
    if new_for_manifest:
        append_manifest(args.manifest, new_for_manifest)
        print(f"[manifest] seeded with {len(new_for_manifest):,} new entries from hub list", flush=True)

    existing = manifest_set | hub_set
    print(f"[plan] {len(existing):,} unique paths to skip (manifest ∪ hub)", flush=True)

    # NOTE: loader (samples_loader.py) and a README live alongside this
    # script in src/evals/scripts/. They are intentionally NOT pushed to
    # the dataset repo yet — release them when the paper drops.

    todo = [(s, d) for s, d in pairs if d not in existing]
    print(f"[plan] {len(todo):,} new files to push (rest skipped — additive-only)")
    if args.limit_files is not None:
        todo = todo[: args.limit_files]
        print(f"[plan] --limit-files: capped to {len(todo)}")
    if not todo:
        print("[plan] nothing to push.")
        return 0

    # Project sizes by sampling the strip ratio on the first 50 files.
    sample_n = min(50, len(todo))
    raw_sample = sum(s.stat().st_size for s, _ in todo[:sample_n])
    stripped_sample = sum(len(stripped_bytes(s)) for s, _ in todo[:sample_n])
    ratio = (stripped_sample / raw_sample) if raw_sample else 1.0
    raw_total = sum(s.stat().st_size for s, _ in todo)
    print(f"[size] raw total: {raw_total/1e9:.2f} GB  -> "
          f"after strip: ~{raw_total*ratio/1e9:.2f} GB  (sampled ratio: {ratio:.1%})")

    if args.dry_run:
        for src, dst in todo[:5]:
            print(f"  (dry-run) would upload {src.name} -> {dst}")
        if len(todo) > 5:
            print(f"  ... + {len(todo) - 5} more")
        return 0

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    print(f"[push] {n_batches} commits of up to {args.batch_size} files each")

    pushed = 0
    for i in range(n_batches):
        if args.max_batches is not None and i >= args.max_batches:
            print(f"[push] --max-batches: stopping after {args.max_batches}", flush=True)
            break
        batch = todo[i * args.batch_size : (i + 1) * args.batch_size]
        push_batch(api, args.repo_id, batch, batch_label=f"batch {i+1}/{n_batches}")
        # Append destination paths to manifest only AFTER the commit
        # succeeds, so a crash mid-batch doesn't mark un-pushed paths.
        append_manifest(args.manifest, [dst for _, dst in batch])
        pushed += len(batch)
        if i + 1 < n_batches:
            time.sleep(args.inter_batch_sleep)
    print(f"[push] done: {pushed} files pushed across {min(i+1, n_batches)} commit(s).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
