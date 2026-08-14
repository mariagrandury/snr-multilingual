"""Read per-instance lm_eval samples from `multilingual-snr/samples`.

This file is intended to ship in the root of the published hub repo. The
companion uploader is `push_samples.py` in this directory.

Layout convention on the hub (1:1 with local eval_logs filenames):

    samples/<NAME>/<task>/<eval_TS>.jsonl

Each line is a stripped lm_eval per-instance record. Fields `doc` and
`arguments` were dropped at upload time (they're ~79% of the bytes
and re-derivable from `(task, doc_id)` by re-running lm_eval setup).
Kept fields: doc_id, target, resps, filtered_resps, filter, metrics,
acc, acc_norm, acc_bytes, exact_match, doc_hash, prompt_hash,
target_hash.

Usage:
    from samples_loader import load_samples, list_eval_timestamps, list_tasks

    # latest eval_TS for the (name, task) pair
    for r in load_samples("apertus-175M-fwEdu30-fw270-seed1904-iter50000", "piqa"):
        print(r["doc_id"], r["filtered_resps"], r["acc"])

    # specific eval timestamp
    for r in load_samples("...", "piqa", eval_ts="2026-05-04T12-13-14.000000"):
        ...

    # discover what's available
    tasks = list_tasks("apertus-175M-fwEdu30-fw270-seed1904-iter50000")
    tss = list_eval_timestamps(name, task)

To recover the original `doc` (question text, choices, etc.):

    from lm_eval.tasks import TaskManager
    task = TaskManager().load_task_or_group("piqa")[0]
    docs = list(task.test_docs())  # indexable by doc_id
    doc = docs[record["doc_id"]]
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterator, Optional

REPO_ID = "multilingual-snr/samples"


@lru_cache(maxsize=1)
def _all_files(token: Optional[str] = None) -> tuple[str, ...]:
    """Cached listing of every file in the repo (large; one fetch per process)."""
    from huggingface_hub import list_repo_files
    return tuple(list_repo_files(REPO_ID, repo_type="dataset", token=token))


def list_names(token: Optional[str] = None) -> list[str]:
    """All NAMEs (checkpoint identifiers) present on the hub."""
    return sorted({f.split("/", 2)[1] for f in _all_files(token) if f.startswith("samples/")})


def list_tasks(name: str, token: Optional[str] = None) -> list[str]:
    """All tasks evaluated for NAME on the hub."""
    prefix = f"samples/{name}/"
    return sorted({
        f[len(prefix):].split("/", 1)[0]
        for f in _all_files(token) if f.startswith(prefix) and f.count("/") >= 3
    })


def list_eval_timestamps(name: str, task: str, token: Optional[str] = None) -> list[str]:
    """All eval_TS strings available for (name, task)."""
    prefix = f"samples/{name}/{task}/"
    sfx = ".jsonl"
    return sorted(
        f[len(prefix):-len(sfx)]
        for f in _all_files(token)
        if f.startswith(prefix) and f.endswith(sfx)
    )


def load_samples(
    name: str,
    task: str,
    eval_ts: Optional[str] = None,
    token: Optional[str] = None,
) -> Iterator[dict]:
    """Yield each per-instance dict for (name, task). If eval_ts is None, picks
    the lexically-latest available timestamp (which == the most recent eval
    given the YYYY-MM-DDTHH-MM-SS.NNNNNN format)."""
    from huggingface_hub import hf_hub_download
    if eval_ts is None:
        tss = list_eval_timestamps(name, task, token=token)
        if not tss:
            raise FileNotFoundError(f"No samples for {name}/{task} on {REPO_ID}")
        eval_ts = tss[-1]
    path = hf_hub_download(
        REPO_ID, f"samples/{name}/{task}/{eval_ts}.jsonl",
        repo_type="dataset", token=token,
    )
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
