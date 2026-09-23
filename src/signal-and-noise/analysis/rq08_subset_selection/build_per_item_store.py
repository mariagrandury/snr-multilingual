"""Build the per-item store the rq08 per-item analysis reads.

For every (model, step, task) row of the ladder frame — the checkpoints on the
shared grid and the noise-grid points (rules 3 and 4), the parent tasks the
cell trains (rules 2 and 6) — read the latest ``samples_<task>_*.jsonl`` under
``<cell>-iter<N>/harness/eval_*/`` (a checkpoint has several eval dirs from
the top-ups; the union is scanned and the newest file per task wins) and keep
one row per item:

    model, step, task, doc_id:int32, acc, acc_norm, ll_gold, margin (float32)

``doc_id`` is lm-eval's positional id, stable across checkpoints. ``ll_gold``
is the log-likelihood of the target choice (``resps`` per choice, ``target``
its index — a digit, or a letter for the MMLU-style tasks) and ``margin`` is
that minus the best other choice; both NaN where the record is not a
multiple-choice one. A parent with no samples file of its own (Global-MMLU,
INCLUDE, mmlu: the harness writes one file per subject) is the union of its
``samples_<parent>_<subject>_*`` files with ``doc_id = FACET_STRIDE * rank +
doc_id``, the rank being the subject's position in the sorted subject list —
stable because every checkpoint runs the same subjects.

Parquet per benchmark family, ``per_item_store/<pool>/<family>.parquet/`` (a
directory of part files, one per flush, so a killed job keeps what it wrote),
plus ``manifest.csv``; a (model, step, task) in the manifest is skipped on the
next run. The folder is data (git-ignored), not a figure.

Records are 3-5 KB and only five fields are read: the head (doc_id) and tail
(acc, acc_norm) are regexes and ``target`` / ``resps`` are sliced by index and
json-decoded alone, which is 1.4x faster than ``json.loads`` of the record; a
line the regexes cannot read is decoded whole.

    python analysis/rq08_subset_selection/build_per_item_store.py --pool predictivity_seeds --workers 64
    python ... --limit-models 1 --limit-tasks 3          # smoke test, one checkpoint dir per model
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.paths import SUBSET_SELECTION  # noqa: E402
from analysis.utils import benchmark_family, ladder_frame, on_noise_grid, on_shared_grid  # noqa: E402
from pretrain.ladder_report import EVAL_LOGS  # noqa: E402

STORE = SUBSET_SELECTION / "per_item_store"
FACET_STRIDE = 10_000        # doc_id offset per subject file of a parent without its own file
COLS = ["model", "step", "task", "doc_id", "acc", "acc_norm", "ll_gold", "margin"]
_FILE = re.compile(r"^samples_(.+)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
_HEAD = re.compile(r'^\{"doc_id": (\d+), ')
_TAIL = re.compile(r'"acc": ([^,}]+)(?:, "acc_norm": ([^,}]+))?\}\s*$')


def gold_index(target) -> int:
    """The target's choice index: a digit, or a letter (A = 0); -1 otherwise."""
    s = str(target)
    return int(s) if s.isdigit() else ord(s) - 65 if len(s) == 1 and s.isupper() else -1


def parse_record(line: str) -> tuple:
    h, t = _HEAD.match(line), _TAIL.search(line, max(0, len(line) - 64))
    if h and t:
        i = line.index('"target": '); j = line.index(', "arguments"', i)
        a = line.index('"resps": ', j); b = line.index(', "filtered_resps"', a)
        doc_id, target, resps = int(h[1]), json.loads(line[i + 10:j]), json.loads(line[a + 9:b])
        acc, acc_norm = float(t[1]), float(t[2]) if t[2] else np.nan
    else:
        o = json.loads(line)
        doc_id, target, resps = o["doc_id"], o["target"], o["resps"]
        acc, acc_norm = o.get("acc", np.nan), o.get("acc_norm", np.nan)
    ll_gold = margin = np.nan
    g = gold_index(target)
    if resps and isinstance(resps[0][0], list) and 0 <= g < len(resps):   # loglikelihood per choice
        ll = np.array([float(r[0][0]) for r in resps])
        ll_gold = ll[g]
        if len(ll) > 1:
            margin = ll_gold - np.max(np.delete(ll, g))
    return doc_id, acc, acc_norm, ll_gold, margin


def files_for(ckpt_dir: Path, tasks: list[str]) -> dict[str, list[tuple[Path, int]]]:
    """task -> [(samples file, doc_id offset)]: the newest file per task over
    every eval dir; a parent without a file is its subject files, in order."""
    latest: dict[str, tuple[str, Path]] = {}
    for f in ckpt_dir.glob("harness/eval_*/samples_*.jsonl"):
        m = _FILE.match(f.name)
        if m and (m[1] not in latest or m[2] > latest[m[1]][0]):
            latest[m[1]] = (m[2], f)
    out = {}
    for task in tasks:
        if task in latest:
            out[task] = [(latest[task][1], 0)]
        else:
            facets = sorted(t for t in latest if t.startswith(task + "_"))
            if facets:
                out[task] = [(latest[t][1], FACET_STRIDE * k) for k, t in enumerate(facets)]
    return out


def extract(job: tuple[str, int, list[str]]) -> tuple[pd.DataFrame, list[dict]]:
    """One checkpoint dir: the rows of every requested task, and its manifest lines."""
    model, step, tasks = job
    frames, manifest = [], []
    for task, files in files_for(EVAL_LOGS / f"{model}-iter{step}", tasks).items():
        rows = [(offset + r[0], *r[1:]) for path, offset in files for r in map(parse_record, open(path))]
        df = pd.DataFrame(rows, columns=COLS[3:])
        frames.append(df.assign(model=model, step=step, task=task))
        manifest.append({"model": model, "step": step, "task": task, "source": str(files[0][0]),
                         "n_files": len(files), "n_records": len(df)})
    out = pd.concat(frames)[COLS] if frames else pd.DataFrame(columns=COLS)
    return out.astype({"step": "int32", "doc_id": "int32", "acc": "float32", "acc_norm": "float32",
                       "ll_gold": "float32", "margin": "float32"}), manifest


def flush(frames: list[pd.DataFrame], manifest: list[dict], out_dir: Path) -> None:
    """Append one part file per family and the manifest lines."""
    if not frames:
        return
    df = pd.concat(frames)
    tag = f"{int(time.time())}-{os.getpid()}"
    for fam, g in df.groupby(df["task"].map(benchmark_family)):
        d = out_dir / f"{fam}.parquet"
        d.mkdir(parents=True, exist_ok=True)
        g.to_parquet(d / f"part-{tag}.parquet", index=False)
    m = pd.DataFrame(manifest)
    path = out_dir / "manifest.csv"
    m.to_csv(path, mode="a", header=not path.exists(), index=False)
    frames.clear(); manifest.clear()


def main(pool: str, workers: int, limit_models: int, limit_tasks: int, flush_every: int) -> None:
    df = ladder_frame(pool)
    df = df[(df["kind"] == "benchmark") & (on_shared_grid(df) | on_noise_grid(df))]
    out_dir = STORE / pool
    if limit_models:          # smoke test: the first models, ONE checkpoint dir each, the first tasks
        models = sorted(set(df["model"]))[:limit_models]
        df = df[df["model"].isin(models)]
        df = df[df["step"] == df.groupby("model")["step"].transform("max")]
    if limit_tasks:
        df = df[df["task"].isin(sorted(set(df["task"]))[:limit_tasks])]
    todo = df[["model", "step", "task"]].drop_duplicates()
    if (out_dir / "manifest.csv").exists():
        done = pd.read_csv(out_dir / "manifest.csv")[["model", "step", "task"]]
        todo = todo.merge(done, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    jobs = [(m, int(s), sorted(g["task"])) for (m, s), g in todo.groupby(["model", "step"])]
    print(f"{pool}: {len(todo)} (model, step, task) to extract over {len(jobs)} checkpoint dirs "
          f"({df[['model', 'step', 'task']].drop_duplicates().shape[0] - len(todo)} already in the store) -> {out_dir}")
    frames, manifest, t0, n_rec, n_files = [], [], time.time(), 0, 0
    with Pool(workers) as p:
        for k, (frame, lines) in enumerate(p.imap_unordered(extract, jobs), 1):
            frames.append(frame); manifest.extend(lines)
            n_rec += len(frame); n_files += sum(l["n_files"] for l in lines)
            if k % flush_every == 0 or k == len(jobs):
                flush(frames, manifest, out_dir)
                dt = time.time() - t0
                print(f"  {k}/{len(jobs)} dirs, {n_files} files, {n_rec} records, {dt:.0f} s "
                      f"({n_files / dt:.1f} files/s, {n_rec / dt:.0f} records/s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="predictivity_seeds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-models", type=int, default=0)
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--flush-every", type=int, default=16, help="checkpoint dirs per parquet part")
    a = ap.parse_args()
    main(a.pool, a.workers, a.limit_models, a.limit_tasks, a.flush_every)
