"""Build the per-item store the rq08 per-item analysis reads.

For every (model, step, task) row of the ladder frame — the checkpoints on the
shared grid and the noise-grid points (rules 3 and 4), the parent tasks the
cell trains (rules 2 and 6) — read the latest ``samples_<task>_*.jsonl`` under
``<cell>-iter<N>/harness/eval_*/`` (a checkpoint has several eval dirs from
the top-ups; the union is scanned and the newest file per task wins) and keep
one row per item:

    model, step, task, doc_id:int32, acc, acc_norm, ll_gold, margin, bytes_gold (float32)

``doc_id`` is lm-eval's positional id, stable across checkpoints. ``ll_gold``
is the log-likelihood of the target choice (``resps`` per choice, ``target``
its index — a digit, or a letter for the MMLU-style tasks) and ``margin`` is
that minus the best other choice; ``bytes_gold`` is the UTF-8 length of the
target's continuation (``arguments.gen_args_<target>.arg_1``, the harness's
``acc_bytes`` length), so ``-ll_gold / ln2 / bytes_gold`` is the item's
bits-per-byte on the gold answer; all three NaN where the record is not a
multiple-choice one. A parent with no samples file of its own (Global-MMLU,
INCLUDE, mmlu: the harness writes one file per subject) is the union of its
``samples_<parent>_<subject>_*`` files with ``doc_id = FACET_STRIDE * rank +
doc_id``, the rank being the subject's position in the sorted subject list —
stable because every checkpoint runs the same subjects.

Parquet per benchmark family, ``per_item_store/<pool>/<family>.parquet/`` (a
directory of part files, one per flush, so a killed job keeps what it wrote),
plus ``manifest.csv``; a (model, step, task) in the manifest is skipped on the
next run. The folder is data (git-ignored), not a figure.

Records are 3-5 KB and only six fields are read: the head (doc_id) and tail
(acc, acc_norm) are regexes and ``target`` / ``arguments`` / ``resps`` are sliced by index and
json-decoded alone, which is 1.4x faster than ``json.loads`` of the record; a
line the regexes cannot read is decoded whole.

    python analysis/rq08_subset_selection/build_per_item_store.py --pool predictivity_seeds --workers 64
    python ... --limit-models 1 --limit-tasks 3          # smoke test, one checkpoint dir per model
    python ... --pool predictivity_schemes --finals-only --families rf_belebele,hellaswag   # bench-BPB DA
    python ... --bench-bpb            # only (re)write utils.BENCH_BPB from the BBPB_POOL folder of the store

``--bench-bpb`` reduces the store to one bits-per-byte value per (model, step,
task), the item mean of ``-ll_gold / ln2 / bytes_gold`` (`bench_bpb`), which
the loader turns into each benchmark's `bbpb_` twin (utils.with_bbpb_twins).
Without a store it writes nothing, so the committed table stays.
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
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.paths import SUBSET_SELECTION  # noqa: E402
from analysis.utils import BBPB, BENCH_BPB, benchmark_family, ladder_frame, on_noise_grid, on_shared_grid  # noqa: E402
from pretrain.ladder_report import EVAL_LOGS  # noqa: E402

STORE = SUBSET_SELECTION / "per_item_store"
BBPB_POOL = "predictivity_schemes"   # the one store folder bench_bpb.csv is reduced from (others may predate bytes_gold)
FACET_STRIDE = 10_000        # doc_id offset per subject file of a parent without its own file
COLS = ["model", "step", "task", "doc_id", "acc", "acc_norm", "ll_gold", "margin", "bytes_gold"]
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
        j = line.index(', "arguments": {"gen_args_'); i = line.rindex('"target": ', 0, j)   # a doc may hold its own "target"
        a = line.index('"resps": ', j); b = line.index(', "filtered_resps"', a)
        doc_id, target, resps = int(h[1]), json.loads(line[i + 10:j]), json.loads(line[a + 9:b])
        args = json.loads(line[j + 15:a - 2])
        acc, acc_norm = float(t[1]), float(t[2]) if t[2] else np.nan
    else:
        o = json.loads(line)
        doc_id, target, resps, args = o["doc_id"], o["target"], o["resps"], o.get("arguments")
        acc, acc_norm = o.get("acc", np.nan), o.get("acc_norm", np.nan)
    ll_gold = margin = bytes_gold = np.nan
    g = gold_index(target)
    if resps and isinstance(resps[0][0], list) and 0 <= g < len(resps):   # loglikelihood per choice
        ll = np.array([float(r[0][0]) for r in resps])
        ll_gold = ll[g]
        if len(ll) > 1:
            margin = ll_gold - np.max(np.delete(ll, g))
        bytes_gold = len(args[f"gen_args_{g}"]["arg_1"].encode("utf-8"))
    return doc_id, acc, acc_norm, ll_gold, margin, bytes_gold


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
                       "ll_gold": "float32", "margin": "float32", "bytes_gold": "float32"}), manifest


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


def bench_bpb(parts) -> pd.DataFrame:
    """Per (model, step, task) of the store `parts`: the item-mean bits per
    byte of the gold answer (`bbpb`) and the mean gold length (`bytes_gold`);
    a task with no multiple-choice record (no bytes_gold) has none."""
    store = pd.concat(pd.read_parquet(d, columns=["model", "step", "task", "ll_gold", "bytes_gold"]) for d in parts)
    store["bbpb"] = -store["ll_gold"] / np.log(2) / store["bytes_gold"]
    return store.groupby(["model", "step", "task"], as_index=False)[["bbpb", "bytes_gold"]].mean().dropna(subset=["bbpb"])


def write_bench_bpb() -> None:
    parts = sorted((STORE / BBPB_POOL).glob("*.parquet"))
    if not parts:
        print(f"no per-item store at {STORE / BBPB_POOL}: {BENCH_BPB.name} not written (build_per_item_store.sbatch builds it)")
        return
    t = bench_bpb(parts)[["model", "step", "task", "bbpb"]]
    t.sort_values(["model", "step", "task"]).to_csv(BENCH_BPB, index=False)
    print(f"wrote {BENCH_BPB}: {len(t)} (model, step, task), {t['model'].nunique()} models, {t['task'].nunique()} tasks")


def main(pool: str, workers: int, limit_models: int, limit_tasks: int, flush_every: int,
         finals_only: bool, families: list[str]) -> None:
    df = ladder_frame(pool)
    df = df[(df["kind"] == "benchmark") & ~df["task"].str.startswith(BBPB)        # a bbpb_ twin has no samples of its own
            & (on_shared_grid(df) | on_noise_grid(df))]
    out_dir = STORE / pool
    if families:
        df = df[df["task"].map(benchmark_family).isin(families)]
    if limit_models:          # smoke test: the first models, ONE checkpoint dir each, the first tasks
        df = df[df["model"].isin(sorted(set(df["model"]))[:limit_models])]
    if limit_models or finals_only:
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
    ap.add_argument("--finals-only", action="store_true", help="only each model's last checkpoint")
    ap.add_argument("--families", type=lambda s: s.split(","), default=[],
                    help="comma-separated benchmark families (benchmark_family) to extract")
    ap.add_argument("--bench-bpb", action="store_true", help="only write the bbpb table from the store, extract nothing")
    a = ap.parse_args()
    if a.bench_bpb:
        write_bench_bpb()
    else:
        main(a.pool, a.workers, a.limit_models, a.limit_tasks, a.flush_every, a.finals_only, a.families)
