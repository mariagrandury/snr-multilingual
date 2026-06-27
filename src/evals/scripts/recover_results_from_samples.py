#!/usr/bin/env python3.11
"""Recover a results_*.json from per-doc samples_*.jsonl for eval runs that
died mid-batch (BATCH_TASKS=1 timeout) and never wrote their aggregate.

lm_eval --log_samples writes one samples_<task>_<ts>.jsonl per finished task,
each record carrying its `metrics` list and the per-doc value as a top-level
key (e.g. "exact_match": 1.0), plus a `filter`. The task aggregate is the mean
of those per-doc values, keyed "<metric>,<filter>" — the exact shape
results_io.collect()/flatten() consume. So combining the samples reconstructs a
valid results file with no recompute.

Per `eval_*/` dir: if a results_*.json ALREADY exists → skip (never overwrite);
else aggregate its samples → write results_recovered_<dir>.json in place, where
collect()/push_all_results.py will pick it up.

  python3.11 scripts/recover_results_from_samples.py [--root DIR] [--dry-run]
"""
import json, re, sys, statistics as st
from pathlib import Path
from collections import defaultdict

ROOT = Path("/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs")
DRY = False
a = sys.argv[1:]
for i, x in enumerate(a):
    if x == "--root" and i + 1 < len(a): ROOT = Path(a[i + 1])
    if x == "--dry-run": DRY = True

TASK_RE = re.compile(r"^samples_(.+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$")

def task_of(p: Path):
    m = TASK_RE.match(p.name)
    return m.group(1) if m else p.stem[len("samples_"):]

def aggregate(eval_dir: Path):
    """samples_*.jsonl in eval_dir -> {task: {"metric,filter": mean}}, n_docs."""
    # acc[task][metric,filter] = list of per-doc values
    acc = defaultdict(lambda: defaultdict(list))
    ndocs = defaultdict(set)
    for sf in eval_dir.glob("samples_*.jsonl"):
        task = task_of(sf)
        try:
            for line in sf.open():
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                filt = d.get("filter", "none") or "none"
                ndocs[task].add(d.get("doc_id"))
                for m in (d.get("metrics") or []):
                    v = d.get(m)
                    if isinstance(v, bool) or not isinstance(v, (int, float)): continue
                    acc[task][f"{m},{filt}"].append(float(v))
        except (json.JSONDecodeError, OSError) as e:
            print(f"    WARN {sf.name}: {e}", file=sys.stderr)
    results = {}
    for task, mk in acc.items():
        row = {"alias": task}
        for key, vals in mk.items():
            if vals: row[key] = st.fmean(vals)
        if len(row) > 1:                      # had at least one metric
            results[task] = row
    return results, {t: len(s) for t, s in ndocs.items()}

scanned = recovered = have_results = no_samples = 0
total_tasks = 0
for eval_dir in sorted(ROOT.glob("*/*/*/harness/eval_*")):
    if not eval_dir.is_dir(): continue
    scanned += 1
    if any(eval_dir.glob("results_*.json")):
        have_results += 1; continue
    if not any(eval_dir.glob("samples_*.jsonl")):
        no_samples += 1; continue
    results, ndocs = aggregate(eval_dir)
    if not results:
        no_samples += 1; continue
    name = eval_dir.parent.parent.name
    out = eval_dir / f"results_recovered_{eval_dir.name}.json"
    total_tasks += len(results)
    print(f"  {'[dry] ' if DRY else ''}{name}/{eval_dir.name}: {len(results)} tasks "
          f"(docs/task med={int(st.median(list(ndocs.values()))) if ndocs else 0}) -> {out.name}")
    if not DRY:
        out.write_text(json.dumps({"results": results}, indent=1))
    recovered += 1

print(f"\nscanned={scanned}  recovered={recovered}  already_have_results={have_results}  "
      f"no_usable_samples={no_samples}  total_tasks_recovered={total_tasks}{'  (DRY-RUN)' if DRY else ''}")
