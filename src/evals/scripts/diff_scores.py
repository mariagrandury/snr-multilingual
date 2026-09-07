#!/usr/bin/env python3.11
"""Compare every metric of one checkpoint between two W&B projects (= two
eval_logs subtrees). Used to prove a pipeline change did not move scores.

    python3.11 diff_scores.py <NAME> <project_a> <project_b>
"""
import glob, json, sys

ROOT = ("/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM"
        "/logs/eval_logs/mariagrandury-epflnlp")

def load(project, name):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/{project}/{name}/harness/eval_*/**/results_*.json",
                              recursive=True)):
        for task, metrics in (json.load(open(f)).get("results") or {}).items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not k.endswith("_stderr,none"):
                    out[f"{task}/{k}"] = v
    return out

name, pa, pb = sys.argv[1], sys.argv[2], sys.argv[3]
a, b = load(pa, name), load(pb, name)
common = sorted(set(a) & set(b))
print(f"{pa}: {len(a)} metrics | {pb}: {len(b)} metrics | comparable: {len(common)}")
if not common:
    raise SystemExit("nothing to compare")
diffs = [(abs(a[k] - b[k]), k) for k in common]
moved = [(d, k) for d, k in diffs if d > 1e-9]
print(f"identical: {len(common) - len(moved)}/{len(common)}   max |delta|: {max(diffs)[0]:.2e}")
for d, k in sorted(moved, reverse=True)[:15]:
    print(f"  {d:>10.4f}  {k}   {a[k]:.4f} -> {b[k]:.4f}")
