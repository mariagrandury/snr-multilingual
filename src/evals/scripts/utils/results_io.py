"""Eval-results I/O helpers lifted out of push_all_results.py.

`collect`, `aggregate_parents`, and `flatten` are imported by both
push_all_results.py (in this repo) and signal-and-noise's
snr/download/apertus.py (cluster-only). Lifting them here gives the
implicit cross-repo sys.path import a stable home.

The bodies are byte-equivalent to push_all_results.py's originals (see
.claude-shared/plans/models-tasks-json-refactor.md R1 for why).
"""

from __future__ import annotations

import json
from pathlib import Path


def collect(name_dir: Path) -> dict[str, dict]:
    """Union of every results_*.json under harness/. Pulls both `results`
    (per-task scores) and `groups` (aggregate scores like `mmlu` that the
    merge step strips). Merged files take precedence over per-task partials
    for the same task name."""
    scores: dict[str, dict] = {}

    def merge_file(path: Path, override: bool):
        try:
            data = json.loads(path.read_text())
        except Exception:
            return
        for source in ("results", "groups"):
            for k, v in (data.get(source) or {}).items():
                if isinstance(v, dict) and (override or k not in scores):
                    scores[k] = v

    base = name_dir / "harness"
    if not base.is_dir():
        return scores
    for f in sorted(base.glob("eval_*/results_*.json")):
        merge_file(f, override=True)
    for f in sorted(base.glob("eval_*/per_task/*/*/results_*.json")):
        merge_file(f, override=False)
    return scores


def aggregate_parents(scores: dict[str, dict]) -> dict[str, dict]:
    """Drop subtopic tasks if their parent aggregate is also in `scores`.

    A task `T` is a subtopic of `P` iff `P` is an underscore-prefix of `T`
    AND `P` is itself in `scores`. So if both `mmlu` (aggregate from `groups`)
    and `mmlu_anatomy` are present, only `mmlu` survives.
    """
    keys = set(scores)

    def has_parent(task: str) -> bool:
        parts = task.split("_")
        for i in range(len(parts) - 1, 0, -1):
            if "_".join(parts[:i]) in keys:
                return True
        return False

    return {t: v for t, v in scores.items() if not has_parent(t)}


def flatten(scores: dict[str, dict],
            task_metric_override=None) -> dict[str, float]:
    """{task: {'metric,filter': val}} → {'task/metric': val}.

    One metric per task. Lookup order:
      1. If `task_metric_override(task)` returns a string, use that metric.
         (Wires to configs/tasks.json `metric` field via
         `from src.evals.scripts.utils.configs import metric_for`.)
      2. Else prefer `acc`; fall back to `exact_match`; skip otherwise.

    `task_metric_override` is a callable for two reasons: tests stub it
    cheaply, and the configs lookup is lazy (the cluster-side
    push_all_results.py opts in by passing
    `task_metric_override=metric_for`).
    """
    out: dict[str, float] = {}
    for task, metrics in scores.items():
        if not isinstance(metrics, dict):
            continue
        override = task_metric_override(task) if task_metric_override else None
        if override is not None:
            key = next(
                (k for k in metrics
                 if k.split(",", 1)[0].strip() == override),
                None,
            )
            if key is not None:
                v = metrics[key]
                if isinstance(v, (int, float)):
                    out[f"{task}/{override}"] = float(v)
            continue
        acc_key = next(
            (k for k in metrics if k.split(",", 1)[0].strip() == "acc"),
            None,
        )
        if acc_key is not None:
            v = metrics[acc_key]
            if isinstance(v, (int, float)):
                out[f"{task}/acc"] = float(v)
            continue
        em_key = next(
            (k for k in metrics
             if k.split(",", 1)[0].strip() == "exact_match"),
            None,
        )
        if em_key is not None:
            v = metrics[em_key]
            if isinstance(v, (int, float)):
                out[f"{task}/exact_match"] = float(v)
    return out
