#!/usr/bin/env python3
"""Collect eval results on disk and push one W&B run per *model*.

Groups every NAME under <eval_logs>/<entity>/<project>/ by its model
(stripping `-iter<N>`, `-step<N>`, `-stage<K>-step-?<N>`, `-main`, etc.),
then pushes one resumeable W&B run per model. Default chart axes:
x = FLOPs (≈ 6 × params × tokens), y = metric value clamped to [0, 1].

Each ckpt is logged as a step:

  for each ckpt: run.log({
      "iter": <N>, "tokens": T, "flops": F,
      "<task>/acc": v,
      "<task>/exact_match": v,   # mgsm-style, only when no acc
      ...
  })

`define_metric("*", step_metric="flops")` makes flops the default x for
every chart on the W&B workspace, each with one line per model. The
`iter` and `tokens` axes are also defined and can be swapped in via
the W&B UI (Edit panel → X-axis).

Per-task metric: exactly one — prefer `acc`, fall back to `exact_match`,
skip the task otherwise (see `flatten`). Subtopic tasks (e.g.
`mmlu_anatomy`, `global_mmlu_full_zh_stem`) collapse into their parent
aggregate (e.g. `mmlu`, `global_mmlu_full_zh`) when the parent is
present (see `aggregate_parents`).

After bulk push, a saved workspace view is created with one LinePlot
per benchmark, x=flops, range_y=(0, 1). Requires the optional
`wandb-workspaces` package; gracefully skipped if missing.

Idempotent — W&B run id is `<model>` (sanitised), so re-runs resume the
same run and accumulate new ckpts. Re-logging the same step appends a
duplicate point at the same x value; the chart simply overlays them.

Two modes:

  Bulk rescue (login node, `snr` conda env has wandb):
    python scripts/push_all_results.py [--dry-run] [--filter REGEX]

  Single-NAME (called from evaluate.sbatch after each successful eval —
  runs in the pyxis container, which has internet via proxy):
    python scripts/push_all_results.py --name <NAME> [--eval-duration <s>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Shared loader + lifted I/O helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from configs import get_model, load_models, metric_for, tokens_for  # noqa: E402
from results_io import aggregate_parents, collect, flatten  # noqa: E402

LOGS_BASE = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
)
ENTITY = "mariagrandury-epflnlp"
PROJECT = "snr-experiments"

# --- name → (model, ckpt) parsing -----------------------------------------
# Tokens / params come from configs/models.json (see configs.tokens_for and
# configs.get_model). `flatten` lives in utils/results_io.py; per-task
# metric override comes from configs.metric_for.

_NAME_RE_MEG = re.compile(
    r"^(?P<model>.+)-iter(?P<n>\d+)$"
)
_NAME_RE_HF_STAGE = re.compile(
    r"^(?P<model>.+)-(?P<branch>(?:stage|step|main)[A-Za-z0-9._\-]*)$"
)


def parse_name(name: str) -> dict | None:
    """NAME → {model, step, tokens}. Reads tokens/params from configs/models.json.

    Returns None if the NAME isn't recognised or the model isn't declared
    in configs/models.json (we can't compute tokens without the JSON entry).
    """
    models = load_models()

    # Megatron iter form: <model>-iter<N>
    m = _NAME_RE_MEG.match(name)
    if m and m.group("model") in models:
        model = m.group("model")
        step = int(m.group("n"))
        return {"model": model, "step": step, "tokens": tokens_for(model, step)}

    # HF branch form: <model>-<branch>. Strip the model name as the longest
    # prefix that's in models.json; the remainder is the branch.
    # (Handles names like "SmolLM3-3B-checkpoints-stage1-step-3440000".)
    for model in sorted(models, key=len, reverse=True):
        if name == model:
            # bare model name = main branch (None tokens if the model has
            # no `main` checkpoint, e.g. a multi-stage checkpoints repo).
            try:
                tokens = tokens_for(model, "main")
            except KeyError:
                tokens = None
            return {"model": model, "step": 0, "tokens": tokens}
        prefix = model + "-"
        if name.startswith(prefix):
            branch = name[len(prefix):]
            try:
                tokens = tokens_for(model, branch)
            except (KeyError, TypeError):
                # KeyError: branch not declared as a canonical ckpt.
                # TypeError: caller fell through to a megatron_iter model
                # with a non-numeric branch suffix (e.g. `iter42000-vllmcheck`
                # for a one-off sanity eval) — that NAME doesn't correspond
                # to a real canonical ckpt; let the outer loop skip it.
                continue
            # `step` is the numeric tail of the branch name where present,
            # else 0 (single-branch refs).
            step_match = re.search(r"-(\d+)$", branch)
            step = int(step_match.group(1)) if step_match else 0
            return {"model": model, "step": step, "tokens": tokens}
    return None


def model_params(model: str) -> int | None:
    """Read `params` straight from configs/models.json. None if model isn't
    declared (FLOPs chart will be empty for that model)."""
    try:
        return get_model(model).get("params")
    except KeyError:
        return None


def _flatten_with_overrides(scores: dict[str, dict]) -> dict[str, float]:
    """Wrap results_io.flatten with the per-task metric override from
    configs/tasks.json (e.g. ifeval → exact_match)."""
    return flatten(scores, task_metric_override=metric_for)


RUN_ID_SUFFIX = "-v6"   # bump if W&B blacklists existing IDs (409 on re-create after delete)


def push_one(model: str, params: int | None,
             entries: list[tuple[int, int, dict[str, float]]],
             entity: str, project: str):
    """Open/resume one W&B run for `model` and log each (step, tokens, flat).

    One metric per task (`<task>/acc` or `<task>/exact_match`, see
    `flatten`). Default x-axis is `flops` (= 6 × params × tokens) — the
    compute-fair view across model sizes. `iter` and `tokens` are also
    logged so the chart's x-axis can be swapped in the W&B UI.
    """
    import wandb
    wb_id = re.sub(r"[^A-Za-z0-9_-]+", "_", model)[:128] + RUN_ID_SUFFIX
    run = wandb.init(
        entity=entity,
        project=project,
        name=model,
        id=wb_id,
        resume="allow",
        reinit=True,
        config={"model": model, "params": params},
        settings=wandb.Settings(init_timeout=300),
    )

    # Axes available for every chart's x. Default x = flops.
    run.define_metric("iter")
    run.define_metric("tokens")
    run.define_metric("flops")
    run.define_metric("eval_duration_seconds")
    run.define_metric("*", step_metric="flops")

    for step, tokens, flat in entries:
        flops = 6 * params * tokens if (params and tokens) else None
        duration = flat.get("eval_duration_seconds")

        log: dict[str, float] = {"iter": step, "tokens": tokens}
        if flops is not None:
            log["flops"] = flops
        if duration is not None:
            log["eval_duration_seconds"] = duration

        for k, v in flat.items():
            if k == "eval_duration_seconds":
                continue
            log[k] = v

        # One log call per ckpt → each metric's history has exactly N points
        # (one per ckpt). With ≥ 2 points wandb auto-renders as a line plot.
        run.log(log)

    n_keys = sum(1 for _, _, m in entries for k in m if k != "eval_duration_seconds")
    suffix = "" if params else " (no params known → flops chart will be empty)"
    print(f"  pushed {model}: {len(entries)} ckpt(s), {n_keys} metric value(s){suffix} → {run.url}")
    run.finish()


def setup_workspace(entity: str, project: str, metrics: set[str]) -> None:
    """Create/update a saved view 'flops vs metric (y∈[0,1])' with one
    LinePlot per `<task>/<metric>` key, x=flops, y-axis clamped to [0, 1].

    Skipped silently if `wandb-workspaces` is not installed or the API call
    fails (e.g. no internet from the calling host)."""
    if not metrics:
        return
    try:
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
    except ModuleNotFoundError:
        print("(wandb-workspaces not installed → skipping y∈[0,1] workspace setup)")
        return

    sections = [
        ws.Section(
            name=metric.split("/", 1)[0],
            panels=[wr.LinePlot(title=metric, x="flops", y=[metric], range_y=(0.0, 1.0))],
        )
        for metric in sorted(metrics)
    ]
    workspace = ws.Workspace(
        entity=entity, project=project,
        name="flops vs metric (y in [0,1])",
        sections=sections, auto_generate_panels=False,
    )
    try:
        workspace.save()
    except Exception as e:
        print(f"(workspace setup failed: {e!r} — runs are still pushed; configure y-axis manually in UI)")
        return
    print(f"  saved workspace 'flops vs metric (y in [0,1])' "
          f"with {len(sections)} panels → https://wandb.ai/{entity}/{project}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--name", help="Single-NAME mode: push only this NAME's results (used by evaluate.sbatch).")
    p.add_argument("--eval-duration", type=int, default=None,
                   help="Single-NAME mode only: seconds the eval took. Logged as "
                        "`tokens/eval_duration_seconds` and `flops/eval_duration_seconds` so "
                        "the duration shares the same axes as the score charts.")
    p.add_argument("--dry-run", action="store_true", help="Preview without pushing.")
    p.add_argument("--filter", help="Bulk mode only: regex on NAME — only matching NAMEs are pushed.")
    args = p.parse_args()

    project_dir = LOGS_BASE / args.entity / args.project
    if not project_dir.is_dir():
        sys.exit(f"No project dir at {project_dir}")

    # Single-NAME mode: just this one ckpt.
    if args.name:
        flat = _flatten_with_overrides(aggregate_parents(collect(project_dir / args.name)))
        if not flat:
            sys.exit(f"No results found for {args.name}")
        parsed = parse_name(args.name)
        if parsed is None:
            sys.exit(f"Unparseable NAME (or no token mapping): {args.name}")
        model, step, tokens = parsed["model"], parsed["step"], parsed["tokens"]
        params = model_params(model)
        if args.eval_duration is not None:
            flat["eval_duration_seconds"] = float(args.eval_duration)
        flops = (6 * params * tokens) if (params and tokens) else None
        flops_str = f", flops={flops:.2e}" if flops else " (flops unknown)"
        tokens_str = f"tokens={tokens:.2e}" if tokens else "tokens=?"
        print(f"Will push 1 model to {args.entity}/{args.project}: "
              f"{model} @ step={step}, {tokens_str}{flops_str}, {len(flat)} metrics")
        if args.dry_run:
            print("(dry-run) — not pushing.")
            return
        push_one(model, params, [(step, tokens, flat)], args.entity, args.project)
        return

    # Bulk mode: every NAME with results, grouped by model.
    pat = re.compile(args.filter) if args.filter else None
    grouped: dict[str, list[tuple[int, int, dict[str, float]]]] = defaultdict(list)
    skipped: list[str] = []

    all_metrics: set[str] = set()
    for name_dir in sorted(project_dir.iterdir()):
        if not name_dir.is_dir():
            continue
        if pat and not pat.search(name_dir.name):
            continue
        flat = _flatten_with_overrides(aggregate_parents(collect(name_dir)))
        if not flat:
            continue
        parsed = parse_name(name_dir.name)
        if parsed is None:
            skipped.append(name_dir.name)
            continue
        grouped[parsed["model"]].append((parsed["step"], parsed["tokens"], flat))
        all_metrics.update(k for k in flat if k != "eval_duration_seconds")

    for model in grouped:
        grouped[model].sort(key=lambda e: e[0])

    if not grouped:
        print("No NAMEs with results found.")
        return

    print(f"Will push {len(grouped)} model(s) to {args.entity}/{args.project}:")
    for model, entries in sorted(grouped.items()):
        params = model_params(model)
        steps = [s for s, _, _ in entries]
        toks = [t for _, t, _ in entries if t is not None]
        tok_str = (f"tokens ∈ [{min(toks):.2e}, {max(toks):.2e}]"
                   if toks else "tokens=?")
        n_metrics = sum(len(m) for _, _, m in entries)
        params_str = f"params={params:.2e}" if params else "params=?"
        print(f"  {model}: {len(entries)} ckpt(s) at steps {steps}, "
              f"{tok_str}, {params_str}, {n_metrics} metrics")
    if skipped:
        print(f"  skipped ({len(skipped)} unparseable NAME(s)): {skipped}")

    if args.dry_run:
        print("\n(dry-run) — not pushing.")
        return

    for model, entries in sorted(grouped.items()):
        push_one(model, model_params(model), entries, args.entity, args.project)

    setup_workspace(args.entity, args.project, all_metrics)

    print(f"\nDone. View at: https://wandb.ai/{args.entity}/{args.project}")


if __name__ == "__main__":
    main()
