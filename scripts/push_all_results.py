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

LOGS_BASE = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs"
)
ENTITY = "mariagrandury-epflnlp"
PROJECT = "snr-experiments"

# --- Tokens-per-step lookup tables ---------------------------------------
# Megatron training config (see ../../pretrain/submit-apertus-data-mix.sh):
# global_batch_size = 504, sequence_length = 4096
MEG_TOKENS_PER_ITER = 504 * 4096  # ≈ 2.064 M

# HF reference model stages → cumulative tokens at the *end* of each stage.
# We only eval the last ckpt of each stage, so this is the value at that ckpt.
# Numbers are best-effort from each provider's docs; refine when exact figures land.
HF_STAGE_TOKENS = {
    ("Olmo-3-1025-7B", 1): 6_000_000_000_000,    # ~6 T  (long pretrain)
    ("Olmo-3-1025-7B", 2): 6_190_000_000_000,    # +~190 B mid-training mix
    ("Olmo-3-1025-7B", 3): 6_240_000_000_000,    # +~50 B annealing
    ("SmolLM3-3B-checkpoints", 1): 7_200_000_000_000,
    ("SmolLM3-3B-checkpoints", 2): 8_800_000_000_000,
    ("SmolLM3-3B-checkpoints", 3): 9_900_000_000_000,
}

# `<repo>-main` → cumulative tokens at the published checkpoint.
HF_MAIN_TOKENS = {
    "Apertus-8B-2509": 15_000_000_000_000,    # 15 T per Apertus card
    "Apertus-70B-2509": 15_000_000_000_000,
}


def model_params(model: str) -> int | None:
    """Approximate total params from the model display name (the value parse_name
    uses as run name). Used for FLOPs ≈ 6 × params × tokens.

    Recognises:
      - apertus-<size>-fwEdu...                 → <size>
      - <repo>-stage<K> / <repo> with `-NB-`    → NB
    """
    m = re.match(r"^apertus-(\d+)([MB])-", model, re.IGNORECASE)
    if m:
        return int(m.group(1)) * (10**6 if m.group(2).upper() == "M" else 10**9)
    # Match any "-NB-" or trailing "-NB" anywhere in the name (Apertus-8B-2509,
    # SmolLM3-3B-checkpoints-stage1, Olmo-3-1025-7B-stage1, …).
    m = re.search(r"-(\d+)B(?:-|$)", model)
    if m:
        return int(m.group(1)) * 10**9
    return None


def parse_name(name: str) -> dict | None:
    """NAME → {model, step, tokens}. Returns None if unparseable or if we
    don't know how to translate the NAME to a token count.

    - apertus-<size>-fwEdu<X>-fw<Y>-seed<S>-iter<N>
        → step=N, tokens = N × MEG_TOKENS_PER_ITER
    - <repo>-stage<K>-step-?<N>(-...)?
        → step=N, tokens = HF_STAGE_TOKENS[(repo, K)]  (None ⇒ skip)
    - <repo>-step<N>-tokens<M>(B|T)
        → step=N, tokens = M × {1e9, 1e12}
    - <repo>-main
        → step=0, tokens = HF_MAIN_TOKENS[repo]        (None ⇒ skip)
    """
    m = re.match(r"^(?P<model>apertus-\d+[MB]-fwEdu\d+-fw\d+-seed\d+)-iter(?P<n>\d+)$", name)
    if m:
        n = int(m.group("n"))
        return {"model": m.group("model"), "step": n, "tokens": n * MEG_TOKENS_PER_ITER}
    m = re.match(r"^(?P<repo>.+?)-stage(?P<stage>\d+)-step-?(?P<n>\d+)(?:-.*)?$", name)
    if m:
        repo, stage = m.group("repo"), int(m.group("stage"))
        tokens = HF_STAGE_TOKENS.get((repo, stage))
        if tokens is None:
            return None
        return {"model": f"{repo}-stage{stage}", "step": int(m.group("n")), "tokens": tokens}
    m = re.match(r"^(?P<repo>.+?)-step(?P<n>\d+)-tokens(?P<mag>[\d.]+)(?P<unit>[BT])$", name)
    if m:
        unit = 1e9 if m.group("unit") == "B" else 1e12
        tokens = int(float(m.group("mag")) * unit)
        return {"model": m.group("repo"), "step": int(m.group("n")), "tokens": tokens}
    m = re.match(r"^(?P<repo>.+?)-main$", name)
    if m:
        repo = m.group("repo")
        tokens = HF_MAIN_TOKENS.get(repo)
        if tokens is None:
            return None
        return {"model": repo, "step": 0, "tokens": tokens}
    return None


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
    and `mmlu_anatomy` are present, only `mmlu` survives. Same for
    `global_mmlu_full_zh` vs `global_mmlu_full_zh_stem`,
    `global_mmlu_full_zh_humanities`, …

    Singleton benchmarks with no parent (`arc_challenge`, `belebele_eng_Latn`,
    `hellaswag`) pass through untouched.
    """
    keys = set(scores)

    def has_parent(task: str) -> bool:
        parts = task.split("_")
        for i in range(len(parts) - 1, 0, -1):
            if "_".join(parts[:i]) in keys:
                return True
        return False

    return {t: v for t, v in scores.items() if not has_parent(t)}


def flatten(scores: dict[str, dict]) -> dict[str, float]:
    """{task: {'metric,filter': val}} → {'task/metric': val}.

    Exactly one metric per task: prefer `acc`; fall back to `exact_match`
    (mgsm-style); skip the task otherwise. Stderr / acc_norm / acc_bytes /
    degeneration are intentionally dropped — the workspace charts only show
    the headline number per benchmark.
    """
    out: dict[str, float] = {}
    for task, metrics in scores.items():
        if not isinstance(metrics, dict):
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
            (k for k in metrics if k.split(",", 1)[0].strip() == "exact_match"),
            None,
        )
        if em_key is not None:
            v = metrics[em_key]
            if isinstance(v, (int, float)):
                out[f"{task}/exact_match"] = float(v)
    return out


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
        flops = 6 * params * tokens if params else None
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
        flat = flatten(aggregate_parents(collect(project_dir / args.name)))
        if not flat:
            sys.exit(f"No results found for {args.name}")
        parsed = parse_name(args.name)
        if parsed is None:
            sys.exit(f"Unparseable NAME (or no token mapping): {args.name}")
        model, step, tokens = parsed["model"], parsed["step"], parsed["tokens"]
        params = model_params(model)
        if args.eval_duration is not None:
            flat["eval_duration_seconds"] = float(args.eval_duration)
        flops = (6 * params * tokens) if params else None
        flops_str = f", flops={flops:.2e}" if flops else " (params unknown)"
        print(f"Will push 1 model to {args.entity}/{args.project}: "
              f"{model} @ step={step}, tokens={tokens:.2e}{flops_str}, {len(flat)} metrics")
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
        flat = flatten(aggregate_parents(collect(name_dir)))
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
        tok_lo = min(t for _, t, _ in entries)
        tok_hi = max(t for _, t, _ in entries)
        n_metrics = sum(len(m) for _, _, m in entries)
        params_str = f"params={params:.2e}" if params else "params=?"
        print(f"  {model}: {len(entries)} ckpt(s) at steps {steps}, "
              f"tokens ∈ [{tok_lo:.2e}, {tok_hi:.2e}], {params_str}, {n_metrics} metrics")
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
