#!/usr/bin/env python3
"""Build a HF dataset from local eval_logs and push it to epfl-nlp.

Mirrors the structure of allenai/signal-and-noise: a flat per-suite parquet
with one row per (model, model_revision, task), aggregate metrics only — no
per-instance predictions.

Splits emitted (one per `source` in configs/models.json):
  - pretraining_custom: 36 apertus megatron pretrains (4 sizes × 3 mixes × 3 seeds)
  - pretraining_a06:    a06 main runs (apertus3-{1b,3b}-*-nodes)
  - reference_hf:       swiss-ai-reference + huggingface-reference (Qwen3,
                        gemma-3, SmolLM3, Olmo-3, Apertus-8B/70B-2509)

Usage:
  # Local eval_logs only — writes parquet to --out-dir, no upload (default)
  python scripts/build_hf_dataset.py --out-dir /tmp/snr-hf-dataset

  # Local + remote multilingual-snr raw/, then upload
  python scripts/build_hf_dataset.py --include-multilingual-evals \\
      --push --repo-id multilingual-snr/multilingual-snr-eval-results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Reuse logic from push_all_results.py + shared configs loader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from configs import (  # noqa: E402
    family_of,
    get_model,
    load_hf_wandb_config,
    split_for_source,
    tokens_for,
)
from results_io import collect  # noqa: E402
from push_all_results import LOGS_BASE, model_params, parse_name  # noqa: E402

# W&B coordinates + published-dataset config from configs/hf_wandb.json.
_HF_WANDB_CFG = load_hf_wandb_config()
ENTITY = _HF_WANDB_CFG["wandb"]["entity"]
PROJECT = _HF_WANDB_CFG["wandb"]["project"]
REPO_ID = _HF_WANDB_CFG["repo_id"]

# Tasks for which the result file uses these as the "primary" metric (in order).
PRIMARY_METRIC_PRIORITY = ["acc", "exact_match", "pass@1", "f1"]


# Fixed set of metric fields kept in the `metrics` struct. Tasks that don't
# report a given metric get `None`. This makes the parquet schema stable
# across splits so `datasets.load_dataset` can concatenate them.
METRIC_FIELDS = (
    "acc",
    "acc_stderr",
    "acc_norm",
    "acc_norm_stderr",
    "acc_bytes",
    "acc_bytes_stderr",
    "exact_match",
    "exact_match_stderr",
    "f1",
    "f1_stderr",
    "pass_at_1",
    "perplexity",
    "perplexity_stderr",
    "degeneration",
    "degeneration_stderr",
)


def normalize_metrics(task_metrics: dict) -> dict[str, float | None]:
    """{ 'acc,none': 0.5, 'acc_stderr,none': 0.01, 'alias': 'foo' } →
       { 'acc': 0.5, 'acc_stderr': 0.01, 'acc_norm': None, ... }.

    Returns a dict with exactly METRIC_FIELDS keys; missing values are None.
    Filter-qualified variants ('exact_match,strict-match' vs
    'exact_match,flexible-extract') collapse to the same bare name; the
    first encountered wins.
    """
    out: dict[str, float | None] = {k: None for k in METRIC_FIELDS}
    for key, val in task_metrics.items():
        if key == "alias" or not isinstance(val, (int, float)):
            continue
        metric = key.split(",", 1)[0].strip()
        # Map lm-eval names to our canonical set
        canonical = metric.replace("@", "_at_")
        if canonical in out and out[canonical] is None:
            out[canonical] = float(val)
    return out


def pick_primary(metrics: dict[str, float]) -> tuple[str | None, float | None]:
    """Choose the headline (primary_score, primary_metric) for a task."""
    for m in PRIMARY_METRIC_PRIORITY:
        if m in metrics:
            return m, metrics[m]
    return None, None


_CUSTOM_RE = re.compile(
    r"^apertus-\d+[MB]-(?P<mix>fwEdu\d+-fw\d+)-seed(?P<seed>\d+)-iter\d+$"
)


def name_to_metadata(name: str) -> dict | None:
    """Extract model metadata from a NAME (eval_logs subdir).

    Delegates the NAME → (model, step, tokens) parse to push_all_results
    (which reads models.json + tokens_for). Fills the parquet-specific
    fields (size, params, family, mix, seed, model_revision, split) from
    the model's configs/models.json entry.
    """
    parsed = parse_name(name)
    if parsed is None:
        return None
    model, step, tokens = parsed["model"], parsed["step"], parsed["tokens"]
    try:
        entry = get_model(model)
    except KeyError:
        return None
    src = entry.get("source")
    # `split is None` → source is evaluated but not published; skip the row.
    split = split_for_source(src) if src else None
    if split is None:
        return None

    # mix / seed depend on source. Custom Apertus encodes both in the
    # NAME; a06 has neither; HF reference derives mix from the branch
    # (stage<K> or main).
    if src == "snr-pretraining-custom":
        m = _CUSTOM_RE.match(name)
        mix = m.group("mix") if m else None
        seed = int(m.group("seed")) if m else entry.get("seed")
        revision = f"iter{step}"
    elif src == "snr-pretraining-a06":
        mix = "main"
        seed = entry.get("seed")
        revision = f"iter{step}"
    else:
        # HF: revision = remainder after stripping `<model>-`; bare model = main.
        revision = name[len(model) + 1:] if name.startswith(model + "-") else "main"
        if not revision:
            revision = "main"
        m = re.match(r"^stage(\d+)", revision)
        mix = f"stage{m.group(1)}" if m else "main"
        seed = None

    return {
        "name": name,
        "model": model,
        "model_revision": revision,
        "model_type": "custom_pretraining" if src.startswith("snr-pretraining") else "reference_hf",
        "step": step,
        "tokens": float(tokens) if tokens else None,
        "params": entry.get("params"),
        "size": entry.get("size"),
        "family": entry.get("family"),
        "mix": mix,
        "seed": seed,
        "split": split,
    }


def collect_eval_metadata(name_dir: Path) -> dict:
    """Pull global eval-config from any one results file in the NAME's
    harness/. Used for fields like model_source, model_args, total time.
    """
    base = name_dir / "harness"
    out = {
        "model_source": None,
        "model_path": None,
        "model_args": None,
        "n_shot_per_task": {},
        "n_samples_per_task": {},
        "processing_time_per_task": {},
    }
    if not base.is_dir():
        return out

    candidates = list(base.glob("eval_*/results_*.json")) + list(
        base.glob("eval_*/per_task/*/*/results_*.json")
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if out["model_source"] is None:
            out["model_source"] = data.get("model_source")
        if out["model_args"] is None:
            ma = (data.get("config") or {}).get("model_args") or {}
            out["model_args"] = ma if isinstance(ma, dict) else None
            # `pretrained` for HF/vLLM, `load` for megatron
            if isinstance(ma, dict):
                out["model_path"] = ma.get("pretrained") or ma.get("load")
        for tk, n in (data.get("n-shot") or {}).items():
            out["n_shot_per_task"].setdefault(tk, n)
        for tk, info in (data.get("n-samples") or {}).items():
            if isinstance(info, dict):
                out["n_samples_per_task"].setdefault(tk, info.get("effective"))
        # per-task processing_time only available when per_task/<task>/<wid>/
        # results file was produced for a single task — we approximate by
        # parent dir name.
        if "/per_task/" in str(path):
            tk = path.parts[-3]  # eval_*/per_task/<task>/<wid>/results.json
            t = data.get("total_evaluation_time_seconds")
            if t is not None and tk not in out["processing_time_per_task"]:
                try:
                    out["processing_time_per_task"][tk] = float(t)
                except (TypeError, ValueError):
                    pass
    return out


def build_rows(project_dir: Path) -> list[dict]:
    """Walk every NAME under project_dir, emit one row per (NAME, task)."""
    rows: list[dict] = []
    skipped: list[str] = []

    for name_dir in sorted(project_dir.iterdir()):
        if not name_dir.is_dir():
            continue
        # Skip debug/test artifacts
        nm = name_dir.name
        if any(x in nm for x in ("salloc-debug", "salloc-test", "srun-debug")):
            continue

        meta = name_to_metadata(nm)
        if meta is None:
            skipped.append(nm)
            continue

        scores = collect(name_dir)  # union of all per-task results JSON
        if not scores:
            continue
        eval_meta = collect_eval_metadata(name_dir)

        flops = (6 * meta["params"] * meta["tokens"]) if (meta["params"] and meta["tokens"]) else None

        for task, raw_metrics in scores.items():
            if not isinstance(raw_metrics, dict):
                continue
            metrics = normalize_metrics(raw_metrics)
            if not metrics:
                continue
            primary_metric, primary_score = pick_primary(metrics)

            rows.append({
                "split": meta["split"],
                "task": task,
                "name": meta["name"],
                "model": meta["model"],
                "family": meta["family"],
                "model_revision": meta["model_revision"],
                "model_type": meta["model_type"],
                "model_path": eval_meta["model_path"],
                "model_source": eval_meta["model_source"],
                "model_params": float(meta["params"]) if meta["params"] else None,
                "model_tokens": meta["tokens"],
                "flops": float(flops) if flops is not None else None,
                "step": meta["step"],
                "size": meta["size"],
                "mix": meta["mix"],
                "seed": meta["seed"],
                "primary_score": primary_score,
                "primary_metric": primary_metric,
                "metrics": metrics,
                "num_instances": eval_meta["n_samples_per_task"].get(task),
                "num_fewshot": eval_meta["n_shot_per_task"].get(task),
                "processing_time": eval_meta["processing_time_per_task"].get(task),
                # Serialize as JSON: the keys differ between megatron_lm
                # and vllm backends so a struct schema can't be unified.
                "model_config": (
                    json.dumps(eval_meta["model_args"], sort_keys=True)
                    if eval_meta["model_args"] is not None
                    else None
                ),
            })

    if skipped:
        print(f"Skipped {len(skipped)} unparseable NAMEs (e.g. {skipped[:3]})", file=sys.stderr)
    return rows


# --- Colleague's epfl-nlp/multilingual-evals dataset ----------------------
#
# Layout: raw/<benchmark>/<model_dir>/[<ckpt_dir>/[<inner_repo>/]]results_*.json
#
# Models / checkpoint conventions observed at listing time:
#   apertus-8b-2509:
#     - main
#     - step<N>-tokens<X>(B|T)            ← explicit tokens in dir name
#     - depth-3 (no ckpt)                 ← treated as `main`
#   olmo-3-1025-7b:
#     - stage1-step<N>                    ← linear-interp tokens to 6T@1413814
#   smollm3-3b-checkpoints:
#     - stage1-step-<N>  (note hyphen)    ← linear-interp tokens to 7.2T@3440000
#   smollm3-3b-base:
#     - HuggingFaceTB__SmolLM3-3B-Base    ← internal lm-eval output dir; alias to `main`
#     - depth-3 (no ckpt)                 ← treated as `main`

# Remote raw/ sources to merge from + the model_dir → models.json key map,
# all from configs/hf_wandb.json. Listed in priority order — when the same
# (model, revision, task) appears in multiple repos, the first repo's value
# wins (dedupe is order-preserving). `epfl-nlp/multilingual-evals` is
# intentionally omitted from raw_source_repos (private storage over quota →
# 403; the multilingual-snr repo is a superset).
RAW_SOURCE_REPOS = _HF_WANDB_CFG["multilingual_evals"]["raw_source_repos"]

# model_dir (remote raw/ folder name) → models.json key. Display name +
# every other field (params, size, family, split) is looked up from the
# canonical entry — `model` is the only thing we need a manual map for
# because the remote layout uses lowercase / hyphenated variants.
MULTILINGUAL_EVAL_MODELS = _HF_WANDB_CFG["multilingual_evals"]["model_dirs"]


def _branch_tokens(model: str, branch: str) -> float | None:
    """Return canonical-branch tokens via configs.tokens_for, or None when
    the branch isn't declared (e.g. intermediate stage steps)."""
    try:
        v = tokens_for(model, branch)
    except (KeyError, ValueError):
        return None
    return float(v) if v else None


def _stage1_tokens_per_step(model: str) -> float | None:
    """Linear-interp tokens-per-step for stage-1 *intermediate* checkpoints
    that aren't declared as canonical branches in configs/models.json. Per
    the SNR stage convention, the repo's `stage1` is its `pretraining`
    stage — so this is that stage's `tokens_per_iter`. None when unknown."""
    pre = get_model(model).get("stages", {}).get("pretraining")
    return pre.get("tokens_per_iter") if pre else None


def parse_ckpt_dir(model: str, ckpt_dir: str | None) -> dict | None:
    """Map (model, ckpt_dir) → {model_revision, step, tokens, mix}.

    `ckpt_dir is None` means a depth-3 file under the model dir (no checkpoint
    subdir) — treated as the `main` revision.
    """
    # Treat None / 'main' / inner-repo dirs as the canonical 'main' ckpt
    if ckpt_dir is None or ckpt_dir == "main" or ckpt_dir in {
        "HuggingFaceTB__SmolLM3-3B-Base",
        "swiss-ai__Apertus-8B-2509",
        "allenai__Olmo-3-1025-7B",
        "HuggingFaceTB__SmolLM3-3B-checkpoints",
    }:
        return {
            "model_revision": "main",
            "step": 0,
            "tokens": _branch_tokens(model, "main"),
            "mix": "main",
        }

    # Apertus-style: step<N>-tokens<X>(B|T) — tokens are explicit in the dir name.
    m = re.match(r"^step(?P<n>\d+)-tokens(?P<mag>[\d.]+)(?P<unit>[BT])$", ckpt_dir)
    if m:
        unit = 1e9 if m.group("unit") == "B" else 1e12
        return {
            "model_revision": ckpt_dir,
            "step": int(m.group("n")),
            "tokens": float(m.group("mag")) * unit,
            "mix": "main",
        }

    # SmolLM3-checkpoints style: stage<K>-step-<N>  (with hyphen)
    # Olmo style:                 stage<K>-step<N>(-suffix)?
    m = re.match(r"^stage(?P<stage>\d+)-step-?(?P<n>\d+)(?:-.*)?$", ckpt_dir)
    if m:
        stage, n = int(m.group("stage")), int(m.group("n"))
        # Try the canonical-branch lookup first (covers the final step of
        # each stage); fall back to linear interpolation for intermediates.
        tokens = _branch_tokens(model, ckpt_dir)
        if tokens is None and stage == 1:
            per_step = _stage1_tokens_per_step(model)
            if per_step is not None:
                tokens = n * per_step
        if tokens is None:
            return None
        return {
            "model_revision": ckpt_dir,
            "step": n,
            "tokens": tokens,
            "mix": f"stage{stage}",
        }

    return None


def fetch_multilingual_evals_rows(
    repo_ids: list[str] | str | None = None,
) -> list[dict]:
    """List & download every results_*.json in one or more HF dataset repos
    that follow the colleague's `raw/<bench>/<model>/<ckpt>/results_*.json`
    layout. Returns rows in the same shape as build_rows().

    `repo_ids` defaults to `RAW_SOURCE_REPOS` (configs/hf_wandb.json). When
    multiple repos are given, the first repo's path wins on conflict (so
    order matters: put the more authoritative repo first).
    """
    from huggingface_hub import HfApi, hf_hub_download

    if repo_ids is None:
        repo_ids = RAW_SOURCE_REPOS
    if isinstance(repo_ids, str):
        repo_ids = [repo_ids]

    api = HfApi()
    # path → (repo_id, path). First repo to list a given relative path wins.
    seen_path: dict[str, tuple[str, str]] = {}
    for repo in repo_ids:
        try:
            for f in api.list_repo_files(repo, repo_type="dataset"):
                if f.startswith("raw/") and "/results_" in f and f.endswith(".json"):
                    seen_path.setdefault(f, (repo, f))
        except Exception as e:
            print(f"  warn: list_repo_files({repo}) failed: {e}")
    files = list(seen_path.values())
    print(f"Found {len(files)} unique results_*.json files across {len(repo_ids)} repo(s)")

    # Group: (model_dir, ckpt_dir or None) → list[(repo_id, path)]
    groups: dict[tuple[str, str | None], list[tuple[str, str]]] = defaultdict(list)
    for repo, f in files:
        parts = f.split("/")[1:]  # strip raw/
        # parts: [<bench>, <model>, ...]
        bench, model_dir = parts[0], parts[1]
        if model_dir not in MULTILINGUAL_EVAL_MODELS:
            continue
        # Determine ckpt_dir based on depth (3/4/5)
        if len(parts) == 3:
            ckpt_dir = None
        elif len(parts) == 4:
            ckpt_dir = parts[2]
        elif len(parts) == 5:
            # raw/<bench>/<model>/<ckpt>/<inner_repo>/results_*.json — use ckpt
            ckpt_dir = parts[2]
        else:
            continue
        groups[(model_dir, ckpt_dir)].append((repo, f))

    print(f"Grouped into {len(groups)} (model, ckpt) pairs.")

    rows: list[dict] = []
    skipped: list[str] = []
    for (model_dir, ckpt_dir), repo_paths in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
    ):
        model = MULTILINGUAL_EVAL_MODELS[model_dir]
        try:
            entry = get_model(model)
        except KeyError:
            skipped.append(f"{model_dir}/{ckpt_dir} (not in models.json)")
            continue
        ckpt_meta = parse_ckpt_dir(model, ckpt_dir)
        if ckpt_meta is None:
            skipped.append(f"{model_dir}/{ckpt_dir}")
            continue

        # Union of results / groups across every results JSON for this ckpt
        scores: dict[str, dict] = {}
        n_shot: dict = {}
        n_samples: dict = {}
        proc_time: dict[str, float] = {}
        model_path = None
        model_source = None
        model_args = None

        for repo, remote in repo_paths:
            try:
                local = hf_hub_download(repo, remote, repo_type="dataset")
                data = json.loads(Path(local).read_text())
            except Exception as e:
                print(f"  warn: download/parse failed for {remote}: {e}")
                continue
            for source in ("results", "groups"):
                for k, v in (data.get(source) or {}).items():
                    if isinstance(v, dict) and k not in scores:
                        scores[k] = v
            for tk, n in (data.get("n-shot") or {}).items():
                n_shot.setdefault(tk, n)
            for tk, ni in (data.get("n-samples") or {}).items():
                if isinstance(ni, dict):
                    n_samples.setdefault(tk, ni.get("effective"))
            # The colleague runs one benchmark per JSON, so total_evaluation_time
            # is the per-benchmark wall time. Apply it to every task in this file.
            t = data.get("total_evaluation_time_seconds")
            if t is not None:
                try:
                    t_f = float(t)
                    for tk in (data.get("results") or {}):
                        proc_time.setdefault(tk, t_f)
                except (TypeError, ValueError):
                    pass
            if model_source is None:
                model_source = data.get("model_source")
            if model_args is None:
                ma = (data.get("config") or {}).get("model_args") or {}
                if isinstance(ma, dict):
                    model_args = ma
                    model_path = ma.get("pretrained") or ma.get("load")

        if not scores:
            continue

        params = entry.get("params")
        tokens = ckpt_meta["tokens"]
        flops = (6 * params * tokens) if (params and tokens) else None
        full_name = f"{model}-{ckpt_meta['model_revision']}"
        src = entry.get("source")
        split = split_for_source(src) if src else None
        if split is None:
            skipped.append(f"{model_dir}/{ckpt_dir} (source {src!r} not published)")
            continue

        for task, raw_metrics in scores.items():
            if not isinstance(raw_metrics, dict):
                continue
            metrics = normalize_metrics(raw_metrics)
            primary_metric, primary_score = pick_primary(metrics)

            rows.append({
                "split": split,
                "task": task,
                "name": full_name,
                "model": model,
                "family": entry.get("family"),
                "model_revision": ckpt_meta["model_revision"],
                "model_type": "reference_hf",
                "model_path": model_path,
                "model_source": model_source,
                "model_params": float(params) if params else None,
                "model_tokens": float(tokens) if tokens else None,
                "flops": float(flops) if flops else None,
                "step": ckpt_meta["step"],
                "size": entry.get("size"),
                "mix": ckpt_meta["mix"],
                "seed": None,
                "primary_score": primary_score,
                "primary_metric": primary_metric,
                "metrics": metrics,
                "num_instances": n_samples.get(task),
                "num_fewshot": n_shot.get(task),
                "processing_time": proc_time.get(task),
                "model_config": json.dumps(model_args, sort_keys=True) if model_args else None,
            })

    if skipped:
        print(f"  skipped {len(skipped)} (model, ckpt) groups with unknown ckpt format: {skipped[:5]}")
    print(f"Built {len(rows)} rows from {repo_ids}")
    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """If the same (model, model_revision, task) appears in both local and
    multilingual-evals sources, prefer the local one (which has full lm-eval
    metadata + per-task processing_time). Stable: input order matters, first
    occurrence wins."""
    seen = set()
    out = []
    for r in rows:
        key = (r["model"], r["model_revision"], r["task"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _arrow_schema():
    """Single canonical schema used for every split, so HF Datasets can
    concatenate splits without a cast error (unlike upstream, which
    sidesteps this by declaring per-split data_files only)."""
    import pyarrow as pa
    metric_struct = pa.struct([(k, pa.float64()) for k in METRIC_FIELDS])
    return pa.schema([
        ("task", pa.large_string()),
        ("name", pa.large_string()),
        ("model", pa.large_string()),
        ("family", pa.large_string()),
        ("model_revision", pa.large_string()),
        ("model_type", pa.large_string()),
        ("model_path", pa.large_string()),
        ("model_source", pa.large_string()),
        ("model_params", pa.float64()),
        ("model_tokens", pa.float64()),
        ("flops", pa.float64()),
        ("step", pa.int64()),
        ("size", pa.large_string()),
        ("mix", pa.large_string()),
        ("seed", pa.int64()),
        ("primary_score", pa.float64()),
        ("primary_metric", pa.large_string()),
        ("metrics", metric_struct),
        ("num_instances", pa.int64()),
        ("num_fewshot", pa.int64()),
        ("processing_time", pa.float64()),
        ("model_config", pa.large_string()),
    ])


def write_parquets(rows: list[dict], out_dir: Path) -> dict[str, Path]:
    """Group rows by 'split' and write data/<split>-00000-of-00001.parquet
    with a unified arrow schema across splits."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    schema = _arrow_schema()
    paths: dict[str, Path] = {}
    for split, split_rows in sorted(by_split.items()):
        df = pd.DataFrame(split_rows).drop(columns=["split"])
        df = df.sort_values(["model", "step", "task"]).reset_index(drop=True)
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        path = data_dir / f"{split}-00000-of-00001.parquet"
        pq.write_table(table, path)
        paths[split] = path
        print(
            f"  wrote {path.relative_to(out_dir)}: {len(df)} rows, "
            f"{df['model'].nunique()} model(s), {df['task'].nunique()} task(s)"
        )
    return paths


README_TEMPLATE = """---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
  - multilingual
tags:
  - evaluation
  - language-models
  - swiss-ai
  - apertus
  - olmo
  - smollm
  - signal-to-noise
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: pretraining_custom
        path: data/pretraining_custom-*.parquet
      - split: pretraining_a06
        path: data/pretraining_a06-*.parquet
      - split: reference_hf
        path: data/reference_hf-*.parquet
      - split: posttraining
        path: data/posttraining-*.parquet
      - split: distillation
        path: data/distillation-*.parquet
---

# SwissAI Evals — SNR Experiments

Aggregate evaluation results for the SwissAI signal-to-noise (SNR) study.
Results were produced by the SwissAI multilingual SNR pipeline on top of
[lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness).

The schema mirrors [allenai/signal-and-noise](https://huggingface.co/datasets/allenai/signal-and-noise):
one row per `(model, model_revision, task)` with aggregate metrics only — no
per-instance predictions.

## Splits

| Split | Models | Description |
|---|---|---|
| `pretraining_custom` | apertus-{{175M, 350M, 600M, 1B}}-fwEdu{{30,60,90}}-seed{{28,1797,1904}} | 36 custom megatron pretraining curves (4 sizes × 3 mixes × 3 seeds) at canonical iters {{2k, 6k, 12k, 18k, 22k, 28k, 34k, 38k, 42k, 44k, 46k, 48k, 50k}} |
| `pretraining_a06` | apertus3-{{1b, 3b}}-*-nodes | a06 main pretraining runs |
| `reference_hf` | Apertus-8B/70B-2509 (incl. `step<N>-tokens<X>` intermediates), Olmo-3-1025-7B (stage1 intermediates + final), SmolLM3-3B (stage1 intermediates, stages 1/2/3 finals), SmolLM3-3B-Base | External reference checkpoints; merged from local cluster runs and the multilingual-snr raw/ folder |
| `posttraining` | Apertus-8B-Instruct-2509, gemma-3-1b-it, Olmo-3-7B-Instruct (+ other instruct variants as eval results land) | Instruct/post-trained reference checkpoints evaluated on the same task suite |
| `distillation` | apertus-{{0.6b, 1b}}-from8b-TOP256-long | Distillation runs (megatron checkpoints distilled from Apertus-8B) |

## Columns

| Field | Type | Description |
|---|---|---|
| `task` | str | lm-eval-harness task name (e.g. `mmlu`, `belebele_eng_Latn`) |
| `name` | str | Full evaluation NAME — `<model>-<revision>` |
| `model` | str | Base model name |
| `family` | str | Cross-size identity (e.g. `apertus-fwEdu30-fw270-seed1904`, `SmolLM3-checkpoints`) — used to group runs of different sizes when computing decision accuracy |
| `model_revision` | str | `iter<N>`, `stage<K>-step<N>`, or `main` |
| `model_type` | str | `custom_pretraining` or `reference_hf` |
| `model_path` | str | Checkpoint path or HF repo id used for evaluation |
| `model_source` | str | lm-eval backend — `megatron_lm` or `vllm` |
| `model_params` | float | Approximate total parameter count |
| `model_tokens` | float | Cumulative training tokens at this checkpoint |
| `flops` | float | ≈ 6 × params × tokens |
| `step` | int | Training iteration / step |
| `size` | str | Parameter-count tag (e.g. `175M`, `8B`) |
| `mix` | str | Data-mix tag (e.g. `fwEdu30-fw270`, `stage2`, `main`) |
| `seed` | int | Pretraining seed (custom only) |
| `primary_score` | float | Headline metric — `acc` if available else `exact_match` |
| `primary_metric` | str | Name of the primary metric |
| `metrics` | dict | All numeric metric variants reported for the task |
| `num_instances` | int | Number of effective evaluation samples |
| `num_fewshot` | int | Few-shot count used for the task |
| `processing_time` | float | Per-task evaluation time in seconds (when known) |
| `model_config` | dict | lm-eval `--model_args` |

## Loading

```python
from datasets import load_dataset

ds = load_dataset("multilingual-snr/multilingual-snr-eval-results")
df = ds["pretraining_custom"].to_pandas()
```

## Citation

If you use this data, please cite the SwissAI Apertus technical report.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--out-dir", default="/tmp/snr-hf-dataset",
                   help="Local staging directory for parquet + README")
    p.add_argument("--push", action="store_true", help="After writing, upload to HF")
    p.add_argument("--repo-id", default=REPO_ID,
                   help="HF dataset repo (created if missing)")
    p.add_argument("--private", action="store_true", help="Create private dataset")
    p.add_argument(
        "--include-multilingual-evals", action="store_true",
        help=f"Also pull rows from the raw/ sources {RAW_SOURCE_REPOS}",
    )
    p.add_argument(
        "--multilingual-evals-only", action="store_true",
        help="Skip the local cluster eval_logs walk; emit rows only from the "
             "remote multilingual-evals repo.",
    )
    args = p.parse_args()

    rows: list[dict] = []
    if not args.multilingual_evals_only:
        project_dir = LOGS_BASE / args.entity / args.project
        if not project_dir.is_dir():
            sys.exit(f"No project dir at {project_dir}")
        print(f"Walking {project_dir}")
        rows.extend(build_rows(project_dir))
        print(f"  {len(rows)} rows from local eval_logs.")

    if args.include_multilingual_evals or args.multilingual_evals_only:
        n_before = len(rows)
        rows.extend(fetch_multilingual_evals_rows(RAW_SOURCE_REPOS))
        print(f"  +{len(rows) - n_before} rows from raw sources {RAW_SOURCE_REPOS}.")

    rows = dedupe_rows(rows)
    print(f"Built {len(rows)} rows after dedupe.")
    if not rows:
        sys.exit("No rows built — nothing to push.")

    out_dir = Path(args.out_dir)
    paths = write_parquets(rows, out_dir)
    (out_dir / "README.md").write_text(README_TEMPLATE)
    print(f"Wrote README to {out_dir}/README.md")

    if not args.push:
        print("\n(--push not set; skipping HF upload)")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"\nCreating/updating dataset repo {args.repo_id}")
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Initial upload: SwissAI SNR evaluation results",
    )
    print(f"\nDone — https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
