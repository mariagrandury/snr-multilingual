"""Paper-local audit tables, RESHAPED from the pipeline's own tables.

The paper quotes numbers that must equal what `analysis/` computes. Until
2026-09-21 this script re-implemented the decision-accuracy kernel in plain
`csv`/`statistics` to produce them independently — 162 lines that read the
ladder report directly and applied neither `trained_only` (rule 2) nor
`MIN_PAIRS` (rule 5), and invented two per-language floors of its own (a
task count of 2 where rule 8 says `MIN_LANG_TASKS`, and a 5-point minimum
inside its own `corr`). RULES.md says the helpers named there are the only
implementation of each rule: use them, do not re-derive. A second kernel
cannot help but drift, and it did — re-run against a newer report snapshot
it disagreed with the pipeline on 10,419 of 26,485 shared rows, with pair
counts of 45 where the pipeline had 15 (C(10,2) against C(6,2): it counted
families whose mixture never trains the task's language).

So this script no longer computes anything. It reads the tables the pipeline
already wrote, reshapes them into the five paper-local files, and records the
sha256 of every input. The one quantity it derives is `passes_gate`, and it
derives that with `utils.passes_gate` — the shared helper — because the old
table used a `== 1` test that made NA (a task with no chance level: BPB, the
loss, the generative tasks) read as gated, wrong on 620 rows.

Run it after `refresh_analysis.sh`; it is cheap and reads only committed CSVs.

It also rewrites the constants block of main.tex (`% BEGIN generated:
constants`): the release counts \nmodels, \nckpts and \nbenchmarktasks (see
`provenance`), and the compute the sweep was charged on CSCS, read from
plan/compute-costs.json, which src/pretrain/compute_cost.py writes from sacct:

    \npretrainnodehours  pretrain node-hours charged, every allocation
    \nothernodehours     eval + BPB + convert + data node-hours charged
    \ngpuhours           4 GH200 per node x the two above

Charged, not kept: the methodology sentence says what pretraining "consumed",
and charged is what the allocations used, failed and requeued work included.
Node-hours are rounded to the nearest 100 and GPU-hours derived from the
rounded values, so the three macros add up for a reader.

    python3 verify_paper_results.py            # rewrite the five tables
    python3 verify_paper_results.py --check    # exit 1 if any would change
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ANALYSIS = ROOT / "src" / "signal-and-noise" / "analysis"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "signal-and-noise"))

from analysis.utils import (  # noqa: E402
    ANALYSIS_SIZES, LANGUAGE_AGGREGATES, MIN_LANG_TASKS, MIN_PAIRS, TARGET_SIZE,
    assign_language, parents_only, passes_gate)
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq04_surrogates.snr_definition_postprocess import _table  # noqa: E402

COMPUTE_COSTS = ROOT / "plan" / "compute-costs.json"   # src/pretrain/compute_cost.py
POOL = "predictivity"
P = f"pretraining/{POOL}"
RQ00 = ANALYSIS / "rq00_gate_and_curves" / P
RQ02 = ANALYSIS / "rq02_decision_accuracy" / P
RQ03 = ANALYSIS / "rq03_noise_and_snr" / P
# The cross-task pair section 4 names; the by_family tables hold every pair.
CROSS_PAIR = ("hellaswag", "multiblimp")
SOURCES: list[Path] = []


def read(path: Path) -> pd.DataFrame:
    """Read a pipeline table and remember it for the provenance record."""
    SOURCES.append(path)
    return pd.read_csv(path)


def da_ten_checkpoints() -> pd.DataFrame:
    """Per-task DA on the ten checkpoints, melted from rq02's live table.

    `da_all_pooled_per_task_multi_axes.csv` carries both flavours side by side — `da_ref` /
    `n_pairs_ref` against the reference, `da_own` / `n_pairs_own` within the
    proxy's own size — so the two `kind` values of this table are one melt of
    it, not a second computation.

    NOT read: `ten_checkpoints.csv`, which this script used until 2026-09-21.
    Nothing has written that file since 2026-09-19 (`paper_ten_checkpoints.py`
    writes rq2_da_goal_ten_checkpoints_multi_axes.csv/png/svg; no code path produces it), so it is an orphan
    under rule 14, and `documents/paper/figures/ten_checkpoints.csv` is a
    3.3 MB copy of the same orphan. Its gate column disagreed with
    `utils.passes_gate` in both directions on 320 rows, which is what an
    output frozen two days behind its code looks like.
    """
    d = read(RQ02 / "da_all_pooled_per_task_multi_axes.csv")
    SOURCES.append(RQ00 / "above_random_mask.csv")
    parts = []
    for kind, da, n in (("reference", "da_ref", "n_pairs_ref"),
                        ("checkpoint", "da_own", "n_pairs_own")):
        part = d[["task", "proxy_size", "frac", da, n]].rename(
            columns={"proxy_size": "size", da: "da", n: "n_pairs"})
        part.insert(3, "kind", kind)
        parts.append(part)
    t = pd.concat(parts, ignore_index=True)
    t["percent"] = (t["frac"] * 100).round().astype(int)
    t = t[["task", "size", "percent", "kind", "da", "n_pairs"]]
    # rule 1 through the shared helper, the way rq02 applies it: above chance
    # at the proxy size AND at the reference. 0 rejects, 1 passes, NA (no
    # chance level: BPB, the loss, the generative tasks) passes.
    mask = load_mask(POOL)
    ok = pd.Series(False, index=t.index)
    for size in t["size"].unique():
        rows = t["size"] == size
        ok[rows] = passes_gate(mask, t.loc[rows, "task"], size, TARGET_SIZE).to_numpy()
    t["passes_gate"] = ok.to_numpy()
    return t.sort_values(["task", "size", "percent", "kind"]).reset_index(drop=True)


def da_summary(ten: pd.DataFrame) -> pd.DataFrame:
    """Mean DA per (kind, size, checkpoint percent, population)."""
    g = ten.copy()
    g["group"] = ["bpb" if str(t).startswith("bpb_") else "benchmarks" for t in g["task"]]
    out = (g.dropna(subset=["da"]).groupby(["kind", "size", "percent", "group"])
            .agg(mean_da=("da", "mean"), tasks=("task", "nunique")).reset_index())
    return out.sort_values(["kind", "size", "percent", "group"])


def cross_task_summary() -> pd.DataFrame:
    """Reach counts for the pair the paper names, from rq02's by_family tables.

    Those tables carry shares over `n_cells`; the paper quotes counts, so the
    shares are multiplied back out here rather than counted a second time.
    """
    rows = []
    for kind, name in (("size", "cross_task_da_size_by_family_multi_axes"),
                       ("checkpoint", "cross_task_da_ckpt_by_family_multi_axes")):
        d = read(RQ02 / f"{name}.csv")
        for proxy, target in (CROSS_PAIR, CROSS_PAIR[::-1]):
            cell = d[(d["proxy"] == proxy) & (d["target"] == target)]
            if cell.empty:
                continue
            r = cell.iloc[0]
            n = int(r["n_cells"])
            reached = round(r["share_reached"] * n)
            never = round(r["share_never"] * n)
            gated = round(r["share_gated"] * n)
            rows.append({"proxy_family": proxy, "target_family": target, "kind": kind,
                         "total": n, "reached": int(reached), "never": int(never),
                         "gated": int(gated), "missing": int(n - reached - never - gated)})
    return pd.DataFrame(rows)


def surrogates_by_language() -> pd.DataFrame:
    """Per-language Pearson r per SNR variant, from rq04's own kernel.

    `_table` is what `top_variants_overall` ranks on, so this table and the
    paper's ranking cannot disagree: rule 8's `MIN_LANG_TASKS` floor and the
    trained-language and parent-task filters are applied inside it.
    """
    df = read(RQ03 / "snr_variants_per_task.csv").set_index("task")
    rows = []
    for kind, label in (("size", "size"), ("ckpt", "checkpoint")):
        t = _table(df, kind)
        for variant in t.index:
            for language, r in t.loc[variant].dropna().items():
                rows.append({"language": language, "kind": label,
                             "variant": variant, "pearson_r": float(r)})
    out = pd.DataFrame(rows)
    # `n` is the population behind each mean a reader might take over this
    # table: the languages the variant covers (rule 13 — the count travels).
    out["n"] = out.groupby(["kind", "variant"])["language"].transform("nunique")
    return out.sort_values(["language", "kind", "variant"])


def compute_hours() -> dict:
    """Charged node-hours (pretrain / everything else) and GPU-hours, rounded."""
    SOURCES.append(COMPUTE_COSTS)
    c = json.loads(COMPUTE_COSTS.read_text())
    nh = {task: v["charged"] for task, v in c["node_hours"].items()}
    pretrain = round(nh.pop("pretrain"), -2)
    other = round(sum(nh.values()), -2)
    return {"as_of": c["date"], "pretrain_node_hours": int(pretrain),
            "other_node_hours": int(other),
            "gpu_hours": int(c["gpus_per_node"] * (pretrain + other))}


def provenance(ten: pd.DataFrame) -> dict:
    """What was read, and the population the paper's prose must quote."""
    compute = compute_hours()
    from snr.download.ladder import load_predictivity_eval_results
    d = load_predictivity_eval_results()
    d = d[d["size"].isin(ANALYSIS_SIZES)]
    head = d[(d["seed"] == 1904) & (d["data"].isin(["A", "B"]))]
    bench = parents_only(d.loc[d["kind"] == "benchmark", ["task"]].drop_duplicates())
    return {
        "compute_charged": compute,
        "generated_by": "documents/paper/sections/verify_paper_results.py (reshape, no re-derivation)",
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(set(SOURCES))},
        "constants": {"MIN_PAIRS": MIN_PAIRS, "MIN_LANG_TASKS": MIN_LANG_TASKS,
                      "TARGET_SIZE": TARGET_SIZE, "ANALYSIS_SIZES": list(ANALYSIS_SIZES)},
        "healthy_175M_to_1_7B": int(d["model"].nunique()),
        "headline_seed1904_A_B_runs": int(head["model"].nunique()),
        "grid_seed1904_runs": int(d.loc[d["seed"] == 1904, "model"].nunique()),   # the `predictivity` pool
        "sizes": {k: int(v) for k, v in d.groupby("size")["model"].nunique().items()},
        "da_rows": int(len(ten)),
        # The release counts main.tex's \nmodels / \nckpts / \nbenchmarktasks quote:
        # every evaluated (run, checkpoint) and every per-language parent benchmark task
        # (rule 6, twins included; BPB, the loss and the cross-language aggregates are not).
        "evaluated_checkpoints": int(len(d[["model", "step"]].drop_duplicates())),
        "benchmark_tasks": int((~bench["task"].map(assign_language).isin(LANGUAGE_AGGREGATES)).sum()),
    }


def write_constants(p: dict) -> None:
    """Rewrite the counts block of main.tex from the provenance record."""
    tex = HERE / "main.tex"
    t = tex.read_text()
    B, E = "% BEGIN generated: constants (verify_paper_results.py)", "% END generated: constants"
    i, j = t.index(B), t.index(E)
    c = p["compute_charged"]
    block = (f"\\newcommand{{\\nmodels}}{{{p['healthy_175M_to_1_7B']} }}\n"
             f"\\newcommand{{\\nckpts}}{{{p['evaluated_checkpoints']:,} }}\n"
             f"\\newcommand{{\\nbenchmarktasks}}{{{p['benchmark_tasks']:,} }}\n"
             f"\\newcommand{{\\npretrainnodehours}}{{{c['pretrain_node_hours']:,} }}\n"
             f"\\newcommand{{\\nothernodehours}}{{{c['other_node_hours']:,} }}\n"
             f"\\newcommand{{\\ngpuhours}}{{{c['gpu_hours']:,} }}\n")
    tex.write_text(t[:i] + B + "\n" + block + t[j:])


def main(check: bool) -> int:
    ten = da_ten_checkpoints()
    tables = {
        "verified_da_ten_checkpoints.csv": ten,
        "verified_da_summary.csv": da_summary(ten),
        "verified_cross_task_summary.csv": cross_task_summary(),
        "verified_surrogates_by_language.csv": surrogates_by_language(),
    }
    moved = []
    for name, df in tables.items():
        path = HERE / name
        new = df.to_csv(index=False)
        stale = not path.exists() or path.read_text() != new
        if stale:
            moved.append(name)
            if not check:
                path.write_text(new)
        print(f"{'would change' if check and stale else 'wrote':>12}  {name}  ({len(df)} rows)")
    text = json.dumps(provenance(ten), indent=2, sort_keys=True) + "\n"
    prov = HERE / "verified_results_provenance.json"
    p = json.loads(text)
    if not check:
        prov.write_text(text)
        write_constants(p)
    print(f"{'wrote':>12}  {prov.name}")
    print(f"\npopulation: {p['healthy_175M_to_1_7B']} healthy runs 175M-{TARGET_SIZE}, "
          f"{p['grid_seed1904_runs']} in the grid pool (seed 1904, every data build), "
          f"{p['headline_seed1904_A_B_runs']} of them on data A/B")
    print(f"            per size {p['sizes']}")
    print("            the same two counts live in documents/ladder-facts.json, which\n"
          "            facts.py diffs on every refresh — the paper's prose quotes them.")
    if check and moved:
        print("\n!!! stale: " + ", ".join(moved))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
