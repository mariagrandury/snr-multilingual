"""Data of the gate page's additional views (docs/interactive/views/gate.js),
read from rq00's committed tables; dumped as gate_extra.json by
scripts/build_site_data.py. No statistic is computed here except counts and
pooled ratios of the rows (the funnel per size, the per-benchmark exclusion DA
pooled over the proxies as matching / pairs, as the token-exclusion README
does) and the score a task needs (the gate's own Wilson bound)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
from analysis import paths  # noqa: E402
from analysis.utils import PROBE_SIZES  # noqa: E402

SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"]
# internal checkpoints never reach the site (the same guard as build_site_data.UNRELEASED)
UNRELEASED = r"apertus3|a06|from8b|distill"


def _rows(df: pd.DataFrame) -> list[dict]:
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
             if not (isinstance(v, float) and v != v)} for r in df.to_dict("records")]


def _fmt(family: str) -> str:
    return "LLM-RF" if family.startswith("rfgm_") else "RF" if family.startswith("rf_") else "original"


def _funnel(mask: pd.DataFrame, sizes: list[str]) -> list[dict]:
    """Per size: tasks above chance there, at chance there but above at another
    size, and never above chance at any of `sizes`."""
    m = mask[sizes]
    never = int((m.fillna(0) == 0).all(axis=1).sum())
    out = []
    for s in sizes:
        above = int((m[s] == 1).sum())
        out.append(dict(size=s, above=above, elsewhere=len(m) - above - never, never=never, n=len(m)))
    return out


def build() -> dict:
    from analysis.grids import paper_name
    from analysis.rq00_gate_and_curves.above_random import wilson_lcb

    g = paths.GATE_AND_CURVES / "pretraining/predictivity"
    tok = paths.CHANCE_VS_TRAIN_TOKENS / "pretraining"

    # the trained tasks: read on runs that train the language at every size
    share = pd.read_csv(g / "above_random_share.csv")
    trained = (share[[f"{s}__population" for s in SIZES]] == "trained").all(axis=1)
    mask = pd.read_csv(g / "above_random_mask.csv").set_index("task")
    tasks = share.loc[trained, "task"]
    m = mask.loc[tasks]
    m = m.assign(fmt=m["family"].map(_fmt))
    population = {"Analysed": _funnel(m, SIZES)}
    for f, label in [("original", "Analysed · original"), ("RF", "Analysed · RF"), ("LLM-RF", "Analysed · LLM-RF")]:
        population[label] = _funnel(m[m["fmt"] == f], SIZES)

    v = pd.read_csv(g / "gate_screened_verdicts_all_considered.csv")
    # utils.PROBE_SIZES, less a size with no probe verdict in this snapshot (3B until its probe jobs land)
    probe = [s for s in PROBE_SIZES if f"above_{s}" in v.columns and s in m.columns]
    v = v.rename(columns={f"above_{s}": s for s in probe})
    population["Screened, not analysed"] = _funnel(v[v["group"] == "probe"], probe)
    population["Discarded"] = _funnel(v[v["group"] == "discarded"], probe)
    # the pending benchmarks (utils.pending_benchmarks) are in the trained gate and in the screened
    # group at once: count them once, as screened, so the total is the considered table's 1,454
    population["All considered"] = _funnel(pd.concat([m.loc[~m.index.isin(v["task"]), probe],
                                                      v.set_index("task")[probe]]), probe)

    # the score a run needs on each trained task (above_random_example.needed_score)
    def needed(n: int, chance: float):
        ks = np.arange(n + 1)
        ok = ks[wilson_lcb(ks / n, np.full(n + 1, n)) > chance]
        return round(float(ok[0] / n), 4) if len(ok) else None

    def first(r) -> str:
        for i, s in enumerate(SIZES):
            if r[s] == 1 and all(r[t] == 1 for t in SIZES[i:]):
                return s
        return "never"

    margin = [dict(task=t, family=r["family"], language=r["language"], n=int(r["n_items"]),
                   chance=round(float(r["random_baseline"]), 4),
                   need=needed(int(r["n_items"]), float(r["random_baseline"])), first=first(r))
              for t, r in m.iterrows() if r["n_items"] == r["n_items"]]

    maj = pd.read_csv(g.parent.parent / "majority_chance/majority_pass_counts.csv")
    maj_b = pd.read_csv(g.parent.parent / "majority_chance/majority_by_benchmark.csv")

    early = pd.read_csv(g / "early_passes.csv")

    exc = pd.read_csv(tok / "predictivity/token_exclusion_da_size_both_axes.csv")
    fam = pd.read_csv(tok / "predictivity/token_exclusion_da_size_per_family_both_axes.csv")
    fam = fam[fam["reference"] == "same policy"].groupby(["benchmark", "axes", "policy"])[["n_matching", "n_pairs"]].sum()
    fam = (fam["n_matching"] / fam["n_pairs"]).rename("da").reset_index()

    thr = pd.read_csv(tok / "predictivity_seeds/pass_threshold_tokens_by_benchmark_1904_ckpts.csv")
    thr = thr[["benchmark", "name", "format", "n_languages", "margin_median", "n_items_median", "sizes", "level",
               "status", "bin_lo", "lowest_bin_lo", "highest_bin_hi", "share_in_bin"]]

    ext = pd.read_csv(g / "above_random_external_lines.csv")
    ext = ext[~ext["line"].str.contains(UNRELEASED, case=False) & ~ext["model"].str.contains(UNRELEASED, case=False)]

    keys = set(m["family"]) | set(maj_b["family"]) | set(fam["benchmark"]) | set(thr["benchmark"])
    return {"gate_extra": dict(
        sizes=SIZES, names={k: paper_name(k) for k in sorted(keys)},
        analysed=dict(tasks=len(m), benchmarks=int(m["family"].nunique()), languages=int(m["language"].nunique())),
        population=population, margin=margin,
        majority=_rows(maj), majority_by_benchmark=_rows(maj_b),
        early=_rows(early),
        exclusion=_rows(exc), exclusion_by_benchmark=_rows(fam),
        thresholds=_rows(thr),
        external_lines=_rows(ext))}
