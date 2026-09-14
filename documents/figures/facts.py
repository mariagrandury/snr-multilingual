"""The headline numbers, in one machine-readable file.

Prose in the report, the deck and the compendium quotes these. Figures and
generated tables refresh themselves; hand-written sentences cannot, so this
records every number worth quoting and diffs it against the last refresh. A
number that moved is named, with its old and new value, so the sentence
carrying it can be found and fixed instead of quietly going stale.

    python3 facts.py            # write documents/ladder-facts.json, print the diff
    python3 facts.py --check    # exit 1 if anything moved (for a pre-commit gate)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
ANALYSIS = REPO / "src" / "signal-and-noise" / "analysis"
P = "pretraining/predictivity"
PS = "pretraining/predictivity_seeds"
OUT = REPO / "documents" / "ladder-facts.json"


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def collect():
    import data as D
    f = {}

    w = D.wide()
    c = D.cells(w)
    ok = c["run__complete"] == 1
    f["grid"] = {
        "cells_in_report": int(len(c)),
        "complete": int(ok.sum()),
        "complete_not_diverged": int((ok & (c.run__diverged != 1)).sum()),
        "healthy": int((ok & (c.run__diverged != 1) & (c.run__off_trend != 1)).sum()),
        "complete_by_size": {k: int(v) for k, v in c[ok].groupby("size").size().items()},
        "complete_L": sorted(int(x) for x in c[ok]["L"].unique()),
        "schemes_present": sorted(c["scheme"].dropna().unique().tolist()),
    }

    g = pd.read_csv(ANALYSIS / f"rq00_acc_vs_flops/{P}/above_random_mask.csv")
    g = g[g["n_options"].notna()]
    sizes = [s for s in ["90M", "175M", "350M", "600M", "1B", "1.7B"] if s in g.columns]
    clears = g[sizes].eq(1).any(axis=1)
    fam = g.groupby("family").apply(lambda d: d[sizes].eq(1).any(axis=1).any(), include_groups=False)
    f["gate"] = {
        "gated_tasks": int(len(g)),
        "clear_any_size": int(clears.sum()),
        "by_options": {str(int(k)): [int(g[g.n_options == k][sizes].eq(1).any(axis=1).sum()),
                                     int((g.n_options == k).sum())]
                       for k in sorted(g.n_options.dropna().unique())},
        "per_size": {s: int((g[s] == 1).sum()) for s in sizes},
        "never_clearing_families": sorted(fam[~fam].index.tolist()),
    }

    tv = pd.read_csv(ANALYSIS / f"rq02_snr_definition/{P}/top_variants_overall.csv")
    best = tv.iloc[0]
    d = pd.read_csv(ANALYSIS / f"rq02_snr_definition/{P}/top_benchmarks_per_language.csv")
    d["is_bpb"] = d.task.str.startswith("bpb_")
    r1 = d[(d["rank"] == 1) & (d.language != "multi")]
    bench = d[~d.is_bpb & (d.language != "multi")]
    wins = 0
    for lang, gg in bench.groupby("language"):
        wins += gg.snr.max() > d[(d.language == lang) & d.is_bpb].snr.max()
    bpb_best = d[d.is_bpb & (d.language != "multi")].groupby("language")["snr"].max()
    f["snr"] = {
        "best_variant": str(best["variant"]),
        "best_r_da_ckpt": _r(best["mean_r_da_ckpt"]),
        "best_r_da_size": _r(best["mean_r_da_size"]),
        "best_r_overall": _r(best["mean_r_overall"]),
        "validation_languages": int(len(r1)),
        "rank1_is_bpb": int(r1.is_bpb.sum()),
        "languages_with_surviving_benchmark": int(bench.language.nunique()),
        "benchmark_beats_bpb": int(wins),
        "bpb_snr_median_with_benchmark": _r(bpb_best[bpb_best.index.isin(bench.language)].median()),
        "bpb_snr_median_without": _r(bpb_best[~bpb_best.index.isin(bench.language)].median()),
    }

    h = pd.read_csv(ANALYSIS / f"rq02_snr_definition/pretraining/"
                    f"predictivity_seeds_train__vs__predictivity_seeds_test/headline_metrics.csv")
    f["seed_holdout"] = {f"{r.metric}_{r.da_kind}": _r(r.value) for r in h.itertuples()}

    iv = pd.read_csv(ANALYSIS / f"rq06_proxy_predictivity/{PS}/intervention_da.csv")
    sl = pd.read_csv(ANALYSIS / f"rq06_proxy_predictivity/{PS}/scaling_law_error.csv")
    ev = pd.read_csv(ANALYSIS / f"rq06_proxy_predictivity/{PS}/effect_vs_noise.csv")
    ratio = (ev.seed_noise / ev.ckpt_noise_detrended).replace([np.inf, -np.inf], np.nan).dropna()
    arch = ev.effect_arch_over_seed.dropna()
    f["rq6"] = {
        "intervention_cells": int(len(iv)),
        "reference_sizes": sorted(iv.reference_size.dropna().unique().tolist()),
        "scaling_fits": int(len(sl)),
        "fits_by_reference": {k: int(v) for k, v in sl.reference_size.value_counts().items()},
        "rel_error_median_abs_by_L": {str(k): _r(v) for k, v in
                                      sl.groupby("L")["rel_error"].apply(lambda s: s.abs().median()).items()},
        "seed_over_ckpt_noise": _r(ratio.median(), 2),
        "seed_over_ckpt_n": int(len(ratio)),
        "depth_over_seed_median": _r(arch.median(), 2),
        "depth_over_seed_above2_share": _r((arch > 2).mean(), 2),
        "depth_over_seed_n": int(len(arch)),
        "benchmark_population": [
            {"L": int(r.L), "proxy": r.proxy_size, "n": int(r.n_items), "da": _r(r.decision_acc)}
            for r in iv[iv.population == "benchmark"].itertuples()],
        "bpb_trained_scheme": {f"L{int(r.L)}@{r.proxy_size}": _r(r.decision_acc)
                               for r in iv[iv.population == "bpb_trained"].itertuples()},
    }

    from predictivity import min_predictive_size, kind
    mp = min_predictive_size()
    mp = mp.assign(kind=mp["task"].map(kind))
    f["surrogates"] = {k: {"works": _r(v["min_size"].notna().mean(), 3), "n": int(len(v))}
                       for k, v in mp.groupby("kind")}
    return f


def flatten(d, prefix=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else enumerate(d)):
        key = f"{prefix}{k}"
        if isinstance(v, (dict, list)) and not (isinstance(v, list) and not any(
                isinstance(x, (dict, list)) for x in v)):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


if __name__ == "__main__":
    new = collect()
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    fo, fn = flatten(old), flatten(new)
    moved = [(k, fo.get(k), fn[k]) for k in fn if k in fo and fo[k] != fn[k]]
    added = [k for k in fn if k not in fo]
    gone = [k for k in fo if k not in fn]

    check = "--check" in sys.argv
    if not check:  # a gate that rewrites the baseline passes on its second run
        OUT.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {OUT.relative_to(REPO)}")
    if moved:
        print(f"\n{len(moved)} numbers MOVED — find the sentences that quote them:")
        for k, a, b in moved:
            print(f"  {k}: {a} -> {b}")
    if added:
        print(f"\n{len(added)} new: {', '.join(added[:12])}{' …' if len(added) > 12 else ''}")
    if gone:
        print(f"\n{len(gone)} gone: {', '.join(gone[:12])}")
    if not (moved or added or gone):
        print("nothing moved since the last refresh")
    if check and (moved or gone):
        sys.exit(1)
