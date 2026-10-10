"""Data of the noise-and-SNR page's additional views (rq03_noise_and_snr).

- noise_snr_map: per (task, size) the `rel_std` signal, its detrended
  checkpoint noise and the raw/detrended noise ratio (the raw-noise SNR is
  `snr * ckpt_noise_detrended / ckpt_noise_raw`, run_apertus_snr_variants.py),
  plus the task's DA-size at that proxy; rows of snr_variants_per_task.csv
  where rq03 gives an SNR (the gate applied), only reshaped.
- noise_snr_curves_<size>: the score of every seed-1904 design variant of the
  size along its run (10 % grid) for the accuracy tasks with an SNR there;
  rows of rq00's predictivity_seeds/benchmark_curves.csv, in per-mille.
- noise_snr_effect: effect_vs_noise_paper.csv (median |effect| over seed /
  checkpoint noise per decision, population and size).
- noise_snr_flips: flip_attribution_da_size_by_axis_both_axes.csv at the
  proxy's own seed sd (sd_scope = size) and the depth-by-seed table.
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
from analysis import paths  # noqa: E402
from analysis.grids import paper_name  # noqa: E402
from analysis.utils import BBPB, assign_language, benchmark_family  # noqa: E402

PRED = "pretraining/predictivity"
SEEDS = "pretraining/predictivity_seeds"
SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"]


def _r(x, d=5):
    return None if pd.isna(x) else round(float(x), d)


def kind(task: str) -> str:
    if task.startswith("bpb_"):
        return "bpb"
    return "bbpb" if task.startswith(BBPB) else "acc"


def build() -> dict:
    v = pd.read_csv(paths.NOISE_AND_SNR / PRED / "snr_variants_per_task.csv")
    v = v[~v["task"].isin(["bpb_macro", "train_loss"])]          # not a language's (rule 7)
    tasks = sorted(v["task"])
    idx = {t: i for i, t in enumerate(tasks)}
    meta = [dict(t=t, f=benchmark_family(t), l=assign_language(t), k=kind(t)) for t in tasks]
    fams = sorted({m["f"] for m in meta})
    rows = []
    for r in v.to_dict("records"):
        for s in SIZES:
            snr = r.get(f"snr_rel_std_{s}")
            if pd.isna(snr) or snr <= 0:
                continue
            det, raw = r.get(f"ckpt_noise_detrended_{s}"), r.get(f"ckpt_noise_raw_{s}")
            rows.append([idx[r["task"]], s, _r(r[f"signal_rel_std_{s}"]), _r(r[f"noise_rel_std_{s}"]),
                         _r(raw / det if det and not pd.isna(raw) else None, 4),
                         _r(r.get(f"decision_acc_size_{s}"), 3)])
    out = {"noise_snr_map": dict(
        tasks=meta, names={f: paper_name(f) for f in fams},
        columns=["i", "size", "signal", "noise", "raw_over_detr", "da"], data=rows)}

    bc = pd.read_csv(paths.GATE_AND_CURVES / SEEDS / "benchmark_curves.csv")
    bc = bc[bc["model"].str.endswith("-seed1904")]                  # the predictivity pool's seed
    bc["frac"] = bc["frac"].round(2)
    fracs = sorted(bc["frac"].unique())
    has_snr = {(tasks[r[0]], r[1]) for r in rows}
    for s in SIZES:
        b = bc[bc["size"] == s]
        b = b[[(t, s) in has_snr for t in b["task"]]]
        models = b[["model", "ladder", "data"]].drop_duplicates("model").sort_values("model")
        models["L"] = models["model"].str.extract(r"-L(\d+)-", expand=False).astype(int)
        models = models.sort_values(["L", "ladder", "data"]).reset_index(drop=True)
        mi = {m: i for i, m in enumerate(models["model"])}
        wide = b.pivot_table(index=["task", "model"], columns="frac", values="primary_score").reindex(columns=fracs)
        curves = {}
        for (task, model), ys in wide.iterrows():
            curves.setdefault(task, []).append([mi[model]] + [None if pd.isna(y) else int(round(1000 * y)) for y in ys])
        out[f"noise_snr_curves_{s.replace('.', '_')}"] = dict(
            fracs=fracs, models=[dict(L=int(m.L), ladder=m.ladder, data=m.data) for m in models.itertuples()],
            curves=curves)

    e = pd.read_csv(paths.NOISE_AND_SNR / SEEDS / "effect_vs_noise_paper.csv")
    m = e["panel"].str.extract(r"^effect_(.+)_over_(seed|ckpt)$")
    e["decision"], e["noise"] = m[0], m[1]
    out["noise_snr_effect"] = [dict(decision=r.decision, noise=r.noise, population=r.population, size=r.size,
                                    median=_r(r.median, 3)) for r in e.dropna(subset=["decision"]).itertuples()]

    f = pd.read_csv(paths.NOISE_AND_SNR / PRED / "flip_attribution_da_size_by_axis_both_axes.csv")
    f = f[f["sd_scope"] == "size"]
    keep = ["population", "axes", "axis", "size", "threshold", "size_matched", "n_pairs", "n_comparisons",
            "share_a_b", "share_a_not_b", "share_not_a_b", "share_not_a_not_b", "share_b", "sd_source"]
    flips = [{k: (_r(x, 4) if isinstance(x, float) else x) for k, x in r.items()} for r in f[keep].to_dict("records")]
    d = pd.read_csv(paths.NOISE_AND_SNR / PRED / "flip_attribution_da_size_depth_by_seed_mono_axis.csv")
    d = d[d["L"] != "all"]
    seeds = [dict(size=r.size, L=int(r.L), deep_seed=int(r.deep_seed), n=int(r.n_comparisons), flips=int(r.n_b),
                  proxy=_r(r.median_proxy_diff, 4), ref=_r(r.median_ref_diff, 4)) for r in d.itertuples()]
    out["noise_snr_flips"] = dict(axes=flips, depth_by_seed=seeds)
    return out
