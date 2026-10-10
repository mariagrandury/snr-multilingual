"""Data of the decision-accuracy page's additional views (RQ2, paper §4.2).

Every number is a row of an rq02 output, filtered and renamed, except the
pair explorer's (`decision_accuracy_pairs`): there the multi-axis DA-size of
the rq02 figures is split back into its decisions. For each pair of design
variants with a 1.7B run and each proxy size, it counts the tasks on which the
proxy's final checkpoint orders the pair like the 1.7B final, with the
pipeline's sign rule (`analysis.utils.pair_agreement`: a pair tied on both
sides agrees, tied on one side is a miss). The (task, size) cells are those of
the permutation-null table (gated at the proxy and at 1.7B, three pairs or
more), so summed over the pairs the counts are that table's n_matching and
n_pairs; build() asserts it.

    python3 scripts/build_site_data.py        # writes every page's JSON
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src/signal-and-noise", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from analysis import paths  # noqa: E402

PRED = "pretraining/predictivity"
DATA = {"A": "list A", "AT3": "list A, T = 3", "B": "list B", "ZH": "Chinese as 2nd language",
        "ES": "Spanish as 2nd language", "DCLMP": "DCLM without edu filter", "FWEB": "FineWeb English"}


def _rows(df: pd.DataFrame) -> list[dict]:
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
             if not (isinstance(v, float) and v != v)} for r in df.to_dict("records")]


def _label(r) -> str:
    act = " · SwiGLU" if r["activation"] == "swiglu" else ""
    return f"L{r['L']} · {r['arch']}{act} · {DATA.get(r['data'], r['data'])}"


def pairs() -> dict:
    from analysis.utils import (BBPB, SMALL_SIZES, TARGET_SIZE, design_axes, finals, ladder_frame,
                                moved_axes, pair_sets)
    null = pd.read_csv(paths.PERMUTATION_NULL / PRED / "permutation_null_da_size_per_task_both_axes.csv")
    null = null[(null["axes"] == "multi-axis") & ~null["gated"] & null["da"].notna()]   # gated = at chance
    cells = {s: set(g["task"]) for s, g in null.groupby("size")}
    tasks = sorted(set(null["task"]))

    df = ladder_frame("predictivity")
    attrs = design_axes(df)
    fams = sorted(set(df.loc[df["size"] == TARGET_SIZE, "family"]))
    pl = [(a, b) for a, b in pair_sets(attrs.loc[fams])["multi-axis"]]
    fin = finals(df[df["task"].isin(tasks)])
    score = fin.pivot_table(index=["size", "family"], columns="task", values="primary_score", aggfunc="first")
    score = score.reindex(columns=tasks)
    ia = np.array([fams.index(a) for a, _ in pl])
    ib = np.array([fams.index(b) for _, b in pl])

    def signs(size):
        x = score.loc[size].reindex(fams).to_numpy(dtype=float)
        return np.sign(x[ia] - x[ib])                       # pairs x tasks, NaN where a score is missing

    ref = signs(TARGET_SIZE)
    col = {t: i for i, t in enumerate(tasks)}
    bpb = sorted(t for t in tasks if t.startswith("bpb_"))      # the 50 trained languages, English as bpb_dclm
    kinds = {"acc": lambda t: not t.startswith(BBPB) and not t.startswith("bpb_"),
             "bbpb": lambda t: t.startswith(BBPB)}
    out = {"acc": {}, "bbpb": {}, "bpb": {}}
    ref_pick = {}
    for s in SMALL_SIZES:
        p = signs(s)
        ok = ~np.isnan(p) & ~np.isnan(ref)
        agree = ok & (p == ref)
        # the known answer: summed over the pairs, the null table's counts
        chk = null[null["size"] == s].set_index("task")
        idx = [col[t] for t in chk.index]
        assert (agree[:, idx].sum(0) == chk["n_matching"].to_numpy()).all(), f"n_matching differs at {s}"
        assert (ok[:, idx].sum(0) == chk["n_pairs"].to_numpy()).all(), f"n_pairs differs at {s}"
        for k, keep in kinds.items():
            c = [col[t] for t in sorted(cells[s]) if keep(t)]
            out[k][s] = np.stack([agree[:, c].sum(1), ok[:, c].sum(1)], 1).tolist()
        c = [col[t] for t in bpb]
        out["bpb"][s] = ["".join("." if not o else "1" if g else "0" for o, g in zip(oo, gg))
                         for oo, gg in zip(ok[:, c], agree[:, c])]
    # The 1.7B verdict per pair: on how many tasks the reference prefers a, prefers b,
    # or ties (higher accuracy is better, lower bBPB and BPB are), over the tasks the
    # 1B proxy reads (the widest gated set), and per language for BPB.
    better = {"acc": 1, "bbpb": -1}
    for k, keep in kinds.items():
        c = [col[t] for t in sorted(cells[SMALL_SIZES[-1]]) if keep(t)]
        r = ref[:, c] * better[k]
        ref_pick[k] = np.stack([(r > 0).sum(1), (r < 0).sum(1), (r == 0).sum(1)], 1).tolist()
    r = -ref[:, [col[t] for t in bpb]]
    ref_pick["bpb"] = ["".join("." if np.isnan(v) else "a" if v > 0 else "b" if v < 0 else "=" for v in row)
                       for row in r]

    fam_rows = [{"id": f, "label": _label(attrs.loc[f]), "L": int(attrs.loc[f, "L"]), "arch": attrs.loc[f, "arch"],
                 "activation": attrs.loc[f, "activation"], "data": attrs.loc[f, "data"]} for f in fams]
    return dict(sizes=list(SMALL_SIZES), reference=TARGET_SIZE, families=fam_rows,
                bpb_tasks=[t[4:] for t in bpb],
                pairs=[{"a": fams.index(a), "b": fams.index(b),
                        "axes": moved_axes(attrs.loc[a], attrs.loc[b]),
                        "acc": [out["acc"][s][i] for s in SMALL_SIZES],
                        "bbpb": [out["bbpb"][s][i] for s in SMALL_SIZES],
                        "bpb": [out["bpb"][s][i] for s in SMALL_SIZES],
                        "ref": {k: ref_pick[k][i] for k in ref_pick}}
                       for i, (a, b) in enumerate(pl)])


def extra() -> dict:
    """The per-size summaries the dot plot, the checkpoint slider and the
    decisive-pair view read."""
    da = paths.DECISION_ACCURACY / PRED
    summ = pd.read_csv(da / "bench_bpb_da_size_summary_multi_axes.csv")
    summ = summ[summ["group"] == "all benchmarks"][["size", "acc_acc", "acc_acc_n", "bbpb_acc", "bbpb_acc_n",
                                                    "bbpb_bbpb", "bbpb_bbpb_n"]]
    gate = pd.read_csv(da / "bench_bpb_da_size_by_acc_gate_multi_axes_paper.csv")
    gate = gate[gate["group"] == "all benchmarks"].drop(columns=["group", "axes"])
    early = pd.read_csv(da / "early_small_da_goal_summary_multi_axes.csv")
    seed = pd.read_csv(da / "seed_uncertainty_da_all_seed_null_paper.csv")
    seed = seed[seed["view"] == "seed null"][["size", "frac", "da", "da_real", "tasks"]]
    tok = paths.DA_VS_TRAIN_TOKENS / "pretraining/predictivity_seeds"
    ckpt_bpb = pd.read_csv(tok / "da_ckpt_multi_axes_across_langs_bpb.csv")[["proxy_size", "frac", "da", "n_languages"]]
    null = pd.read_csv(paths.PERMUTATION_NULL / PRED / "permutation_null_da_size_paper.csv")
    decisive = pd.read_csv(paths.DECISIVE_PAIRS / PRED / "decisive_pairs_da_size_paper.csv")
    return dict(channels=_rows(summ), acc_gate=_rows(gate), early=_rows(early), seed_null=_rows(seed),
                ckpt_bpb=_rows(ckpt_bpb), null=_rows(null), decisive=_rows(decisive))


def build() -> dict:
    return {"decision_accuracy_pairs": pairs(), "decision_accuracy_extra": extra()}
