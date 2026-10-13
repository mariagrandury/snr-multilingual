"""The tasks that clear chance at very small token counts of their language and
fail at intermediate ones: are their early passes a constant answer, a
preference for the longest option, or the tokens of a related language?

Selection, from tokens_seen.py's cell table
(`pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv`, seed 1904, every
build and ladder, every evaluated tenth): a task whose cells below LOW tokens
of its language are above chance in at least half of them, and whose cells
between LOW and MID are above chance in fewer than half, with MIN_CELLS cells
on each side.

The runs read: the deep scheme-A seed-1904 ladder, the one evaluated on every
language, at every evaluated tenth. A run is `trained, low` / `trained, mid`
by the tokens of the language its checkpoint had seen, or `untrained` (its
mixture lacks the language: no tokens of it), `related` when the mixture
trains a language of the same subfamily (`rq06 family_transfer.family`).
Its verdict is the gate's run-level Wilson test (`above_random.above_chance`).

Two tests, both from the stored per-option log-likelihoods (the harness's
samples files; the pick is the arg max under the task's metric, acc or
acc_norm):

  constant answer   `top_pick_share`, the share of items answered with the
                    run's most picked option, and `constant_acc`, the accuracy
                    that option would score on every item (the share of gold
                    answers at it). A pass is a constant answer when the run
                    picks one option on most items and that option alone clears
                    chance (`constant_above`): an unbalanced answer key on few
                    items is enough.
  longest option    `pick_longest`, the share of items where the pick is the
                    single longest option (the continuation, or for a letter
                    answer the option text in the prompt), against
                    `gold_longest`, the share where the gold answer is.
  related language  the untrained runs at the (size, checkpoint) points where
                    trained low-token runs exist, split by `related`: if the
                    unrelated ones pass as often, neither the language nor a
                    relative is needed for the pass.

!!! RULE 2: the untrained runs are read here as a control of the gate's verdict
(this folder's exemption: the gate covers every task); no score of theirs
enters a mean of the populations.

    early_passes_runs.csv       one row per (run, checkpoint, task) read
    early_passes.csv / .tex     per task: the pass shares by group and the two tests on the passing low-token runs

    python analysis/rq00_gate_and_curves/early_passes.py --pool predictivity
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import accuracy_metric, load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import CHANCE_VS_TRAIN_TOKENS, GATE_AND_CURVES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import above_chance, task_n_items  # noqa: E402
from analysis.rq06_language_transfer.family_transfer import LANGS, family  # noqa: E402
from analysis.rq08_subset_selection.build_per_item_store import files_for, gold_index  # noqa: E402
from analysis.utils import assign_language, benchmark_family, is_trained, language_tokens, ladder_frame, on_shared_grid  # noqa: E402
from pretrain.ladder_report import EVAL_LOGS  # noqa: E402
from pretrain.launch_trainings import cell_languages  # noqa: E402

GITHUB = "https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_gate_and_curves"
CELLS = CHANCE_VS_TRAIN_TOKENS / "pretraining" / "predictivity_seeds" / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv"
LOW, MID = 1e8, 1e10            # tokens of the language: "very small" below LOW, "intermediate" up to MID
MIN_CELLS = 10                  # cells a task needs on each side to be selected
CONSTANT = 0.8                  # a run answering one option on at least this share of items gives a constant answer
LADDER, DATA = "deep", "A"      # the ladder evaluated on every language
STEM = "early_passes"
LETTER = re.compile(r"^\s*([A-Z])\.\s?(.*)$")


def selected() -> list[str]:
    c = pd.read_csv(CELLS).dropna(subset=["train_tokens"])
    c = c[~c["benchmark"].str.startswith("bbpb") & (c["train_tokens"] < MID)]
    c["regime"] = np.where(c["train_tokens"] < LOW, "low", "mid")
    t = c.groupby(["task", "regime"])["above_chance"].agg(["mean", "size"]).unstack()
    ok = ((t[("mean", "low")] >= .5) & (t[("size", "low")] >= MIN_CELLS)
          & (t[("mean", "mid")] < .5) & (t[("size", "mid")] >= MIN_CELLS))
    return sorted(t.index[ok])


def runs(pool: str, tasks: list[str]) -> pd.DataFrame:
    """Every (run, tenth, task) of the deep scheme-A ladder on `tasks`, trained or not, with its group and verdict."""
    df = ladder_frame(pool, untrained=True)
    df = df[on_shared_grid(df) & df["task"].isin(tasks) & (df["ladder"] == LADDER) & (df["data"] == DATA)].copy()
    df["frac"] = (df["frac"] * 10).round() / 10
    df["language"] = df["task"].map(assign_language)
    df["trained"] = [is_trained(t, L, d) for t, L, d in zip(df["task"], df["L"], df["data"])]
    tok = {k: language_tokens(k[0], DATA, k[1], LADDER) or {} for k in set(zip(df["L"], df["size"]))}
    df["tokens"] = [tok[(L, s)].get(lang, 0.0) * f for L, s, lang, f in zip(df["L"], df["size"], df["language"], df["frac"])]
    subfams = {L: {family(x, "sub") for x in cell_languages(int(L), DATA) - {"en"}} - {None} for L in df["L"].unique()}
    df["related"] = [(not tr) and family(lang, "sub") in subfams[L] for tr, lang, L in zip(df["trained"], df["language"], df["L"])]
    df["group"] = np.where(df["trained"], np.select([df["tokens"] < LOW, df["tokens"] < MID], ["trained, low", "trained, mid"],
                                                    "trained, high"),
                           np.where(df["related"], "untrained, related", "untrained, unrelated"))
    df["above"] = above_chance(df["primary_score"].to_numpy(), df["task"].to_numpy()).to_numpy()
    return df[["model", "step", "task", "language", "size", "L", "data", "frac", "tokens", "group", "primary_score", "above"]]


def per_option(job: tuple[str, int, str]) -> dict | None:
    """One (run, checkpoint, task): the two tests' quantities, from its samples file."""
    model, step, task = job
    files = files_for(EVAL_LOGS / f"{model}-iter{step}", [task]).get(task)
    if not files:
        return None
    norm = accuracy_metric(task) == "acc_norm"
    picks, golds, pl, gl = [], [], [], []
    for path, _ in files:
        for line in open(path):
            o = json.loads(line)
            ll = np.array([float(r[0][0]) for r in o["resps"]])
            conts = [o["arguments"][f"gen_args_{i}"]["arg_1"] for i in range(len(ll))]
            lens = np.array([len(x.encode()) for x in conts], float)
            if len(set(lens)) == 1:          # a letter answer: the option texts are the prompt's lettered lines
                opts = [m[2] for m in map(LETTER.match, o["arguments"]["gen_args_0"]["arg_0"].split("\n")) if m]
                lens = (np.array([len(x.encode()) for x in opts[-len(ll):]], float) if len(opts) >= len(ll)
                        else np.full(len(ll), np.nan))
            p = int(np.argmax(ll / np.array([len(x) for x in conts]) if norm else ll))
            g = gold_index(o["target"])
            one = np.isfinite(lens).all() and (lens == lens.max()).sum() == 1
            picks.append(p); golds.append(g)
            pl.append(float(p == np.argmax(lens)) if one else np.nan)
            gl.append(float(g == np.argmax(lens)) if one else np.nan)
    picks, golds = np.array(picks), np.array(golds)
    top = int(np.bincount(picks).argmax())
    nan_mean = lambda v: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return {"model": model, "step": step, "task": task, "n_read": len(picks), "acc": float(np.mean(picks == golds)),
            "top_pick": top, "top_pick_share": float(np.mean(picks == top)), "constant_acc": float(np.mean(golds == top)),
            "pick_longest": nan_mean(np.array(pl)), "gold_longest": nan_mean(np.array(gl))}


def summary(r: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Per task: pass shares by group (untrained at the low-token points), and the tests on the passing low runs."""
    low_pts = set(zip(*[df.loc[df["group"] == "trained, low", k] for k in ("task", "size", "frac")]))
    at_low = df[[k in low_pts for k in zip(df["task"], df["size"], df["frac"])]]
    rows = []
    for task, g in r.groupby("task"):
        lo = g[(g["group"] == "trained, low") & (g["above"] == 1)]
        fam = benchmark_family(task)
        part = task.removeprefix(fam + "_")
        row = {"task": task, "benchmark": f"{G.paper_name(fam)} ({LANGS.get(part, {}).get('language', part.replace('_', ' ').title())})",
               "language": g["language"].iloc[0],
               "n_items": task_n_items(task)}
        for grp in ("trained, low", "trained, mid"):
            h = df[(df["task"] == task) & (df["group"] == grp)]["above"].dropna()
            row[f"pass_{grp.split(', ')[1]}"], row[f"n_{grp.split(', ')[1]}"] = h.mean(), len(h)
        for grp in ("untrained, related", "untrained, unrelated"):
            h = at_low[(at_low["task"] == task) & (at_low["group"] == grp)]["above"].dropna()
            key = grp.split(", ")[1]
            row[f"pass_untrained_{key}"], row[f"n_untrained_{key}"] = (h.mean() if len(h) else np.nan), len(h)
        row |= {"n_passing_low": len(lo), "top_pick_share_median": lo["top_pick_share"].median(),
                "share_constant": (lo["top_pick_share"] >= CONSTANT).mean(), "share_constant_above": lo["constant_above"].mean(),
                "acc_minus_constant_median": (lo["acc"] - lo["constant_acc"]).median(),
                "pick_longest_median": lo["pick_longest"].median(), "gold_longest": g["gold_longest"].median()}
        rows.append(row)
    return pd.DataFrame(rows)


def write_tex(s: pd.DataFrame, path: Path) -> None:
    f = lambda v, p=2: "--" if pd.isna(v) else f"{v:.{p}f}"
    lines = ["\\begin{tabular}{llrcccccccc}", "\\toprule",
             " & & & \\multicolumn{3}{c}{Share of runs above chance} & \\multicolumn{3}{c}{Passing low-token runs} "
             "& \\multicolumn{2}{c}{Longest option} \\\\",
             "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}",
             "Benchmark & Lang. & Items & Low & Mid & Untrained & Runs & One option & Constant passes & Picked & Gold \\\\",
             "\\midrule"]
    for _, r in s.iterrows():
        lines.append(" & ".join([r["benchmark"], r["language"], f"{r['n_items']:,.0f}".replace(",", "{,}"),
                                 f(r["pass_low"]), f(r["pass_mid"]), f(r["pass_untrained_unrelated"]), str(int(r["n_passing_low"])),
                                 f(r["top_pick_share_median"]), f(r["share_constant_above"]),
                                 f(r["pick_longest_median"]), f(r["gold_longest"])]) + " \\\\")
    path.with_suffix(".tex").write_text("\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""]))


def generate_readme(pool: str, s: pd.DataFrame, r: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    lo = r[(r["group"] == "trained, low") & (r["above"] == 1)]
    rows = [[f"`{x['task']}`", x["language"], f"{x['n_items']:.0f}", f"{x['pass_low']:.2f} ({x['n_low']})",
             f"{x['pass_mid']:.2f} ({x['n_mid']})", f"{x['pass_untrained_unrelated']:.2f} ({x['n_untrained_unrelated']})",
             x["n_passing_low"], f"{x['top_pick_share_median']:.2f}", f"{x['share_constant_above']:.2f}",
             f"{x['pick_longest_median']:.2f}", f"{x['gold_longest']:.2f}"] for _, x in s.iterrows()]
    body = "\n\n".join([
        "## The early passes: a constant answer, the longest option, or a related language?",
        f"**Gate verdicts per run · deep scheme-A seed-1904 ladder, every evaluated tenth, trained and untrained "
        f"languages (a rule 2 opt-out, since this is a control of the gate, not a population).** "
        f"The tasks are chosen on the seed-1904 cell table of `../rq00_chance_vs_train_tokens` (every build and ladder). "
        f"A task is chosen when it is above chance in at least half of its cells below {LOW:.0e} tokens of the language "
        f"and in fewer than half of those between {LOW:.0e} and {MID:.0e}, with at least {MIN_CELLS} cells on each side. "
        f"{len(s)} tasks qualify. For each task the table gives the share of runs above chance with few (`low`) and "
        f"intermediate (`mid`) tokens of the language. It also gives that share for the untrained runs (no tokens of the "
        f"language and no trained language of its subfamily) at the same (size, checkpoint) points as the low ones. "
        f"The group `untrained, related` has {int(s['n_untrained_related'].sum())} runs. "
        f"Over the passing low-token runs, it then gives the median share of items answered with the run's most picked "
        f"option, the share of those runs whose most picked option alone would clear chance, and the median share of "
        f"picks and of gold answers on the single longest option. Regenerate with "
        f"`python analysis/rq00_gate_and_curves/early_passes.py --pool {pool}`.",
        md_table(["task", "lang.", "items", "pass low (runs)", "pass mid (runs)", "pass untrained (runs)",
                  "passing low runs", "one-option share", "constant passes", "picks longest", "gold longest"], rows),
        f"Over the {len(lo)} passing low-token runs: {(lo['top_pick_share'] >= CONSTANT).mean():.0%} answer one option on "
        f"at least {CONSTANT:.0%} of the items, and in {lo['constant_above'].mean():.0%} that option alone clears chance.",
        f"Files: [`{STEM}.csv`]({GITHUB}/{rel}/{STEM}.csv) / [`.tex`]({GITHUB}/{rel}/{STEM}.tex), "
        f"[`{STEM}_runs.csv`]({GITHUB}/{rel}/{STEM}_runs.csv)."])
    replace_block(GATE_AND_CURVES / "README.md", "early-passes", body, f"early_passes.py --pool {pool}")


def main(pool: str, out_dir: Path, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("!!! RULE 2: untrained runs are read as a control of the gate verdict; none enters a population mean")
    tasks = selected()
    print(f"{len(tasks)} tasks pass early and fail at intermediate tokens: {', '.join(tasks)}")
    df = runs(pool, tasks)
    tr = df[df["group"].isin(["trained, low", "trained, mid"])]
    with Pool(workers) as p:
        got = [x for x in p.imap_unordered(per_option, sorted(set(zip(tr["model"], tr["step"], tr["task"]))), chunksize=4) if x]
    r = pd.DataFrame(got).merge(tr, on=["model", "step", "task"])
    r["constant_above"] = above_chance(r["constant_acc"].to_numpy(), r["task"].to_numpy()).to_numpy()
    r.sort_values(["task", "model", "step"]).to_csv(out_dir / f"{STEM}_runs.csv", index=False)
    s = summary(r, df)
    s.to_csv(out_dir / f"{STEM}.csv", index=False)
    write_tex(s, out_dir / STEM)
    print(s.round(2).to_string(index=False))
    if out_dir == GATE_AND_CURVES / load_pools()[pool].get("stage", "pretraining") / pool:
        generate_readme(pool, s, r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=None, help="write elsewhere (a test run)")
    a = ap.parse_args()
    main(a.pool, Path(a.out_dir) if a.out_dir else GATE_AND_CURVES / load_pools()[a.pool].get("stage", "pretraining") / a.pool,
         a.workers)
