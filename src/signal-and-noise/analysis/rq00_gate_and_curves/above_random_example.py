"""The above-random gate on ONE benchmark-language task, drawn so a reader can
follow the rule (`above_random.py`) run by run.

The gate is computed per task (`hellaswag_ta`, `include_v2_og_hungarian_hungary`, never per family). Three
things are shown for the chosen task, on the runs of the pool whose mixture
trains the task's language (the runs the gate counts):

  (a) every run's score along training (x in Chinchilla multiples, one thin
      line per run coloured by size, the final checkpoint marked), the chance
      level 1/n_options, the score a run NEEDS for its one-sided 95 % Wilson
      lower bound over the task's n_items to clear chance, and for one
      highlighted run the band between its score and that bound — the bound,
      not the score, has to clear the chance line;
  (b) per size, every run's final score with its lower bound, in colour where
      the bound clears chance and grey where it does not, and the verdict per
      size: k of n runs pass, above random when k/n >= MIN_SHARE;
  (c) what the verdict does downstream: rq02's decision-accuracy cells of
      this task (`da_per_task.csv`, the multi-axis pairs) — DA-size from
      each proxy to the reference and DA-ckpt at 90 % of each run — drawn
      grey where the gate blanks them (at chance at the proxy, or at the
      reference for DA-size) and in colour where every RQ reads them.
  The rule itself is the three lines of the header, with its two caveats:
  chance is uniform guessing (a constant answer is not caught), and the runs
  of a size are correlated, not independent trials.

    above_random_example.png / .csv   per run (final score, n_items, LCB,
                                      passes), per size (share, verdict) and
                                      per rq02 cell (da, gated)

    python analysis/rq00_gate_and_curves/above_random_example.py --pool predictivity --task include_v2_og_hungarian_hungary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, GATE_AND_CURVES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import (  # noqa: E402
    ALPHA, MIN_SHARE, load_mask, scores_and_mask, task_chance, task_n_items, task_n_options, wilson_lcb)
from analysis.utils import TARGET_SIZE, assign_language, ladder_frame, on_shared_grid, one_axes, size_order  # noqa: E402

GITHUB = "https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis"
# hellaswag_es clears the gate at every size (the mask is 1 everywhere), so it
# shows nothing; hellaswag_ta is at chance up to 600M and above it from 1B, a
# clean 0/6 → 5/6 step. The default straddles the limit instead: 1, 2, 2, 3, 4
# of 6 runs pass across the sizes, so 2/6 is at chance and 3/6 is above.
TASK = "include_v2_og_hungarian_hungary"
mpl.rcParams.update(S.RC)


def needed_score(task: str) -> float:
    """The smallest accuracy over the task's n_items whose Wilson lower bound
    clears chance: the score a single run has to reach."""
    n, chance = int(task_n_items(task)), task_chance(task)
    ks = np.arange(n + 1)
    return ks[wilson_lcb(ks / n, np.full(n + 1, n)) > chance][0] / n


def figure(pool: str, task: str, highlight: str | None, out_dir: Path) -> pd.DataFrame:
    if np.isnan(task_n_options(task)) or np.isnan(task_n_items(task)):
        raise SystemExit(f"{task} has no chance level or item count (BPB, the loss, a generative task): the gate never reads it, "
                         f"so there is nothing to explain — pick a multiple-choice task")
    df = ladder_frame(pool)                       # the runs whose mixture trains the task's language (rule 2)
    d = df[(df["task"] == task) & on_shared_grid(df)].copy()
    _, mask, _, runs, share, _ = scores_and_mask(d, runs=True)
    chance, n_items, need = task_chance(task), int(task_n_items(task)), needed_score(task)
    sizes = size_order(runs["bucket"].unique())
    if highlight is None:                          # the 350M deep scheme-A run with the fewest languages
        c = d[(d["size"] == "350M") & (d["arch"] == "deep") & (d["scheme"] == "A")]
        highlight = c.loc[c["L"].idxmin(), "model"] if len(c) else runs["model"].iloc[0]
    lang = assign_language(task)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.6, 1, .8]})
    # (a) every run along training, the chance and needed-score lines, the highlighted run's bound
    for _, g in d.sort_values("frac").groupby("model"):
        size, arch, scheme = g[["size", "arch", "scheme"]].iloc[0]
        x = g["frac"] * G.CHINCHILLA_AT_FULL
        ax_a.plot(x, g["primary_score"], color=S.SIZE_COLOR[size], lw=1.0, alpha=.85, zorder=3)
        ax_a.plot(x.iloc[-1], g["primary_score"].iloc[-1], "o", ms=3.5, color=S.SIZE_COLOR[size], zorder=4)
    h = d[d["model"] == highlight].sort_values("frac")
    hx, hs = h["frac"] * G.CHINCHILLA_AT_FULL, h["primary_score"]
    hl = wilson_lcb(hs, np.full(len(h), n_items))
    ax_a.fill_between(hx, hl, hs, color=S.SERIES[1], alpha=.25, lw=0, zorder=2)
    ax_a.plot(hx, hs, color=S.SERIES[1], lw=2, zorder=5)
    ax_a.errorbar(hx.iloc[-1], hs.iloc[-1], yerr=[[hs.iloc[-1] - hl[-1]], [0]], color=S.SERIES[1], capsize=4, lw=1.6, zorder=6)
    ax_a.annotate(f"{highlight}\nfinal {hs.iloc[-1]:.3f}, LCB {hl[-1]:.3f} "
                  f"{'>' if hl[-1] > chance else '≤'} chance {chance:.2f} → {'passes' if hl[-1] > chance else 'fails'}",
                  (hx.iloc[-1], hl[-1]), xytext=(-8, -14), textcoords="offset points", ha="right", va="top",
                  fontsize=6.5, color=S.SERIES[1])
    ax_a.axhline(chance, color=S.INK, lw=.9, ls=":")
    box = dict(facecolor=S.SURFACE, edgecolor="none", alpha=.8, pad=1)
    ax_a.text(0.35, chance, f"chance 1/{int(1 / chance)} = {chance:.2f}", va="bottom", fontsize=6.5, color=S.INK, bbox=box, zorder=7)
    ax_a.axhline(need, color=S.MUTED, lw=.9, ls="--")
    ax_a.text(0.35, need, f"needed score at n = {n_items:,}: {need:.3f} (LCB > chance)", va="bottom", fontsize=6.5, color=S.MUTED, bbox=box, zorder=7)
    ax_a.set_xticks([1, 2, 3, 4, 5]); ax_a.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
    ax_a.set_xlim(0.3, 5.3); ax_a.set_xlabel("training tokens (× Chinchilla)"); ax_a.set_ylabel(f"{task} accuracy")
    ax_a.set_title(f"(a) every run that trains {lang}, along training", loc="left", fontsize=8.5)
    ax_a.legend(handles=[plt.Line2D([], [], color=S.SIZE_COLOR[s_], lw=2, label=s_) for s_ in sizes]
                + [plt.Line2D([], [], color=S.SERIES[1], lw=2, label="example run: score − LCB band")],
                fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    ax_a.grid(color=S.GRID, lw=.6); S.clean(ax_a)

    # (b) the finals per size with their bounds, passers in colour, the verdict per size
    runs = runs.assign(passes=runs["above"] == 1.0, chance=chance, needed_score=need).sort_values(["bucket", "model"])
    for i, s_ in enumerate(sizes):
        g = runs[runs["bucket"] == s_]
        xs = i + np.linspace(-.3, .3, len(g)) if len(g) > 1 else np.array([i])
        for x, (_, r) in zip(xs, g.iterrows()):
            c = S.SIZE_COLOR[s_] if r["passes"] else S.MUTED
            ax_b.plot([x, x], [r["lcb"], r["score"]], color=c, lw=1.2, alpha=.8, zorder=3)
            ax_b.plot(x, r["score"], "o", ms=4, color=c, zorder=4)
            ax_b.plot(x, r["lcb"], "_", ms=6, color=c, zorder=4)
        k, n = int(g["passes"].sum()), len(g)
        ok = k / n >= MIN_SHARE
        ax_b.text(i, g["score"].max() + 0.012 + 0.006 * (i % 2), f"{k} of {n} pass\n→ {'above random' if ok else 'at chance'}",
                  ha="center", va="bottom", fontsize=6.5, color=S.SIZE_COLOR[s_] if ok else S.MUTED, fontweight="bold" if ok else None)
    ax_b.axhline(chance, color=S.INK, lw=.9, ls=":")
    ax_b.axhline(need, color=S.MUTED, lw=.9, ls="--")
    ax_b.set_xticks(range(len(sizes))); ax_b.set_xticklabels(sizes); ax_b.set_xlim(-.6, len(sizes) - .4)
    ax_b.set_ylim(ax_a.get_ylim()[0], ax_a.get_ylim()[1] + 0.03)
    ax_b.set_xlabel("size"); ax_b.set_ylabel("final score, with its one-sided 95 % Wilson LCB")
    ax_b.set_title(f"(b) the gate per size: mask = {', '.join(f'{s_}:{int(mask.loc[task, s_])}' for s_ in sizes)}", loc="left", fontsize=8.5)
    ax_b.grid(color=S.GRID, lw=.6, axis="y"); S.clean(ax_b)

    # (c) the consequence: rq02's cells of this task, grey where the gate blanks them
    stage = load_pools()[pool].get("stage", "pretraining")
    da = one_axes(pd.read_csv(DECISION_ACCURACY / stage / pool / "da_per_task.csv"))
    da = da[da["task"] == task].iloc[0] if (da["task"] == task).any() else None
    proxies = [s_ for s_ in sizes if s_ != TARGET_SIZE]
    rows = [(f"DA-size\n(proxy final vs\n{TARGET_SIZE} final)", "decision_acc_size_{}", proxies,
             lambda s_: mask.loc[task, s_] == 0 or mask.loc[task, TARGET_SIZE] == 0),
            ("DA-ckpt\n(90 % of the run\nvs its own final)", "decision_acc_ckpt_f90_{}", sizes,
             lambda s_: mask.loc[task, s_] == 0)]
    cells = []
    for j, (label, col, szs, gated_at) in enumerate(rows):
        for i, s_ in enumerate(sizes):
            if s_ not in szs:
                continue
            v = float(da[col.format(s_)]) if da is not None and col.format(s_) in da.index else np.nan
            g = bool(gated_at(s_))
            cells.append({"kind": "cell", "task": task, "size": s_, "model": label.replace("\n", " "), "score": v, "passes": not g})
            face = S.MUTED if g else S.SEQ(0.25 + 0.7 * (v if v == v else 0))
            ax_c.add_patch(plt.Rectangle((i - .45, j - .4), .9, .8, color=face, alpha=.55 if g else .9, lw=0))
            ax_c.text(i, j, "gated" if g else (f"{v:.2f}" if v == v else "—"), ha="center", va="center", fontsize=7.5,
                      color=S.INK if g else ("white" if v == v and v > .55 else S.INK), fontweight=None if g else "bold")
    ax_c.set_xlim(-.6, len(sizes) - .4); ax_c.set_ylim(-.6, len(rows) + .1); ax_c.invert_yaxis()
    ax_c.set_xticks(range(len(sizes))); ax_c.set_xticklabels(sizes)
    ax_c.set_yticks(range(len(rows))); ax_c.set_yticklabels([r[0] for r in rows], fontsize=7)
    ok_size = [s_ for s_ in proxies if not (mask.loc[task, s_] == 0 or mask.loc[task, TARGET_SIZE] == 0)]
    ok_ckpt = [s_ for s_ in sizes if not mask.loc[task, s_] == 0]
    ax_c.set_title("(c) what rq02 reads for this task (multi-axis pairs): grey = gated", loc="left", fontsize=8.5)
    ax_c.text(0, len(rows) - .35, f"→ DA-size: only {', '.join(f'{s_} → {TARGET_SIZE}' for s_ in ok_size) or 'no cell'}; "
              f"DA-ckpt: only the {', '.join(ok_ckpt) or 'no'} run(s)", transform=ax_c.transData, ha="left", va="top",
              fontsize=6.8, color=S.INK)
    ax_c.tick_params(length=0); S.clean(ax_c, spines=())

    top = G._header(fig, f"The above-random gate on one task: {task} ({lang}, {int(1 / chance)} options, {n_items:,} items)",
                    f"the rule, per benchmark-language task — \n"
                    f"1. per run: one-sided 95 % Wilson lower bound (LCB) of the final "
                    f"accuracy over n_items > chance = 1/n_options (proportion_confint, alpha = {ALPHA}, two-sided {1 - ALPHA:.0%});\n"
                    f"2. per (task, size): at least {MIN_SHARE:.0%} of the size's runs that train the language pass, else the cell is at "
                    f"chance and every RQ reads it as grey;\n"
                    f"3. BPB, the loss and generative tasks have no chance level and are never gated (mc2's chance is its mean true-option share). "
                    f"Chance is UNIFORM guessing: a constant answer scores the majority gold label's share, which the gate does not test; "
                    f"the runs of a size answer the same items, so their verdicts are correlated. "
                    f"Population: the {len(runs)} runs whose "
                    f"mixture trains {lang}, {len(sizes)} sizes, one line per run in (a).\n "
                    f"(c) what the verdict means for rq02: a DA-size cell needs the proxy AND the reference above chance, "
                    f"a DA-ckpt cell needs the run's own size above chance; the grey cells are the decisions this task never enters.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, out_dir / "above_random_example.png", dpi=150)

    per_size = pd.DataFrame({"kind": "size", "size": sizes, "task": task,
                             "score": [runs.loc[runs["bucket"] == s_, "score"].mean() for s_ in sizes],
                             "n_items": n_items, "chance": chance, "needed_score": need,
                             "n_runs": [int((runs["bucket"] == s_).sum()) for s_ in sizes],
                             "n_pass": [int(runs.loc[runs["bucket"] == s_, "passes"].sum()) for s_ in sizes],
                             "share": [share.loc[task, s_] for s_ in sizes],
                             "passes": [bool(mask.loc[task, s_] == 1) for s_ in sizes]})
    per_run = runs.rename(columns={"bucket": "size"}).assign(kind="run", highlighted=runs["model"] == highlight)
    table = pd.concat([per_run, per_size, pd.DataFrame(cells)], ignore_index=True)[
        ["kind", "task", "size", "model", "highlighted", "score", "n_items", "lcb", "chance", "needed_score", "passes", "n_runs", "n_pass", "share"]]
    table.to_csv(out_dir / "above_random_example.csv", index=False)
    return table


def generate_readme(pool: str, task: str, t: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel, gh = f"{stage}/{pool}", f"{GITHUB}/rq00_gate_and_curves/{stage}/{pool}"
    per_size, per_run, cells = t[t["kind"] == "size"], t[t["kind"] == "run"], t[t["kind"] == "cell"]
    gated_n = int((~cells["passes"].astype(bool)).sum())
    ok_size = [r.size for r in cells.itertuples() if r.passes and r.model.startswith("DA-size")]
    ok_ckpt = [r.size for r in cells.itertuples() if r.passes and r.model.startswith("DA-ckpt")]
    hi = per_run[per_run["highlighted"]].iloc[0]
    chance, need, n_items = hi["chance"], hi["needed_score"], int(hi["n_items"])
    first = per_size.loc[per_size["passes"], "size"]
    verdicts = ", ".join(f"{r.size} {int(r.n_pass)}/{int(r.n_runs)} → {'above' if r.passes else 'at chance'}" for r in per_size.itertuples())
    body = "\n\n".join([
        f"### The gate on one task: `{task}`",
        f"An educational reading of the rule above on one benchmark-language task (the gate is per task, never per family). "
        f"(a) every run of `{pool}` that trains the language along training, the chance level and the score a run needs for its "
        f"Wilson lower bound to clear chance at this task's item count, with one run's score − LCB band; (b) per size the final "
        f"scores with their bounds, passers in colour, and the verdict; (c) the consequence — rq02's DA-size and DA-ckpt cells of the task, "
        f"grey where the gate blanks them. The rule is the header's three lines. "
        f"Regenerate with `python analysis/rq00_gate_and_curves/above_random_example.py --pool {pool} --task {task}`.",
        f"![The above-random gate on {task}]({rel}/above_random_example.png)",
        "Key findings:",
        "\n".join([
            f"- `{task}` has {int(1 / chance)} options ({chance:.2f} chance) and {n_items:,} items, so a single run needs {need:.3f} "
            f"(+{need - chance:.3f} over chance) for its LCB to clear chance; at n = {n_items:,} the bound sits "
            f"{hi['score'] - hi['lcb']:.3f} below the score.",
            f"- Verdict per size (runs passing / runs that train the language, ≥ {MIN_SHARE:.0%} needed): {verdicts}; "
            f"the task enters every RQ from **{first.iloc[0] if len(first) else 'no size'}** on and is grey below it.",
            f"- What that means in rq02 (panel c): a DA-size cell needs the proxy and the reference above chance, so `{task}` "
            f"contributes {'no DA-size cell' if not ok_size else 'only the ' + ', '.join(f'{s_} → {TARGET_SIZE}' for s_ in ok_size) + ' DA-size cell(s)'}; "
            f"a DA-ckpt cell needs the run's own size above chance, so it contributes {'no DA-ckpt cell' if not ok_ckpt else 'only the ' + ', '.join(ok_ckpt) + ' run(s)'}. "
            f"{gated_n} of {len(cells)} cells are blanked; the task is never a decision at {', '.join(s_ for s_ in per_size['size'] if s_ not in ok_ckpt) or 'no size'}.",
            f"- The highlighted run `{hi['model']}` ends at {hi['score']:.3f} with LCB {hi['lcb']:.3f}: "
            f"{'above' if hi['passes'] else 'not above'} chance — a score above the dotted line is not enough, the band's lower edge has to be.",
            f"- Population: {len(per_run)} runs over {per_size['size'].nunique()} sizes (rule 2: only the runs that train the language "
            f"count; a run scored on a language it never saw sits at chance and would drag the share down). The six runs of a size "
            f"answer the same items, so their verdicts are correlated, not six independent trials.",
            f"- What the gate is not: chance is uniform guessing (1/{int(1 / chance)}), so a run that always picks the majority gold "
            f"label is not caught (on `{task}` that scores the majority label's share, above {chance:.2f}); and the gate is not a "
            f"filter on DA — it blanks a cell by the gate alone, whatever the DA there "
            f"(DA-size per proxy: {'; '.join(f'{r.size} {r.score:.2f}' + ('' if r.passes else ' gated') for r in cells[cells.model.str.startswith('DA-size')].itertuples() if r.score == r.score)})."]),
        "Follow-ups:",
        "\n".join([
            f"- `--task hellaswag_ta`: the clean step (0/6 at 175M–600M, 5/6 at 1B, 6/6 at 1.7B), with no run near the limit.",
            f"- `--task hellaswag_eu`: a task whose verdict is not monotone in size (mask "
            f"{'/'.join(str(v) for v in load_mask(pool).loc['hellaswag_eu'])}), to show the share moving with the runs.",
            f"- The same figure on a {int(task_n_items('cultural_bench_easy_argentina'))}-item task (`cultural_bench_easy_argentina`), where the needed "
            f"margin is +{needed_score('cultural_bench_easy_argentina') - task_chance('cultural_bench_easy_argentina'):.2f}: the item count, not the model, decides.",
            "- A `--pool predictivity_all` version with the replicate seeds, to see how much the per-run verdict moves with the seed."]),
        f"Files: [`above_random_example.png`]({gh}/above_random_example.png), [`above_random_example.csv`]({gh}/above_random_example.csv)."])
    replace_block(GATE_AND_CURVES / "README.md", "above-random-example", body,
                  f"above_random_example.py --pool {pool} --task {task}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    p.add_argument("--task", default=TASK, help="a benchmark-language task with a chance level")
    p.add_argument("--highlight", default=None, help="the run whose LCB band is drawn (default: the 350M deep scheme-A run with the fewest languages)")
    args = p.parse_args()
    out = GATE_AND_CURVES / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out.mkdir(parents=True, exist_ok=True)
    t = figure(args.pool, args.task, args.highlight, out)
    print(t[t["kind"] == "size"].to_string(index=False))
    generate_readme(args.pool, args.task, t)
