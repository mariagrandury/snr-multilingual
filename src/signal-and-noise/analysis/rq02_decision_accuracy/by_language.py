"""Scale convergence conditioned on ONE language: for each of the eight
languages of the L8 setting, how small a fully trained model still decides
like the reference on THAT language's benchmarks — one line per language-count
regime that trains the language, so the lines within a panel differ in how
much of the language the models saw.

The lines are `scale_convergence.py --by L`'s (pairs of design variants sharing
the L, seed 1904, every scheme), read over the tasks in one language instead of
over every language at once. Within scheme A the share of a language is a
function of L alone (the lists are nested; ru is 15.0 % of every token at L8,
10.2 % at L30), so the legend carries the share and the tokens at the reference
size. Two things the reader must know:

  * tokens of a language = share(L, scheme) × D(N), the share being of ALL
    tokens (English is the other 50 %): a RELABELLING of the
    (L, size) grid, not a new measurement. Its one honest test is the collapse
    test of `_tokens` below.
  * an L regime pools arch, list and temperature decisions at once and the mix
    differs by regime (`share_*` columns); rule 5 forbids holding it fixed.

Two populations, two names — and which one supports inference:

    scale_convergence_lang_all       every gated task of the language (rules 1, 5).
                                     THE INFERENCE VERSION: no selection on DA.
    scale_convergence_lang_above_66_size
                                     the tasks reliable on DA-size (reliable_tasks.py),
                                     kept for continuity with the pooled paper figure.
                                     A cut on the very quantity drawn, so a line is
                                     biased up by construction; a conditional reading.

    <stem>.png / .csv        2 × 4 panels, x = non-embedding parameters; every point
                             carries its task count; the pooled `all pairs` line its
                             leave-one-family-out band
    <stem>_tokens.png / .csv the same decisions with x = tokens of that language the
                             proxies trained on (mean over the decision's two members):
                             do the L lines collapse onto one curve? The panel title
                             carries R² of one log-linear fit through every regime's
                             points under each x; a higher R² under tokens than under
                             size says exposure explains what language count does not.
                             `all pairs` has no place here (a cross-L pair has no one
                             exposure) and is left out.
    <stem>_coverage.csv      per (language, regime, size): families scored, decisions,
                             tasks clearing MIN_PAIRS, and why a cell draws nothing —
                             the table that says which panels CAN exist

    python analysis/rq02_decision_accuracy/by_language.py --pool predictivity
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

from evals.scripts.utils.configs import fineweb_language, load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import FILTERS, load_reliable  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import (  # noqa: E402
    L_COLOUR, OVERALL, POOL, TAU, aggregate, decisions, decorate, draw_lines, keep_cells, pair_axis,
    pairs_by_group, reliability)
from analysis.utils import (  # noqa: E402
    AXES_SUFFIX, GRID_SEED, MIN_PAIRS, NON_EMB, TARGET_SIZE, assign_language, design_axes, finals,
    ladder_frame, language_token_share, language_tokens, size_order)
from pretrain.launch_trainings import cell_fineweb_subsets  # noqa: E402

OUT_ROOT = DECISION_ACCURACY
# The eight languages of the L8 setting, English (the DCLM half) first, then in
# the list's own order — which is resource order.
LANGS = ["en"] + [fineweb_language(s) for s in cell_fineweb_subsets(8, "A")]
VARIANTS = ("", "above_66_size")
REGIMES = ["L1", "L2", "L8", "L15", "L30", "L50"]
mpl.rcParams.update(S.RC)


def coverage(dec: pd.DataFrame, lang_of: pd.Series, sizes: list) -> pd.DataFrame:
    """Per (language, regime, size): what the decision rows hold, and why a
    cell draws nothing. Read this before reading the panels."""
    d = dec.astype({"task": str, "group": str, "size": str, "family_a": str, "family_b": str})
    d["language"] = d["task"].map(lang_of)
    rows = []
    for lang in LANGS:
        for grp in REGIMES + [OVERALL]:
            for size in sizes:
                g = d[(d["language"] == lang) & (d["group"] == grp) & (d["size"] == size)]
                per_task = g.groupby("task").size()
                fams = len(set(g["family_a"]) | set(g["family_b"]))
                ok = int((per_task >= MIN_PAIRS).sum())
                why = ("" if ok else "no family scored" if not fams
                       else f"{fams} families: below MIN_PAIRS ({MIN_PAIRS}) on every task" if fams < 3
                       else f"no task with ≥ {MIN_PAIRS} comparable pairs")
                rows.append({"language": lang, "group": grp, "size": size, "families": fams,
                             "decisions": len(g), "tasks": int(per_task.size), "tasks_ok": ok, "why_empty": why})
    return pd.DataFrame(rows)


def tokens_axis(out: pd.DataFrame, dec: pd.DataFrame, cells: pd.DataFrame, pool: str,
                attrs: pd.DataFrame, lang: str) -> pd.DataFrame:
    """`out` with a `tokens` column: per (group, size), the mean over the kept
    decisions of the two members' training tokens in `lang`. The reference row
    gets the reference's own tokens. NaN where a build's record is unreachable."""
    kept = keep_cells(cells, pool)[["task", "group", "size", "frac"]]
    d = dec.astype({c: str for c in ("task", "group", "size", "family_a", "family_b")}).merge(kept)
    cache: dict = {}

    def tok(fam: str, size: str) -> float:
        key = (fam, size)
        if key not in cache:
            r = attrs.loc[fam]
            t = language_tokens(int(r["L"]), r["scheme"], size, r["arch"])
            cache[key] = float("nan") if t is None else t.get(lang, 0.0)
        return cache[key]

    d["tokens"] = [(tok(a, s) + tok(b, s)) / 2 for a, b, s in zip(d["family_a"], d["family_b"], d["size"])]
    t = d.groupby(["group", "size"])["tokens"].mean().reset_index()
    return out.merge(t, on=["group", "size"], how="left")


def collapse_r2(out: pd.DataFrame, x: str) -> float:
    """R² of one straight line R ~ log10(x) through every regime's proxy points
    at once. If exposure is what matters, the L lines fall onto one curve under
    x = tokens and the R² rises against x = size."""
    pts = out[(out["population"] == "all benchmarks") & (out["group"] != OVERALL) & (out["size"] != TARGET_SIZE)]
    pts = pts.dropna(subset=[x, "reliability"])
    pts = pts[pts[x] > 0]
    if len(pts) < 3 or pts["group"].nunique() < 2:
        return float("nan")
    lx, y = np.log10(pts[x].to_numpy()), pts["reliability"].to_numpy()
    fit = np.polyval(np.polyfit(lx, y, 1), lx)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - fit) ** 2).sum() / ss_tot) if ss_tot else float("nan")


def legend_labels(lang: str, groups: list) -> dict:
    """`L8 — ru 15.0 % of tokens — 25.0 B @1.7B`: scheme A's share (the lists
    are A's; AT3/B lines in the same regime have their own) and the tokens a
    deep scheme-A cell of the reference size saw."""
    out = {OVERALL: f"{OVERALL} (cross-L included)"}
    for grp in groups:
        if grp == OVERALL:
            continue
        L = int(grp[1:])
        share = language_token_share(L, "A")
        if share is None or lang not in share:
            out[grp] = f"{grp} — share unavailable"
            continue
        out[grp] = (f"{grp} — {lang} {share[lang]:.1%} of tokens — "
                    f"{language_tokens(L, 'A', TARGET_SIZE, 'deep')[lang] / 1e9:.1f} B @{TARGET_SIZE}")
    return out


def figure(per_lang: dict, cov: pd.DataFrame, path: Path, pool: str, variant: str, x: str, r2: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharey=True)
    flat = axes.ravel()
    sizes = size_order(cov["size"].unique())
    for ax, lang in zip(flat, LANGS):
        out = per_lang.get(lang)
        drawn = [] if out is None else [g for g in REGIMES if g in set(out["group"])]
        groups = ([OVERALL] if x == "non_emb" else []) + drawn
        colours = {g: L_COLOUR[g] for g in drawn}
        if out is not None and len(drawn):
            draw_lines(ax, out[out["population"] == "all benchmarks"], groups, colours, x, counts=True,
                       labels=legend_labels(lang, groups))
            ax.legend(fontsize=5.5, frameon=False, loc="lower right")
        else:
            c = cov[(cov["language"] == lang) & (cov["group"].isin(REGIMES)) & (cov["size"] != TARGET_SIZE)]
            why = c[c["why_empty"] != ""]["why_empty"].mode()
            ax.text(.5, .5, f"no regime draws a line\n{why.iloc[0] if len(why) else ''}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color=S.MUTED)
        title = lang
        if x == "tokens" and not np.isnan(r2.get(lang, (np.nan, np.nan))[0]):
            rs, rt = r2[lang]
            title += f"   R² of one line through every regime: {rs:.2f} (size) → {rt:.2f} (tokens)"
        elif out is not None:
            n = out[(out["group"] == OVERALL) & (out["size"] != TARGET_SIZE)]["n_tasks"]
            title += f"   ({int(n.max()) if len(n) else 0} tasks at most)"
        ax.set_title(title, loc="left", fontsize=8)
        ax.axhline(TAU, color=S.MUTED, lw=.8, ls=":"); ax.set_ylim(0, 1.0); ax.set_xscale("log")
        if x == "non_emb":
            ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes)
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    for ax in flat[4:]:
        ax.set_xlabel("non-embedding parameters (log)" if x == "non_emb" else "training tokens of this language (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"decision reliability vs {TARGET_SIZE} final")
    pop = ("every gated benchmark task of the language (rules 1, 5) — no selection on DA" if not variant else
           f"the tasks reliable on DA-size (DA ≥ {FILTERS[variant][1]:g}, {FILTERS[variant][0]} reduction): a cut on the "
           f"quantity drawn, so every line is biased up by construction — a conditional reading")
    top = G._header(fig, f"Scale convergence per language: how small a model still decides like {TARGET_SIZE} on "
                         f"{'each language of the L8 setting' if x == 'non_emb' else 'a language, against its exposure to it'}",
                    f"Panel = one language; line = pairs of design variants sharing that language count (seed {GRID_SEED}, "
                    f"every scheme), read on that language's benchmarks only; R = matching / comparable decisions against "
                    f"the {TARGET_SIZE} final, pooled over tasks; the number under a point is its task count. Population: {pop}. "
                    + (f"Legend: scheme A's share of the language and the tokens a deep {TARGET_SIZE} cell of that regime saw; "
                       f"AT3 (T=3) and B lines in the same regime saw less or more. Shaded band on `{OVERALL}` = "
                       f"leave-one-family-out jackknife, 90 %. " if x == "non_emb" else
                       f"x = mean over a decision's two members of the tokens of this language they trained on "
                       f"(share of all tokens × D(N)) — a relabelling of (L, size), not a new axis; the test is whether the "
                       f"regime lines collapse onto one curve, which the R² in each title measures. ")
                    + f"A regime pools arch, list and temperature decisions at once (`share_*` in the CSV). A regime "
                    f"with no line here is one whose members are not trained on this language or fall below "
                    f"{MIN_PAIRS} pairs — `_coverage.csv` says which. Gate and pair minimum as everywhere in rq02; "
                    f"pairs from {POOL}, gated with {pool}'s mask.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, per_lang: dict, r2: dict, stem: str) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rows = []
    for lang in LANGS:
        out = per_lang.get(lang)
        rec = {"language": lang}
        for grp in REGIMES:
            g = None if out is None else out[(out["population"] == "all benchmarks") & (out["group"] == grp)
                                             & (out["size"] != TARGET_SIZE)]
            rec[grp] = ("—" if g is None or not len(g) else
                        f"{g.sort_values('non_emb')['reliability'].iloc[0]:.2f}→{g.sort_values('non_emb')['reliability'].iloc[-1]:.2f} "
                        f"[{int(g['n_tasks'].max())}]")
        rs, rt = r2.get(lang, (np.nan, np.nan))
        rec["R² size / tokens"] = "—" if np.isnan(rs) else f"{rs:.2f} / {rt:.2f}"
        rows.append(rec)
    t = pd.DataFrame(rows)
    body = "\n\n".join([
        "## Scale convergence per language",
        f"For each language of the L8 setting, the `--by L` lines read on that language's benchmarks alone: "
        f"R at the smallest → largest proxy [tasks], per regime that trains the language, on every gated task (no "
        f"selection on DA — the inference version; the `above_66_size` twin is the conditional one). The last column is "
        f"the collapse test: R² of one log-linear line through every regime's points with x = model size, then with "
        f"x = tokens of the language (its share of all tokens × D(N)); a rise under tokens says exposure explains what language "
        f"count does not. Regimes pool arch, list and temperature decisions at once. Regenerate with "
        f"`python analysis/rq02_decision_accuracy/by_language.py --pool {pool}`; `{stem}_coverage.csv` says why a "
        f"cell is empty.",
        md_table(list(t.columns), t.values.tolist()),
        f"![Scale convergence per language]({stage}/{pool}/{stem}.png)",
        f"![Scale convergence per language, tokens axis]({stage}/{pool}/{stem}_tokens.png)"])
    replace_block(OUT_ROOT / "README.md", "scale-convergence-by-language", body, f"by_language.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose gate applies and whose folder receives the outputs")
    p.add_argument("--axes", default="multi-axis", choices=["multi-axis", "mono-axis"],
                   help="the pair set (rule 15); under mono-axis the regimes have no line below 1B (rule 5)")
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    df = ladder_frame(POOL)
    fin = finals(df)
    attrs = design_axes(df)
    groups, axis_of = pairs_by_group(attrs, "L", args.axes), pair_axis(attrs)
    sizes = size_order(fin["size"].unique())
    dec = decisions(fin.assign(frac=1.0), groups, sizes, fin)
    cells_all = reliability(dec)
    lang_of = pd.Series({t: assign_language(t) for t in cells_all["task"].unique()})
    for variant in VARIANTS:
        stem = f"scale_convergence_lang_{variant or 'all'}" + AXES_SUFFIX[args.axes]
        cells = cells_all
        if variant:
            keep = load_reliable(out_dir, variant, args.axes)
            if keep is None:
                continue
            cells = cells[cells["task"].isin(set(keep["task"]))]
        cov = coverage(dec[dec["task"].astype(str).isin(set(cells["task"]))], lang_of, sizes)
        cov.to_csv(out_dir / f"{stem}_coverage.csv", index=False)
        per_lang, per_lang_tok, r2, tables = {}, {}, {}, []
        for lang in LANGS:
            c = cells[cells["task"].map(lang_of) == lang]
            if not len(keep_cells(c, args.pool)):
                continue
            keys = ["population", "group", "size"]
            out = decorate(aggregate(c, args.pool, TAU), dec, c, args.pool, keys, axis_of)
            out = tokens_axis(out, dec, c, args.pool, attrs, lang)
            per_lang[lang] = out
            r2[lang] = (collapse_r2(out, "non_emb"), collapse_r2(out, "tokens"))
            # the panel title's two numbers, so the CSV carries what the PNG shows (rule 12)
            tables.append(out.assign(language=lang, collapse_r2_size=r2[lang][0], collapse_r2_tokens=r2[lang][1]))
        table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        table.to_csv(out_dir / f"{stem}.csv", index=False)
        table.drop(columns=["lo", "hi", "se"], errors="ignore").to_csv(out_dir / f"{stem}_tokens.csv", index=False)
        print(f"\n--- {stem}: {len(per_lang)} of {len(LANGS)} languages draw a panel")
        for lang in LANGS:
            g = cov[(cov["language"] == lang) & (cov["group"].isin(REGIMES)) & (cov["tasks_ok"] > 0)]
            print(f"  {lang:3s} regimes with a line: {sorted(set(g['group']), key=REGIMES.index) or '—'}"
                  + (f"   R² size {r2[lang][0]:.2f} / tokens {r2[lang][1]:.2f}" if lang in r2 and not np.isnan(r2[lang][0]) else ""))
        figure(per_lang, cov, out_dir / f"{stem}.png", args.pool, variant, "non_emb", r2)
        figure(per_lang, cov, out_dir / f"{stem}_tokens.png", args.pool, variant, "tokens", r2)
        if not variant and args.axes == "multi-axis":
            generate_readme(args.pool, out_dir, per_lang, r2, stem)
