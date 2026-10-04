"""Melt decision-accuracy values into a long per-(language, benchmark) table.

One step *back* from rq04's analyze_snr_variants.py (which correlates SNR
variants with DA): this script exposes the raw decision-accuracy values so you
can read off, per language, which benchmarks are most predictive across sizes.

It reads the `decision_acc_*` columns of rq02's `da_per_task.csv` (compute_da.py,
with the pair counts in `da_n_pairs_per_task.csv`) and reshapes them to long
form. Two DA definitions live there:

  DA-size  — small-bucket ranking @last vs the reference's ranking @last
             (`decision_acc_size_<small>` is small→TARGET_SIZE; the
             `decision_acc_size_<small>_to_<large>` columns are the scaling
             ladder and are never pooled into "DA-size").
  DA-ckpt  — within one bucket, the ranking at an early checkpoint (each of
             the nine evaluated tenths before the final, rule 3) vs that
             bucket's final ranking.

Every cell is over at least MIN_PAIRS pairs (rule 5, the kernel). The README
means are over the benchmark tasks that pass the above-random gate at the
proxy and, for DA-size, at the reference (rule 1, `utils.passes_gate`: a task
without a chance level passes), and over the per-language BPB tasks; the two
whole-mixture aggregates (`bpb_macro`, `train_loss`) are reported on their own
line and never enter a mean (rule 7).

Outputs, next to `da_per_task.csv`: the long table, the per-family grids and
the README block.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from evals.scripts.utils.configs import (  # noqa: E402
    bucket_order, load_languages, load_pools)
from analysis.rq00_gate_and_curves.above_random import (  # noqa: E402
    TABLE_STYLE, above_random_slides, fmt_cell, md_table)
from analysis.autodoc import CANONICAL_POOL  # noqa: E402
from analysis.utils import (LANGUAGE_AGGREGATES, one_axes, passes_gate,  # noqa: E402
    _BUCKET_RE, TARGET_SIZE, assign_language, benchmark_family)
from snr.constants import PLOT_DIR  # noqa: E402
from analysis.paths import DECISION_ACCURACY

# Slidev deck the appendix slides are written into (repo-root/documents).
_SLIDES = Path(__file__).resolve().parents[4] / "documents" / "slides.md"

_SIZE_CANON = re.compile(rf"^decision_acc_size_({_BUCKET_RE})$")
_SIZE_SCALE = re.compile(rf"^decision_acc_size_({_BUCKET_RE})_to_({_BUCKET_RE})$")
_CKPT = re.compile(rf"^decision_acc_ckpt_(f\d+)_({_BUCKET_RE})$")


def melt_da(df: pd.DataFrame) -> pd.DataFrame:
    """Long table of every non-NaN decision-accuracy cell in the per-task CSV."""
    rows = []
    for task in df.index:
        lang = assign_language(task)
        if lang in LANGUAGE_AGGREGATES:     # rule 7: bpb_macro / train_loss are the README's own lines, not a language's row
            continue
        bench = benchmark_family(task)
        for col in df.columns:
            val = df.at[task, col]
            if pd.isna(val):
                continue
            if (m := _SIZE_SCALE.match(col)):
                da_def, comparison = "DA-size", f"{m.group(1)}→{m.group(2)}"
                frm, to = m.group(1), m.group(2)
            elif (m := _SIZE_CANON.match(col)):
                da_def, comparison = "DA-size", f"{m.group(1)}→{TARGET_SIZE}"
                frm, to = m.group(1), TARGET_SIZE
            elif (m := _CKPT.match(col)):
                da_def, comparison = "DA-ckpt", f"{m.group(1)}@{m.group(2)}"
                frm, to = m.group(2), m.group(2)
            else:
                continue
            rows.append({"language": lang, "benchmark": bench, "task": task,
                         "da_def": da_def, "comparison": comparison,
                         "size_from": frm, "size_to": to,
                         "decision_acc": float(val)})
    return pd.DataFrame(rows)


def _pivot(long: pd.DataFrame, da_def: str) -> pd.DataFrame:
    """Wide view for one DA definition: rows=(language, benchmark, task),
    cols=comparison, sorted by mean DA so the most-predictive benchmarks sit on
    top within each language."""
    sub = long[long["da_def"] == da_def]
    if sub.empty:
        return pd.DataFrame()
    wide = sub.pivot_table(index=["language", "benchmark", "task"],
                           columns="comparison", values="decision_acc")
    wide = wide.assign(_mean=wide.mean(axis=1))
    wide = wide.sort_values(["language", "_mean"], ascending=[True, False])
    return wide.drop(columns="_mean").reset_index()


def run(pool: str, out_dir: Path) -> None:
    csv_path = out_dir / "da_per_task.csv"
    df = one_axes(pd.read_csv(csv_path)).set_index("task")
    long = melt_da(df)
    long = long.sort_values(["da_def", "language", "benchmark", "comparison"])

    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_dir / "da_per_benchmark.csv", index=False)
    _pivot(long, "DA-size").to_csv(out_dir / "da_per_benchmark_size.csv", index=False)
    _pivot(long, "DA-ckpt").to_csv(out_dir / "da_per_benchmark_ckpt.csv", index=False)

    n_size = (long["da_def"] == "DA-size").sum()
    n_ckpt = (long["da_def"] == "DA-ckpt").sum()
    print(f"Pool '{pool}': {len(long)} DA cells "
          f"({n_size} DA-size, {n_ckpt} DA-ckpt) over "
          f"{long['language'].nunique()} languages × {long['benchmark'].nunique()} benchmarks")
    print(f"  DA-size comparisons present: {sorted(long.loc[long.da_def=='DA-size','comparison'].unique())}")
    print(f"  DA-ckpt comparisons present: {sorted(long.loc[long.da_def=='DA-ckpt','comparison'].unique())}")
    print(f"Wrote {out_dir/'da_per_benchmark.csv'} (+ _size / _ckpt pivots)")

    generate_slides(long, pool)
    generate_readme(df, pool, out_dir)


def generate_readme(df: pd.DataFrame, pool: str, out_dir: Path) -> None:
    """Highlight + results blocks of the rq02 README (canonical pool only):
    mean DA-size per proxy over the above-random benchmark tasks and over the
    per-language BPB tasks, a family x proxy heatmap, and DA-ckpt per bucket
    and fraction. Pair counts come from da_n_pairs_per_task.csv."""
    if pool != CANONICAL_POOL:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from analysis import style as S
    from analysis.autodoc import fmt, md_table as md_tbl, replace_block
    from analysis.rq00_gate_and_curves.above_random import load_mask
    stage = load_pools()[pool].get("stage", "pretraining")
    npairs_path = out_dir / "da_n_pairs_per_task.csv"
    npairs = one_axes(pd.read_csv(npairs_path)).set_index("task") if npairs_path.is_file() else None
    mask = load_mask(pool)
    # the two aggregates are DA proxies of their own, not members of a mean
    is_bench = ~df.index.str.startswith("bpb_") & (df.index != "train_loss")
    is_bpb = df.index.str.startswith("bpb_") & (df.index != "bpb_macro")
    fam = pd.Series(df.index.map(benchmark_family), index=df.index)
    sizes = [b for b in bucket_order() if f"decision_acc_size_{b}" in df.columns
             and df[f"decision_acc_size_{b}"].notna().any()]
    bullets, rows = [], []
    for b in sizes:
        col = f"decision_acc_size_{b}"
        gated = passes_gate(mask, df.index, b, TARGET_SIZE).to_numpy() & is_bench   # rule 1: at the proxy and at the reference; NA passes
        bench, bpb = df.loc[gated & is_bench, col].dropna(), df.loc[is_bpb, col].dropna()
        med_n = int(npairs[col].reindex(bench.index).median()) if npairs is not None and len(bench) else 0
        rows.append([f"{b} → {TARGET_SIZE}", fmt(bench.mean()), len(bench), med_n, fmt(bpb.mean()), len(bpb)])
    if rows:
        bullets.append("- **DA-size, proxy → " + TARGET_SIZE + "** (mean over the above-random benchmark tasks / over the "
                       "per-language BPB tasks): " + "; ".join(f"{r[0]} {r[1]} / {r[4]}" for r in rows) + ".")
        for agg in ("bpb_macro", "train_loss"):
            if agg in df.index:
                bullets.append(f"- **DA-size of `{agg}`** (one task, kept out of the means above): "
                               + "; ".join(f"{b} {fmt(df.at[agg, f'decision_acc_size_{b}'])}" for b in sizes) + ".")
    ckpt = [c for c in df.columns if c.startswith("decision_acc_ckpt_")]
    ck_rows = []
    if ckpt:
        piv = {}
        for c in ckpt:
            _, _, _, frac, bucket = c.split("_", 4)
            gated = passes_gate(mask, df.index, bucket).to_numpy() & is_bench   # rule 1: DA-ckpt ranks within the size
            piv[(bucket, frac)] = df.loc[gated & is_bench, c].mean()
        buckets = [b for b in bucket_order()          # a bucket with one cell has no pairs: no row
                   if any(k[0] == b and np.isfinite(v) for k, v in piv.items())]
        fracs = sorted({k[1] for k in piv}, key=lambda f: int(f[1:]))
        ck_rows = [[b] + [fmt(piv.get((b, f), float("nan"))) for f in fracs] for b in buckets]
        best = max(piv.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else -1)
        bullets.append(f"- **DA-ckpt** (early checkpoint vs final, above-random benchmark tasks): highest at "
                       f"{best[0][0]} {int(best[0][1][1:])} % ({fmt(best[1])}).")
    blocks = []
    if rows:
        blocks += ["**DA-size by proxy size** (`n` tasks; median pairs per cell):",
                   md_tbl(["comparison", "benchmarks", "n", "pairs", "BPB", "n"], rows)]
        # family x proxy heatmap over the above-random tasks
        fams = sorted(fam[is_bench].unique())
        mat = np.full((len(fams), len(sizes)), np.nan)
        for j, b in enumerate(sizes):
            col = f"decision_acc_size_{b}"
            gated = passes_gate(mask, df.index, b, TARGET_SIZE).to_numpy() & is_bench   # rule 1: at the proxy and at the reference; NA passes
            g = df.loc[gated & is_bench, col].groupby(fam).mean()
            for i, f in enumerate(fams):
                if f in g.index and np.isfinite(g[f]):
                    mat[i, j] = g[f]
        keep = ~np.isnan(mat).all(axis=1)
        fams, mat = [f for f, k in zip(fams, keep) if k], mat[keep]
        if fams:
            fig, ax = plt.subplots(figsize=(1.4 * len(sizes) + 3, 0.32 * len(fams) + 1.2))
            im = ax.imshow(mat, vmin=0, vmax=1, cmap=S.SEQ, aspect="auto")
            for i in range(len(fams)):
                for j in range(len(sizes)):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if mat[i, j] > 0.7 else S.INK)
            ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f"{b}→{TARGET_SIZE}" for b in sizes])
            ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=8)
            ax.set_title("DA-size per family, mean over its above-random tasks", loc="left")
            S.clean(ax, spines=()); ax.tick_params(length=0)
            fig.colorbar(im, ax=ax, fraction=0.04, label="decision accuracy")
            pd.DataFrame(mat, index=fams, columns=sizes).rename_axis("family").to_csv(out_dir / "da_size_by_family.csv")
            S.save(fig, out_dir / "da_size_by_family.png", dpi=150)
            blocks.append(f"![DA-size by family]({stage}/{pool}/da_size_by_family.png)")
    if ck_rows:
        blocks += ["**DA-ckpt by bucket and fraction of the run** (mean over the above-random benchmark tasks):",
                   md_tbl(["bucket"] + [f"{int(f[1:])} %" for f in fracs], ck_rows)]
    readme = DECISION_ACCURACY / "README.md"
    gen = f"da_per_benchmark.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + "\n".join(bullets), gen)
    replace_block(readme, "results", "## Results\n\n"
                  + f"Numbers from the `{pool}` pool (`da_per_task.csv`, pairs from `da_n_pairs_per_task.csv`, "
                  f"gate from rq00). Regenerate with `python analysis/rq02_decision_accuracy/da_per_benchmark.py --pool {pool}`.\n\n"
                  + "\n\n".join(blocks), gen)
    print(f"Wrote auto README blocks → {readme}")


# --- Slidev appendix slides -------------------------------------------------
# The above-random slides live in above_random.py (imported above); this module
# owns the per-language decision-accuracy slides and stitches the full block.

_DA_BOLD = 0.75         # bold decision-accuracy cells at/above this
_BEGIN = "<!-- BEGIN generated signal slides (analysis/rq02_decision_accuracy/da_per_benchmark.py) -->"
_END = "<!-- END generated signal slides -->"

# Display names from configs/languages.json where curated; the tag otherwise.
_LANG_NAME = {code: e["language"] for code, e in load_languages()["languages"].items()}


def _comparison_key(comp: str) -> tuple[int, int]:
    """Order 'A→B' columns by (small-bucket, large-bucket) ladder position."""
    order = bucket_order()
    a, b = comp.split("→")
    bi = lambda x: order.index(x) if x in order else 99
    return bi(a), bi(b)


def _da_language_slides(long: pd.DataFrame) -> list[str]:
    """One DA-size slide per language: benchmark rows × every computable size
    pair, cell = decision accuracy (bold ≥ _DA_BOLD), most-predictive first."""
    size = long[long["da_def"] == "DA-size"]
    # The ladder resolves 96 languages, but only the ones it actually pretrains
    # on carry a claim. `groups.trained` is that set, capped at the 50-language
    # setting because the 100-language distribution is not settled. The CSVs keep
    # every language.
    report_langs = [l for l in load_languages()["groups"]["trained"]
                    if l in set(size["language"])]
    slides = []
    for lang in sorted(report_langs, key=lambda l: (l != "en", l)):
        sub = size[size["language"] == lang]
        comps = sorted(sub["comparison"].unique(), key=_comparison_key)
        wide = sub.pivot_table(index=["benchmark", "task"], columns="comparison",
                               values="decision_acc").reindex(columns=comps)
        wide = (wide.assign(_m=wide.mean(axis=1))
                .sort_values("_m", ascending=False).drop(columns="_m"))
        # Disambiguate duplicate family labels within a language by task token.
        fam_counts = wide.index.get_level_values("benchmark").value_counts()

        header = ["benchmark"] + comps
        rows = []
        for (fam, task), r in wide.iterrows():
            lbl = fam if fam_counts[fam] == 1 else task
            cells = [f"`{lbl}`"]
            for c in comps:
                v = r[c]
                txt = fmt_cell(v)
                if txt and v >= _DA_BOLD:
                    txt = f"**{txt}**"
                cells.append(txt)
            rows.append(cells)

        name = _LANG_NAME.get(lang, lang)
        slides.append(
            f"---\n"
            f"title: Appendix — Decision accuracy across sizes\n"
            f"subtitle: \"{name} ({lang}) · small→large size pair (bold ≥ {_DA_BOLD})\"\n"
            f"---\n\n"
            f"{md_table(header, rows)}\n\n"
            f"{TABLE_STYLE}\n"
        )
        # The same table as a picture, for reading a language at a glance.
        # documents/figures/fig_appendix.py renders these from the same CSV.
        slides.append(_figure_slide(
            f"/ladder/appendix/da_{lang}.png",
            "Appendix — Decision accuracy across sizes",
            f"{name} ({lang}) · the table before, as a heatmap"))
    return slides


def _figure_slide(image: str, title: str, subtitle: str, height: str = "72vh") -> str:
    return (f"---\n"
            f"layout: figure\n"
            f"image: {image}\n"
            f"fit: contain\n"
            f"height: {height}\n"
            f"title: {title}\n"
            f"subtitle: \"{subtitle}\"\n"
            f"---\n")


def _overview_slides() -> list[str]:
    """The two aggregate views that open the appendix: the grid collapsed over
    languages, then over benchmarks."""
    return [
        _figure_slide("/ladder/appendix/da_by_benchmark.png",
                      "Appendix — Decision accuracy, all languages at once",
                      "Benchmark × size pair, averaged over every language it covers"),
        _figure_slide("/ladder/appendix/da_by_language.png",
                      "Appendix — Decision accuracy, all benchmarks at once",
                      "Language × size pair, averaged over every benchmark it has",
                      height="78vh"),
    ]


def generate_slides(long: pd.DataFrame, pool: str) -> None:
    """Rewrite the deck's appendix (between BEGIN/END markers) from `long`:
    the above-random slides + 1 DA-size slide per language. Canonical pool
    only, like every other generator; idempotent — replaces an existing
    block, else appends to slides.md."""
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    ar, lang_slides = above_random_slides(stage), _da_language_slides(long)
    overview = _overview_slides()
    block = "\n".join([
        _BEGIN,
        "",
        "---\nlayout: section\n---\n\n"
        "# Appendix — Signal & Predictability across Sizes\n",
        "",
        *ar,
        *overview,
        *lang_slides,
        _END,
    ]) + "\n"

    text = _SLIDES.read_text()
    if _BEGIN in text and _END in text:
        text = re.sub(re.escape(_BEGIN) + r".*?" + re.escape(_END), block.rstrip(),
                      text, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + block
    _SLIDES.write_text(text)
    print(f"Wrote appendix slides → {_SLIDES} "
          f"({len(ar)} above-random + {len(overview)} overview + "
          f"{len(lang_slides)} per-language DA)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default=CANONICAL_POOL,
                   help=f"Pool name from configs/models.json (default: {CANONICAL_POOL}).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    run(pool=args.pool, out_dir=DECISION_ACCURACY / stage / args.pool)


if __name__ == "__main__":
    main()
