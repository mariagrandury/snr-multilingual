"""Melt decision-accuracy values into a long per-(language, benchmark) table.

One step *back* from analyze_snr_variants.py (which correlates SNR variants
with DA): this script just exposes the raw decision-accuracy values so you can
read off, per language, which benchmarks are most predictive across sizes.

It reads the `decision_acc_*` columns already computed in
`snr_definition/<stage>/<pool>/snr_variants_per_task.csv` and reshapes them to
long form. Two DA definitions live in that CSV:

  DA-size  — small-bucket ranking @last vs large-bucket ranking @last
             (`decision_acc_size_<small>` is small→TARGET_SIZE; the
             `decision_acc_size_<small>_to_<large>` columns are the scaling
             ladder). Answers "can size A predict the ranking at the larger
             size B?".
  DA-ckpt  — within a single bucket, early-ckpt ranking vs that bucket's
             max-ckpt ranking (`decision_acc_ckpt_<frac>_<bucket>`). Answers
             "can an early checkpoint predict the final ranking at this size?".

**Coverage caveat (cross-size DA):** DA-size needs ≥2 model *families* present
at BOTH buckets. The custom Apertus ladder stops at 1B (9 mix×seed families
spanning 175M…1B → well-powered), so every sub-1B → >1B pair has no shared
family and is absent. Above 1B only reference models span sizes, and most
bucket-pairs share just 2 families (DA is then binary 0/1). So the size pairs
present here are exactly the computable ones — the gaps are not bugs.

Outputs (under `snr_definition/<stage>/<pool>/`):
- `da_per_benchmark.csv`        — long: one row per (language, benchmark, task,
                                  da_def, comparison) with the DA value.
- `da_per_benchmark_size.csv`   — wide pivot, rows=(language, benchmark),
                                  cols=size comparison.
- `da_per_benchmark_ckpt.csv`   — wide pivot, rows=(language, benchmark),
                                  cols=ckpt comparison (frac@bucket).

`generate_slides()` additionally rewrites the data-driven appendix of the
Slidev deck (`documents/slides.md`, between BEGIN/END markers — idempotent):
2 above-random slides (custom / all models) + 1 DA-size slide per language.

    python analysis/rq01_decision_accuracy/da_per_benchmark.py --pool custom_swissai_hf
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

from evals.scripts.utils.configs import bucket_order, load_pools  # noqa: E402
from analysis.rq00_acc_vs_flops.above_random import (  # noqa: E402
    TABLE_STYLE, above_random_slides, fmt_cell, md_table)
from analysis.utils import (  # noqa: E402
    _BUCKET_RE, TARGET_SIZE, assign_language, benchmark_family)
from snr.constants import PLOT_DIR  # noqa: E402
from analysis.paths import SNR_DEFINITION

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
    csv_path = out_dir / "snr_variants_per_task.csv"
    df = pd.read_csv(csv_path, index_col="task")
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


# --- Slidev appendix slides -------------------------------------------------
# The above-random slides live in above_random.py (imported above); this module
# owns the per-language decision-accuracy slides and stitches the full block.

_DA_BOLD = 0.75         # bold decision-accuracy cells at/above this
_BEGIN = "<!-- BEGIN generated signal slides (analysis/rq01_decision_accuracy/da_per_benchmark.py) -->"
_END = "<!-- END generated signal slides -->"

_LANG_NAME = {
    "en": "English", "es": "Spanish", "ar": "Arabic", "zh": "Chinese",
    "ru": "Russian", "hi": "Hindi", "vi": "Vietnamese", "eu": "Basque",
    "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ja": "Japanese",
    "de": "German", "fr": "French",
}


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
    slides = []
    for lang in sorted(size["language"].unique(), key=lambda l: (l != "en", l)):
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
    return slides


def generate_slides(long: pd.DataFrame, pool: str) -> None:
    """Rewrite the deck's appendix (between BEGIN/END markers) from `long`:
    2 above-random slides (custom / all models) + 1 DA-size slide per language.
    Idempotent — replaces an existing block, else appends to slides.md."""
    stage = load_pools()[pool].get("stage", "pretraining")
    block = "\n".join([
        _BEGIN,
        "",
        "---\nlayout: section\n---\n\n"
        "# Appendix — Signal & Predictability across Sizes\n",
        "",
        *above_random_slides(stage),
        *_da_language_slides(long),
        _END,
    ]) + "\n"

    text = _SLIDES.read_text()
    if _BEGIN in text and _END in text:
        text = re.sub(re.escape(_BEGIN) + r".*?" + re.escape(_END), block.rstrip(),
                      text, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + block
    _SLIDES.write_text(text)
    n_lang = long.loc[long.da_def == "DA-size", "language"].nunique()
    print(f"Wrote appendix slides → {_SLIDES} "
          f"(2 above-random + {n_lang} per-language DA)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default="custom_swissai_hf",
                   help="Pool name from configs/models.json (default: custom_swissai_hf).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    run(pool=args.pool, out_dir=SNR_DEFINITION / stage / args.pool)


if __name__ == "__main__":
    main()
