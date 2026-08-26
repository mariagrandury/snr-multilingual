#!/usr/bin/env python3
"""Per-language token coverage of every data mixture, as a heatmap.

Columns are the language settings (scheme B only where it differs from A:
L8, L15, L30); rows are the FineWeb-2 languages in corpus-size order, which is
the order the sets are built from, so the nested structure (L2 subset of L8
subset of L15 ...) reads down the diagonal. A cell is the number of tokens that
language contributes to that mixture.

A column is left UNCOLOURED when its .bin does not exist yet — "not built"
is visually distinct from "built but this language contributes nothing".

Per-language token counts come from the best source available, per mixture:
  * <prefix>.checkpoint.json  — the builder's own plan + per-source progress.
    Exact. Present while a build is running or after it was interrupted.
  * <prefix>.manifest.json    — written when a build completes. Exact.
  * otherwise                 — estimated from the FineWeb-2 corpus byte
    shares in fineweb2-language-distribution.csv. The builder allocates
    proportionally to *sampled token* estimates, and bytes-per-token varies by
    script, so these are approximate; the plot marks them.

    python3.11 data_progress.py                  # writes data_progress.png
    python3.11 data_progress.py --out /tmp/x.png --print
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_data_mixtures import fineweb_target_tokens  # noqa: E402

DEFAULT_DATA_DIR = Path(
    "/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data")
BYTES_PER_TOKEN = 4          # Megatron .bin element size for a 131k vocab
SETTINGS = [1, 2, 8, 15, 30, 50, 100]
SCHEME_B_SETTINGS = {8, 15, 30}   # the only settings where B differs from A


def language_sets(scheme: str) -> dict[int, list[str]]:
    sets = json.loads(
        (SCRIPT_DIR / f"language_sets_scheme{scheme}.json").read_text())["sets"]
    return {L: sets[f"FW_L{L}"] for L in SETTINGS if f"FW_L{L}" in sets}


def corpus_bytes() -> dict[str, int]:
    """utf8 bytes per FineWeb-2 language subset (train split only)."""
    out: dict[str, int] = {}
    with open(SCRIPT_DIR / "fineweb2-language-distribution.csv") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train" or row["subset"].endswith("_removed"):
                continue
            try:
                out[row["subset"]] = out.get(row["subset"], 0) + int(row["utf8_bytes"])
            except ValueError:
                pass
    return out


def mixture_paths(data_dir: Path, L: int, scheme: str) -> dict[str, Path]:
    base = (data_dir if scheme == "A" else data_dir / "schemeB") / f"fineweb_L{L}"
    return {"bin": base.with_suffix(".bin"),
            "ckpt": Path(f"{base}.checkpoint.json"),
            "plan": Path(f"{base}.plan.json"),
            "manifest": Path(f"{base}.manifest.json")}


# Builds now log to capstor beside the data; older runs logged to scratch.
BUILD_LOG_DIRS = [
    DEFAULT_DATA_DIR / "logs",
    Path(f"/iopsstor/scratch/cscs/{os.environ.get('USER','')}/data/logs"),
]
PLAN_RE = re.compile(r"^\s*fineweb_(\S+)\s+target=([0-9.]+)B", re.M)


def plan_from_build_log(L: int, scheme: str) -> dict[str, int] | None:
    """Exact per-language token targets, parsed from the build's own stdout.

    create_data_mixture.print_plan() emits one `target=<N>B` line per source,
    computed by tokenizing a sample of each language — the real allocation the
    build then follows. Nothing else on disk records it: the training mixtures
    write no manifest (the only manifest is the validation one) and
    remove_checkpoint() deletes the checkpoint on success. So the log is the
    only durable copy of these numbers, and the numbers the analysis needs.
    """
    logs = sorted((f for d in BUILD_LOG_DIRS if d.is_dir()
                   for f in d.glob(f"build-{scheme.lower()}-L{L}-*.out")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        hits = PLAN_RE.findall(log.read_text(errors="ignore"))
        if hits:
            # print_plan runs once per build attempt; later lines supersede.
            return {lang: int(float(tok) * 1e9) for lang, tok in hits}
    return None


def exact_tokens(paths: dict[str, Path]) -> tuple[dict[str, int], str] | None:
    """Per-language tokens from the builder's own records, if available."""
    # Written by create_data_mixture.write_plan — the durable record, and the
    # one that survives the log being swept.
    if paths["plan"].is_file():
        pl = json.loads(paths["plan"].read_text()).get("sources", {})
        got = {k.removeprefix("fineweb_"): v["target_tokens"]
               for k, v in pl.items() if k.startswith("fineweb_")}
        if got:
            return got, "plan.json"
    if paths["manifest"].is_file():
        m = json.loads(paths["manifest"].read_text())
        src = m.get("sources", m) if isinstance(m, dict) else {}
        got = {k.removeprefix("fineweb_"): v.get("tokens", 0)
               for k, v in src.items() if isinstance(v, dict)}
        if got:
            return got, "manifest"
    if paths["ckpt"].is_file():
        c = json.loads(paths["ckpt"].read_text())
        prog = c.get("source_progress", {})
        got = {k.removeprefix("fineweb_"): v.get("source_toks", 0)
               for k, v in prog.items() if isinstance(v, dict)}
        if got:
            return got, "checkpoint"
    return None


def column(data_dir: Path, L: int, scheme: str, langs: list[str],
           shares: dict[str, int]) -> dict:
    """One mixture: per-language tokens, build state, and where numbers came from."""
    paths = mixture_paths(data_dir, L, scheme)
    built = paths["bin"].is_file()
    size_b = paths["bin"].stat().st_size if built else 0
    target = fineweb_target_tokens(L)
    have = size_b // BYTES_PER_TOKEN

    got = exact_tokens(paths)
    plan = plan_from_build_log(L, scheme)
    if got:
        tokens, source = got
        tokens = {l: tokens.get(l, 0) for l in langs}
    elif plan:
        tokens = {l: plan.get(l, 0) for l in langs}
        source = "build log"
    else:
        # Last resort. Byte share is NOT token share — tokenizer efficiency
        # varies by script, and against the builder's true plan this lands
        # between -50% and +45% per language. Never use for analysis.
        tot = sum(shares.get(l, 0) for l in langs) or 1
        tokens = {l: int(have * shares.get(l, 0) / tot) for l in langs}
        source = "ESTIMATED (rough)"

    # create_data_mixture.remove_checkpoint() runs only after a build finishes,
    # so a surviving checkpoint means the build is still going (or was killed).
    # A finished build can still land under target when the languages simply
    # run out of corpus — L2 draws on rus_Cyrl alone (~72B) against a 92B
    # target. That is "source-limited", not "incomplete", and the plan calls
    # for recording it rather than chasing the target.
    if not langs:
        # L1 is 100% English (english_dclm); there is no FineWeb mixture to
        # build, so it is neither "built" nor "missing".
        state = "English-only"
    elif not built:
        state = "not built"
    elif paths["ckpt"].is_file():
        state = "in progress"
    elif have < 0.98 * target:
        state = "source-limited"
    else:
        state = "complete"
    return {"L": L, "scheme": scheme, "built": built, "tokens": tokens,
            "have": have, "target": target, "source": source, "state": state,
            "frac": (have / target) if target else 0.0}


def collect(data_dir: Path) -> tuple[list[dict], list[str]]:
    shares = corpus_bytes()
    cols, seen = [], []
    for L in SETTINGS:
        for scheme in ("A", "B"):
            if scheme == "B" and L not in SCHEME_B_SETTINGS:
                continue
            langs = language_sets(scheme).get(L, [])
            cols.append(column(data_dir, L, scheme, langs, shares))
            seen += langs
    # rows: every language any mixture uses, biggest corpus first — the order
    # the sets themselves are built from, so the nesting reads as a staircase.
    rows = sorted(set(seen), key=lambda l: -shares.get(l, 0))
    return cols, rows


def label(c: dict) -> str:
    return f"L{c['L']}" + ("B" if c["scheme"] == "B" else "")


def render(cols: list[dict], rows: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    vals = np.full((len(rows), len(cols)), np.nan)
    for j, c in enumerate(cols):
        if not c["built"]:
            continue                      # leave the whole column blank
        for i, lang in enumerate(rows):
            t = c["tokens"].get(lang, 0)
            if t > 0:
                vals[i, j] = t

    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 4, 0.22 * len(rows) + 3))
    finite = vals[np.isfinite(vals)]
    norm = LogNorm(vmin=max(finite.min(), 1), vmax=finite.max()) if finite.size else None
    im = ax.imshow(vals, aspect="auto", cmap="viridis", norm=norm)

    ax.set_xticks(range(len(cols)))
    xlabels = []
    for c in cols:
        pct = "—" if not c["built"] else "{:.0f}%".format(c["frac"] * 100)
        xlabels.append("{}\n{}".format(label(c), pct))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)

    for j, c in enumerate(cols):
        if not c["built"]:
            ax.add_patch(plt.Rectangle((j - .5, -.5), 1, len(rows), facecolor="#eeeeee",
                                       edgecolor="none", zorder=3))
            ax.text(j, len(rows) / 2, c["state"], rotation=90, ha="center",
                    va="center", fontsize=9, color="#888888", zorder=4)
            continue
        for i, lang in enumerate(rows):
            t = c["tokens"].get(lang, 0)
            if t > 0:
                ax.text(j, i, f"{t/1e9:.1f}" if t >= 1e8 else f"{t/1e6:.0f}m",
                        ha="center", va="center", fontsize=4.5,
                        color="white" if t < finite.max() / 8 else "black")

    fig.colorbar(im, ax=ax, label="tokens in mixture (log scale)", fraction=0.02)
    est = ", ".join(label(c) for c in cols if c["built"] and c["source"] == "estimated")
    ax.set_title("Data-mixture coverage — tokens per language per setting\n"
                 f"exact where the builder left a manifest/checkpoint; estimated for: {est or 'none'}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "data_progress.png")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="also print the per-mixture status table")
    args = ap.parse_args()

    cols, rows = collect(args.data_dir)
    print(f"{'mixture':8} {'state':>14} {'tokens':>10} {'target':>8} {'done':>6}  source")
    for c in cols:
        print(f"{label(c):8} {c['state']:>14} {c['have']/1e9:>9.1f}B "
              f"{c['target']/1e9:>7.0f}B {c['frac']*100:>5.1f}%  {c['source']}")
    if args.show:
        for c in cols:
            if not c["built"]:
                continue
            top = sorted(c["tokens"].items(), key=lambda kv: -kv[1])[:5]
            print(f"\n{label(c)} top languages: " +
                  ", ".join(f"{k} {v/1e9:.1f}B" for k, v in top))
    render(cols, rows, args.out)


if __name__ == "__main__":
    main()
