"""Shared helpers for the auto-generated README / slides blocks.

Each analysis script owns a ``generate_readme(pool)`` / ``generate_slides(pool)``
that recomputes its headline numbers and rewrites a *marker-delimited* block in
the corresponding doc. Hand-written sections (Research Question, Experimental
setup, TODO; the hand-tuned results slide) live OUTSIDE the markers and are
never touched.

Each block is keyed so a file can carry several:

    <!-- BEGIN auto:KEY (generator) -->
    ... regenerated content ...
    <!-- END auto:KEY -->

``replace_block`` is idempotent: it swaps an existing block in place (matching on
the KEY, regardless of the generator note), or appends one if the markers aren't
there yet. The generators run per-pool inside the pipeline, so each gates itself
on its canonical pool (see ``CANONICAL_POOL``) — only that pool writes docs.
"""

from __future__ import annotations

import re
from pathlib import Path

# Canonical pool whose numbers each README/slide reflects: the predictivity
# ladder as planned (one run per size x L x arch x scheme, seed 1904). The
# seed-replicate pools feed the noise estimates and the seed holdout; the
# 36-sweep pools (`custom_swissai_hf`, `seeds_*`) keep their committed results
# as history. A generator no-ops on other pools.
CANONICAL_POOL = "predictivity"
ALLENAI_POOL = "predictivity"
# The seed holdout the rq02 README reports (compare_seed_splits.py).
HOLDOUT = ("predictivity_seeds_train", "predictivity_seeds_test")

# The Slidev deck (repo-root/documents/slides.md).
SLIDES = Path(__file__).resolve().parents[3] / "documents" / "slides.md"


def replace_block(path: Path, key: str, body: str, generator: str = "") -> Path:
    """Idempotently replace the ``auto:KEY`` block in ``path`` with ``body``.

    Matches an existing ``<!-- BEGIN auto:KEY ... --> … <!-- END auto:KEY -->``
    block (the generator note inside BEGIN may differ) and swaps it; if absent,
    appends a fresh block at the end of the file.
    """
    path = Path(path)
    note = f" ({generator})" if generator else ""
    begin = f"<!-- BEGIN auto:{key}{note} -->"
    end = f"<!-- END auto:{key} -->"
    block = f"{begin}\n{body.rstrip()}\n{end}"
    pat = re.compile(rf"<!-- BEGIN auto:{re.escape(key)}.*?-->.*?<!-- END auto:{re.escape(key)} -->",
                     flags=re.DOTALL)
    text = path.read_text()
    if pat.search(text):
        text = pat.sub(lambda _m: block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)
    return path


def fmt(v, p: int = 2) -> str:
    """Fixed-precision float, blank for NaN/None (table-friendly)."""
    try:
        if v is None or (isinstance(v, float) and v != v):
            return ""
    except TypeError:
        return str(v)
    return f"{float(v):.{p}f}"


def md_table(header: list[str], rows: list[list]) -> str:
    """GitHub-flavoured markdown table. Right-aligns numeric-looking columns."""
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)
