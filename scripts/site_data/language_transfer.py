"""Data of the language-transfer page's additional views (rq06_language_transfer).

- `groups`: the list decision (scheme A vs B, L8–L30) by language group, DA-size
  per proxy size and DA-ckpt along the 1.7B run (the key figure's table,
  `transfer_da_all_lines_mono_axis_paper.csv`).
- `cross`: the cross-language maps of `cross_task_transfer.py`, all-languages
  population, as {source language: {target language: DA-size}} for HellaSwag
  and for the per-language pools of the tasks above chance and of the rq2 tasks.
- `family`: `family_transfer.py`'s share of untrained-language tasks above
  chance by family group, both family levels, against English only, and the
  lift per subfamily on at least MIN_VERDICTS verdicts (the paper's three
  cells: L2 A, L8 A, L8 B).
Rows only filtered and renamed."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
from analysis import paths  # noqa: E402
from analysis.rq06_language_transfer.family_transfer import MIN_VERDICTS  # noqa: E402

ALL = "pretraining/predictivity_seeds"
PRED = "pretraining/predictivity"
CELLS = ["L2 A", "L8 A", "L8 B"]      # the cells with untrained-language results (the paper's figures)


def _r(x):
    return None if pd.isna(x) else round(float(x), 4)


def _matrix(df: pd.DataFrame, src: str, tgt: str) -> dict:
    out = {}
    for s, t, v in zip(df[src], df[tgt], df["da_size"]):
        if pd.notna(v):
            out.setdefault(s, {})[t] = round(float(v), 3)
    return out


def build() -> dict:
    d6 = paths.LANGUAGE_TRANSFER
    lines = pd.read_csv(d6 / ALL / "transfer_da_all_lines_mono_axis_paper.csv")
    groups = [dict(group=row.split(" / ")[0], kind="size" if p.startswith("DA-size") else "ckpt",
                   x=c, da=_r(v)) for p, row, c, v in zip(lines["panel"], lines["row"], lines["col"], lines["value"])]

    pre = d6 / PRED
    hs = pd.read_csv(pre / "cross_task_da_size_by_language_hellaswag_all_languages_mono_axis.csv")
    cross = {
        "hellaswag": _matrix(hs, "proxy_language", "target_language"),
        "above_chance": _matrix(pd.read_csv(pre / "cross_task_da_size_by_language_above_chance_all_languages_mono_axis.csv"), "proxy", "target"),
        "above_66_either": _matrix(pd.read_csv(pre / "cross_task_da_size_by_language_above_66_either_all_languages_mono_axis.csv"), "proxy", "target"),
    }

    fam = pd.read_csv(pre / "above_chance_untrained_by_family.csv")
    fam = fam[fam["cell"].isin(CELLS)]
    share = [dict(cell=x.cell, level=x.level, group=x.group, size=x.size, share=_r(x.share_above),
                  base=_r(x.baseline_share_above), n=int(x.n_tasks), langs=int(x.n_languages))
             for x in fam.itertuples()]
    lift = pd.read_csv(pre / "above_chance_untrained_lift_by_subfamily_paper.csv")
    lift = lift[lift["n_verdicts"] >= MIN_VERDICTS]          # what the paper figure draws
    lift = [dict(cell=x.cell, sub=x.family_sub, trained=bool(x.same_sub), n=int(x.n_verdicts),
                 langs=x.languages, share=_r(x.share_above), base=_r(x.baseline_share_above),
                 sizes=x.sizes, lift=_r(x.lift)) for x in lift.itertuples()]
    return {"language_transfer_views": dict(groups=groups, cross=cross, family=dict(share=share, lift=lift))}
