"""Load the predictivity ladder's eval results from ladder_report.csv.

`ladder_report.py --plot --publish --push-hf` (src/pretrain) writes one wide
table — one row per checkpoint (`<cell>-iter<N>`), one column per measurement
(`bench__<task>`, `bpb__<subset>`, `ppl__<subset>`, `macro_bpb`, `loss`,
`run__<metric>`) — and publishes it to the HF dataset named by
configs/hf_wandb.json `repo_id_ladder_report`. That file is the source of
truth for every predictivity analysis: nothing here reads eval_logs or W&B.

This module melts the wide table into the long schema the rest of the
pipeline consumes (`snr.dataloader.get_slice`, `analysis.utils.build_snr_pool`):
one row per (model, step, task) with `primary_score`, plus the ladder's axes
(`size`, `L`, `ladder`, `arch`, `activation`, `optimizer`, `data`, `scheme`,
`T`, `seed`) as columns: `ladder` is the cell name's token (the trained
configuration), `arch`, `activation` and `optimizer` its design levels
(launch_trainings.LADDERS); `data` is the build label the report calls
`scheme` (A, AT3, B, ZH, ES, DCLMP, FWEB: what names cells and data dirs), and
`scheme` (the letter, A/B/C) and `T` are its design levels
(launch_trainings.DATA_SCHEMES `letter`, `temp`). Per-language BPB enters as
tasks of its own — `bpb_<subset>` (`bpb_rus_Cyrl`, `bpb_dclm` for English;
lower is better) and `bpb_macro` — so the same decision-accuracy and SNR
machinery runs on the plan's outcome metric. Decision accuracy is a rank
agreement between two model sets, so a lower-is-better task needs no sign
flip; SNR variants use dispersion over mean, which is sign-free too.

Resolution order for the files: an explicit ``path``, ``$SNR_LADDER_DIR``,
then ``<SNR data dir>/ladder-report`` — downloaded from the Hub on first use.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import sys
from functools import lru_cache, wraps
from pathlib import Path

import pandas as pd

from snr.constants import DATA_DIR

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import load_hf_wandb_config  # noqa: E402
from pretrain.ladder_report import CELL_RE, on_grid  # noqa: E402
from pretrain.launch_trainings import (  # noqa: E402
    BASELINE_LADDER, DATA_SCHEMES, LADDERS, NOISE_GRID, NOISE_WINDOW, SEQ_LEN, cell_gbs, mix_label)

LADDER_FILES = ("ladder_report.csv", "ladder_report_curve.csv", "ladder_report.md")
# Tokens per optimizer step = the RUNG'S OWN global batch x seq. It was a
# constant 504 x 4096 until 2026-09-23, when the 90M and 175M rungs moved to
# batch 84 and 168 (GBS_BY_SIZE; plan/90M-175M-batch-retrain.md). Holding the
# old constant would inflate their tokens — and so their compute, and so every
# point they contribute to a scaling fit — by 6x and 3x. 175M is in
# ANALYSIS_SIZES, so that is not a cosmetic error.
_HYPERPARAMS = _SRC / "pretrain" / "hyperparams"
_META = ["cell", "size", "L", "ladder", *LADDERS[BASELINE_LADDER], "scheme", "seed", "iter"]


def ladder_dir(path: str | Path | None = None) -> Path:
    """Directory holding ladder_report.csv (+ curve + md); pulled from the Hub
    when the resolved directory has no CSV yet."""
    d = Path(path or os.environ.get("SNR_LADDER_DIR", DATA_DIR / "ladder-report"))
    if not (d / "ladder_report.csv").is_file():
        from huggingface_hub import hf_hub_download
        repo = load_hf_wandb_config()["repo_id_ladder_report"]
        for name in LADDER_FILES:
            hf_hub_download(repo, name, repo_type="dataset", local_dir=d)
    return d


def load_ladder_wide(path: str | Path | None = None) -> pd.DataFrame:
    """The wide table as published: one row per checkpoint."""
    return pd.read_csv(ladder_dir(path) / "ladder_report.csv", low_memory=False)


@lru_cache(maxsize=None)
def cell_params(size: str, ladder: str) -> int:
    """Parameters on the FLOPs convention — N_non_emb + d_model x V, the tied
    embedding included — from the reviewed hyperparams file of the ladder."""
    h = json.loads((_HYPERPARAMS / f"hyperparams_{ladder}.json").read_text())
    cfg = h["configs"][size]
    return int(cfg["n_non_emb_params"] + h["global"]["vocab_size"] * cfg["hidden_size"])


# A checkpoint is on the shared grid when it sits within this share of the run
# of a grid point (0.5 %: far below the 1.7 % spacing of the densest, 60-save
# grid, and above the 0.04 % drift of the one cell whose interval does not
# divide its target). The same tolerance decides whether a run's last save
# counts as its final checkpoint.
GRID_TOL = 0.005


def _on_shared_grid(df: pd.DataFrame) -> pd.Series:
    """Rows on the checkpoint grid every size shares.

    Every run saves 20 evenly spaced checkpoints (40 at 1B, 60 at 1.7B) and
    is evaluated on every 2nd one plus the final. Scores are comparable
    across sizes at matched fractions of training, but the late-window
    noise estimate is not unless the window spans the same fraction at every
    size (plan/1b-models.md). BPB is scored on every saved checkpoint, so its
    shared grid is k/20; benchmarks are evaluated on every 2nd, so theirs is
    k/10. The final checkpoint always stays.
    """
    target = pd.to_numeric(df["run__target_iters"], errors="coerce")
    n = df["kind"].map({"benchmark": 10}).fillna(20)
    # nearest grid point within GRID_TOL of the run, not exact divisibility: a
    # cell whose save interval does not divide its target (lm-1B-L8-deep-seed1904
    # saves every 1,143 of 45,740 iterations) is on the grid to within 0.04 %
    pos = df["iter"] / target
    on_grid = (pos - (pos * n).round() / n).abs() <= GRID_TOL
    # The noise window (RULES.md rule 4) is read on the k/20 points at every
    # size, benchmarks included, so 85 % and 95 % are on the grid as well —
    # without this the evals `due_iters` now asks for would be loaded and
    # then dropped here. Outside the window benchmarks stay on the tenths.
    in_window = pos >= 1 - NOISE_WINDOW - GRID_TOL
    on_window_grid = ((pos - (pos * NOISE_GRID).round() / NOISE_GRID).abs() <= GRID_TOL) & in_window
    return on_grid | on_window_grid | ((1 - pos).abs() <= GRID_TOL) | target.isna()


# The melted frame is the same for every step of one refresh and costs ~40 s
# to build, so it is kept on disk (git-ignored), keyed by the bytes of every
# input it is computed from: the report, this loader and the pretrain modules
# and configs it reads. One file per argument set, rewritten when an input
# changes; SNR_CACHE=0 bypasses it, SNR_CACHE_DIR moves it.
CACHE_DIR = Path(os.environ.get("SNR_CACHE_DIR", Path(__file__).resolve().parents[2] / ".cache"))


def _disk_cached(fn):
    sig = inspect.signature(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get("SNR_CACHE", "1") == "0":
            return fn(*args, **kwargs)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        opts = repr(sorted((k, v) for k, v in bound.arguments.items() if k != "path")).encode()
        h = hashlib.sha256(opts + pd.__version__.encode())
        for f in [ladder_dir(bound.arguments["path"]) / "ladder_report.csv", Path(__file__),
                  *(_SRC / "pretrain" / m for m in ("ladder_report.py", "launch_trainings.py", "pretrain_progress.py")),
                  *sorted(_HYPERPARAMS.glob("*.json")), *sorted((_SRC.parent / "configs").glob("*.json"))]:
            h.update(f.name.encode() + f.read_bytes())
        key = h.hexdigest()
        cache = CACHE_DIR / f"{fn.__name__}-{hashlib.sha256(opts).hexdigest()[:12]}.pkl"
        try:                            # key first: a stale file is never unpickled past it
            with open(cache, "rb") as fh:
                if pickle.load(fh) == key:
                    return pickle.load(fh)
        except Exception:               # unreadable (e.g. pandas upgraded): rebuild and overwrite
            pass
        df = fn(*args, **kwargs)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(key, fh, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)          # atomic: a concurrent reader sees the old file or the new one
        return df
    return wrapper


@_disk_cached
def load_predictivity_eval_results(
    path: str | Path | None = None,
    include_diverged: bool = False,
    include_incomplete: bool = False,
    shared_grid: bool = True,
    require_final: bool = True,
) -> pd.DataFrame:
    """One row per (cell, checkpoint, task) for the predictivity ladder.

    Columns: model (the cell name), family (cross-size identity), size, L,
    ladder (the name's token: deep / shallow / swiglu), its levels arch (the
    depth: a swiglu cell is "deep"), activation and optimizer, data (the
    build label: A, AT3, B, ZH, ES, DCLMP, FWEB), its levels scheme (the
    letter: A, B, C) and T (1, 3), seed, mix (`L8-schemeB-deep`: the cell's design variant,
    what the 36-sweep called its data mixture), step, task, kind
    (`benchmark` / `bpb` / `loss`), primary_score, tokens, compute
    (6 x params x tokens on the ladder convention), diverged, complete.

    Only the runs trained at the batch their rung uses NOW enter (the 90M and
    175M rungs were retrained at batch 84 / 168 on 2026-09-23; the diverged
    batch-504 runs they replace are still on disk and in older reports, and
    both versions of a cell would carry one `family`): `ladder_report.on_grid`,
    the report's own test, repeated here because this is where every analysis
    reads from. Diverged runs and runs that have not reached their target are
    dropped by default: their final checkpoint is not the annealed endpoint
    the ladder compares. For the same
    reason `require_final` drops every (cell, task) series whose last scored
    checkpoint is not the run's final save: the evaluation of that cell is
    still in flight, and "the last checkpoint per task" that every analysis
    reads would otherwise be a mid-run checkpoint standing in for the
    reference (2026-09-16: seven 1.7B cells and one 1B cell).
    """
    wide = load_ladder_wide(path)
    wide = wide.dropna(subset=["cell"])
    # A report from before 2026-10-05 has no `ladder` column, and its `arch`
    # holds the name's token (swiglu included): read the ladder from it and the
    # levels from the registry, which is what the report now writes itself.
    # (One concat, not four inserts: the table is ~4,000 columns wide.)
    if "ladder" not in wide.columns:
        levels = pd.DataFrame([LADDERS[x] for x in wide["arch"]], index=wide.index)
        wide = pd.concat([wide.drop(columns="arch"), levels.assign(ladder=wide["arch"])], axis=1)
    matched = wide["cell"].map(CELL_RE.match)
    wide = wide[[bool(m) and on_grid(m) for m in matched]]        # the rung's current batch only
    # A report predating one of these columns is treated as complete and
    # healthy: filling NaN instead would silently drop every row below.
    for c, absent in (("run__diverged", 0), ("run__complete", 1),
                      ("run__target_iters", float("nan"))):
        if c not in wide.columns:
            wide[c] = absent
    families = {
        "benchmark": [c for c in wide.columns if c.startswith("bench__")],
        "bpb": [c for c in wide.columns if c.startswith("bpb__")]
               + (["macro_bpb"] if "macro_bpb" in wide.columns else []),
        "loss": ["loss"] if "loss" in wide.columns else [],
    }
    keep = _META + ["run__diverged", "run__complete", "run__target_iters"]
    parts = []
    for kind, cols in families.items():
        if not cols:
            continue
        m = wide.melt(id_vars=keep, value_vars=cols, var_name="task",
                      value_name="primary_score").dropna(subset=["primary_score"])
        m["task"] = (m["task"].str.replace(r"^bench__", "", regex=True)
                     .str.replace(r"^bpb__", "bpb_", regex=True)
                     .replace({"macro_bpb": "bpb_macro", "loss": "train_loss"}))
        m["kind"] = kind
        parts.append(m)
    df = pd.concat(parts, ignore_index=True)
    df["primary_score"] = pd.to_numeric(df["primary_score"], errors="coerce")
    df = df.dropna(subset=["primary_score"])

    df["diverged"] = pd.to_numeric(df.pop("run__diverged"), errors="coerce").fillna(0).astype(int)
    df["complete"] = pd.to_numeric(df.pop("run__complete"), errors="coerce").fillna(0).astype(int)
    if not include_diverged:
        df = df[df["diverged"] == 0]
    if not include_incomplete:
        df = df[df["complete"] == 1]
    if require_final:
        target = pd.to_numeric(df["run__target_iters"], errors="coerce")
        last = df.groupby(["cell", "task"])["iter"].transform("max")
        df = df[(last >= target * (1 - GRID_TOL)) | target.isna()]   # the last save may sit a few iterations short of the target
    if shared_grid:
        df = df[_on_shared_grid(df)]
    df = df.drop(columns=["run__target_iters"])

    df["L"] = df["L"].astype(int)
    df["seed"] = df["seed"].astype(int)
    # The report's `scheme` is the launcher's word, the BUILD label; in the
    # analysis frame it is `data`, and `scheme` is the letter of the recipe the
    # build implements at its L, `T` its temperature (analysis/RULES.md,
    # Definitions) — the same split as `ladder` vs arch/activation/optimizer.
    df = df.rename(columns={"scheme": "data"})
    df["scheme"] = df["data"].map({d: v["letter"] for d, v in DATA_SCHEMES.items()})
    df["T"] = df["data"].map({d: int(v["temp"]) for d, v in DATA_SCHEMES.items()})
    df["mix"] = [mix_label(L, ladder, data)
                 for L, ladder, data in zip(df["L"], df["ladder"], df["data"])]
    df["family"] = "lm-" + df["mix"] + "-seed" + df["seed"].astype(str)
    df = df.rename(columns={"cell": "model", "iter": "step"})
    df["step"] = df["step"].astype(int)
    df["tokens"] = df["step"] * [float(cell_gbs(sz) * SEQ_LEN) for sz in df["size"]]
    params = {k: cell_params(*k) for k in set(zip(df["size"], df["ladder"]))}
    df["compute"] = 6.0 * df["tokens"] * [params[k] for k in zip(df["size"], df["ladder"])]
    return (df.sort_values(["size", "L", "ladder", "data", "seed", "step", "task"])
              .reset_index(drop=True))
