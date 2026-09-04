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
(`size`, `L`, `arch`, `scheme`, `seed`) as columns. Per-language BPB enters as
tasks of its own — `bpb_<subset>` (`bpb_rus_Cyrl`, `bpb_dclm` for English;
lower is better) and `bpb_macro` — so the same decision-accuracy and SNR
machinery runs on the plan's outcome metric. Decision accuracy is a rank
agreement between two model sets, so a lower-is-better task needs no sign
flip; SNR variants use dispersion over mean, which is sign-free too.

Resolution order for the files: an explicit ``path``, ``$SNR_LADDER_DIR``,
then ``<SNR data dir>/ladder-report`` — downloaded from the Hub on first use.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

from snr.constants import DATA_DIR

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import load_hf_wandb_config  # noqa: E402
from pretrain.launch_trainings import mix_label  # noqa: E402

LADDER_FILES = ("ladder_report.csv", "ladder_report_curve.csv", "ladder_report.md")
TOKENS_PER_ITER = 504 * 4096            # GBS x seq, fixed across the sweep
_HYPERPARAMS = _SRC / "pretrain" / "hyperparams"
_META = ["cell", "size", "L", "arch", "scheme", "seed", "iter"]


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
def cell_params(size: str, arch: str) -> int:
    """Parameters on the FLOPs convention — N_non_emb + d_model x V, the tied
    embedding included — from the reviewed hyperparams file of the arch."""
    h = json.loads((_HYPERPARAMS / f"hyperparams_{arch}.json").read_text())
    cfg = h["configs"][size]
    return int(cfg["n_non_emb_params"] + h["global"]["vocab_size"] * cfg["hidden_size"])


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
    on_grid = (df["iter"] * n) % target == 0
    return on_grid | (df["iter"] == target) | target.isna()


def load_predictivity_eval_results(
    path: str | Path | None = None,
    include_diverged: bool = False,
    include_incomplete: bool = False,
    shared_grid: bool = True,
) -> pd.DataFrame:
    """One row per (cell, checkpoint, task) for the predictivity ladder.

    Columns: model (the cell name), family (cross-size identity), size, L,
    arch, scheme, seed, mix (`L8-schemeB-deep`: the cell's design variant,
    what the 36-sweep called its data mixture), step, task, kind
    (`benchmark` / `bpb` / `loss`), primary_score, tokens, compute
    (6 x params x tokens on the ladder convention), diverged, complete.

    Diverged runs (the 90M rung, see plan/90M-rung-anomaly.md) and runs that
    have not reached their target are dropped by default: their final
    checkpoint is not the annealed endpoint the ladder compares.
    """
    wide = load_ladder_wide(path)
    wide = wide.dropna(subset=["cell"])
    for c in ("run__diverged", "run__complete", "run__target_iters"):
        if c not in wide.columns:
            wide[c] = float("nan")
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
    if shared_grid:
        df = df[_on_shared_grid(df)]
    df = df.drop(columns=["run__target_iters"])

    df["L"] = df["L"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["mix"] = [mix_label(L, arch, scheme)
                 for L, arch, scheme in zip(df["L"], df["arch"], df["scheme"])]
    df["family"] = "lm-" + df["mix"] + "-seed" + df["seed"].astype(str)
    df = df.rename(columns={"cell": "model", "iter": "step"})
    df["step"] = df["step"].astype(int)
    df["tokens"] = df["step"] * float(TOKENS_PER_ITER)
    params = {k: cell_params(*k) for k in set(zip(df["size"], df["arch"]))}
    df["compute"] = 6.0 * df["tokens"] * [params[k] for k in zip(df["size"], df["arch"])]
    return (df.sort_values(["size", "L", "arch", "scheme", "seed", "step", "task"])
              .reset_index(drop=True))
