#!/usr/bin/env python3
"""
Launch predictivity-sweep training jobs on CSCS (sbatch) or Azure ML (az ml).

The grid (see plan/small-to-large-predictivity-training-plan.md):

  * size — the 6-rung ladder (90M..1.7B) shared by the reviewed hyperparams
           files; --arch picks deep (hyperparams/hyperparams_deep.json, the
           baseline) or shallow (hyperparams/hyperparams_shallow.json, the
           model-depth intervention level).
  * L    — language setting in {1, 2, 8, 15, 30, 50, 100}: English + L-1
           FineWeb-2 languages. Every size trains at every setting.
  * scheme — the data build (DATA_SCHEMES, selected with --scheme): A is the
           resource-ranked T=1 baseline; AT3 is the same lists at T=3 and
           supplies both the L50 temperature intervention and L100, which
           exists ONLY at T=3; B is diversity-first at L in {8, 15, 30}; ZH
           and ES swap L2's Russian for Chinese / Spanish.
  * seed — 1904 by default; three seeds on the columns the plan marks x3, and
           the triple differs by size — (64, 313, 1904) at 175M and 600M for
           L in {1, 2, 50}, (28, 1797, 1904) at 1B for L in {1, 2, 30, 50}
           (see SEED_TRIPLES).

Each run trains its size's own budget D(N) = 5 x Chinchilla = 100 x N on the
fixed 50/50 English (DCLM) + FineWeb-2 mix (L=1 is 100% English), blended at
training time from the pre-built datasets (data/build_data_mixtures.py).

Both platforms run the SAME training logic: this script builds one env-var
dict per cell and hands it to megatron_args.sh through a thin platform
wrapper — launch_pretraining_cscs.sh (sbatch --export) or
launch_pretraining_azure.sh (az ml job create --set, via
azure/jobs/pretrain.yml). Azure placement is by size: <=600M on the Spain
low-priority pool, 1B/1.7B on the UK Spot pool.

IDEMPOTENT — re-running is always safe, there is no separate resume script:
  * a cell whose target checkpoint is already on disk is skipped ("done");
  * a cell with a queued/running job is skipped;
  * a partially-trained cell is resubmitted as a RESUME (--save/--load point
    at the same dir): on CSCS the remaining iters size the walltime and a
    stale latest_checkpointed_iteration.txt marker is rewound to the last
    valid checkpoint first; on Azure resubmitting simply continues.
  * a corrupt cell (iter dirs on disk but none loadable) is skipped with a
    warning — cleanup is always a human decision.
(Azure done-detection would need a blob listing per cell, so only the
active-job check runs there; resubmitting a finished cell is a no-op run —
Megatron loads the final checkpoint and exits at --train-iters.)

Usage:
    python launch_trainings.py cscs  [--data_dir DIR] [--time HH:MM:SS]
                                     [--account ACCT] [--dependency DEP]
                                     [--training-steps N] [--test] [filters]
    python launch_trainings.py azure [filters]        # `source azure/env.sh` first

Filters (both platforms): --arch {deep,shallow}, --scheme {A,AT3,B,ZH,ES},
--size 350M[,175M,...], --langs L, --seed N, --dry-run.

Examples:
    # the scheme-A ladder
    python3.11 pretrain/launch_trainings.py cscs --dry-run
    # three sizes, all L                
    python3.11 pretrain/launch_trainings.py cscs --size 175M,350M,600M    
    python3.11 pretrain/launch_trainings.py azure --size 1.7B --langs 30
    # monolingual anchors on Azure
    python3.11 pretrain/launch_trainings.py azure --langs 1        
    # depth intervention       
    python3.11 pretrain/launch_trainings.py cscs --arch shallow --dry-run
    # diversity-first lists 
    python3.11 pretrain/launch_trainings.py cscs --scheme B --langs 8 
    # T=3: L50 and L100    
    python3.11 pretrain/launch_trainings.py cscs --scheme AT3       
    # L2 with Chinese      
    python3.11 pretrain/launch_trainings.py cscs --scheme ZH              
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# .resolve() before .parent: imported through a relative sys.path entry
# (`sys.path.insert(0, '../pretrain')` from src/evals, the documented way to
# build a TASKS list) __file__ keeps the '..', so SCRIPT_DIR.parent.parent
# below lands on src/evals instead of the repo root and every config read
# fails with a path that looks nothing like the mistake.
SCRIPT_DIR = Path(__file__).resolve().parent

# The two reviewed architecture families cover the same six non-embedding
# sizes; "shallow vs deep" (width/depth 128 vs 64) is the model-depth level of
# the intervention axis. Deep is the baseline.
HYPERPARAMS = {
    "deep": SCRIPT_DIR / "hyperparams" / "hyperparams_deep.json",
    "shallow": SCRIPT_DIR / "hyperparams" / "hyperparams_shallow.json",
}

# W&B project for this sweep — single source of truth is configs/hf_wandb.json.
# The entity is the constant "mariagrandury-epflnlp", hardcoded in
# megatron_args.sh (the shared training-argument file).
PROJECT_NAME = json.loads(
    (SCRIPT_DIR.parent.parent / "configs" / "hf_wandb.json").read_text()
)["wandb"]["project"]

# Tokenizer that produced the .bin token IDs (data/build_data_mixtures.py
# default). Must match at train time or the vocab/EOD ids won't line up.
TOKENIZER_MODEL = "swiss-ai/Apertus-70B-2509"

# --- Platform endpoints ------------------------------------------------------

CSCS_SUBMIT_SCRIPT = SCRIPT_DIR / "launch_pretraining_cscs.sh"
# Where training READS english_dclm.* and fineweb_L*.* — override with
# --data_dir. This MUST be an iopsstor path, not the capstor master copy:
# Megatron memmaps the .bin files and reads shuffled (= random access), which
# is capstor's worst case. Measured on identical copies of the same file,
# 112 KB random reads: capstor median 13.5 ms / p99 166 ms / max 433 ms vs
# iopsstor 0.5 ms / 5 ms / 13 ms — ~28x on the median and up to 200x on the
# tail, single-process, before the contention of 12-84 ranks x 4 workers.
# Training off capstor made iterations swing 838 -> 6059 ms at random (see
# CLAUDE.md "The capstor dataloader stall"). capstor stays the durable master
# (iopsstor is purged ~30 days); data/launch_builds.sh writes there and the
# copy is staged here for training.
CSCS_DEFAULT_DATA_DIR = "/iopsstor/scratch/cscs/mariagrandury/data"

# The auto-eval watcher's stdout/stderr. Under the cluster log tree with every
# other generated log, NOT next to the source: it is an append-only file the
# launcher opens on every run, and in the repo it just shows up as untracked
# noise (or, worse, gets committed).
AUTO_EVAL_LOGS = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/auto_evals")

AZURE_JOB_YML = SCRIPT_DIR / "azure" / "jobs" / "pretrain.yml"
DATASTORE = "azureml://datastores/workspaceblobstore/paths/predictivity"
ND_SIZES = {"1B", "1.7B"}  # the 8xH100 pool; everything else runs on the
                          # Spain economy pool. Moved UK South -> Canada
                          # Central 2026-08-26 (meters + the already-granted
                          # low-priority allowance; see azure/env.sh).

# --- Grid definition (edit these to change the sweep) ------------------------

LANG_SETTINGS = [1, 2, 8, 15, 30, 50, 100]
EN_SHARE = 50  # fixed English share for the multilingual (L >= 2) settings

# The ladder, small -> large. The order is load-bearing: a scheme's
# per-setting size cap below means "this rung and every rung under it".
LADDER = ["90M", "175M", "350M", "600M", "1B", "1.7B"]

# Which language settings each size trains at. Every size now covers every
# setting — the 1.7B row gained L=15 and L=50 on 2026-09-10, so the top rung
# exists at every language count.
SIZE_LANG_SETTINGS = {size: LANG_SETTINGS for size in LADDER}

# --- Data schemes -----------------------------------------------------------
#
# A data scheme is one build of the FineWeb-2 half: a language list plus a
# per-language sampling temperature. Scheme A (resource-ranked, T=1) is the
# baseline and carries no name label; every other scheme gets its own build
# directory and its own label, so two schemes can never collide in a
# checkpoint dir, a W&B run id or a models.json entry.
#
#   label     appended to the cell name (empty for the baseline)
#   subdir    directory under --data_dir holding this scheme's fineweb_L*
#   langs     the L settings where the scheme defines data at all
#   max_size  per-setting cap on the ladder; absent = the full ladder
#   temp      sampling temperature for the per-language allocation (T = 1/alpha)
#   sets      which language_sets_scheme<X>.json supplies its language lists
#   seeds     "grid" follows SEED_TRIPLES, "single" is seed 1904 only
#   arches    architecture families the scheme is trained in
#
# L=100 exists ONLY as AT3, deliberately. At T=1 the 99-language allocation is
# so skewed that the median language gets 90M tokens and the smallest 3.5M
# (measured on the filtered subset the builds read — plan, "Re-measured") —
# most per-language BPB numbers would be measuring a model that never saw the
# language. Flattening lifts the median to 373M and the floor to 14.5M, but
# the tail is data-limited: T=2 and T=3 reach the same floor, and at T=3 the
# L100 build realizes 75.4B, short of the 83.6B a 1.7B draws, where T=2 still
# realizes 85.9B. The plan therefore recommends T=2; `temp` here is still 3
# (open decision), and the label and subdir spell the temperature, so a
# switch renames the scheme — free only while nothing is built or trained as
# AT3. L=50 is then built BOTH ways: that pair calibrates the temperature
# change against the T=1 curve running L2..L50, without which L100 could not
# be compared with any other setting.
DATA_SCHEMES = {
    "A": dict(label="", subdir="", langs={1, 2, 8, 15, 30, 50},
              max_size={}, temp=1.0, sets="A", seeds="grid",
              arches=("deep", "shallow")),
    # AT3 runs the whole ladder at both settings: on the filtered subset a 92B
    # L50 build at T=3 realizes 87.1B (13 of 49 languages exhausted), enough
    # for the 83.6B a 1.7B draws (0.96 epochs) — decided 2026-09-10.
    "AT3": dict(label="-AT3", subdir="AT3", langs={50, 100},
                max_size={}, temp=3.0, sets="A", seeds="single",
                arches=("deep", "shallow")),
    "B": dict(label="-schemeB", subdir="schemeB", langs={8, 15, 30},
              max_size={}, temp=1.0, sets="B", seeds="grid",
              arches=("deep", "shallow")),
    # ZH and ES are the second-language intervention at L=2. Their ceiling is
    # the SOURCE, not the budget: the swiss-ai filtered subset holds 71.8B
    # tokens of Russian (scheme A's L2), 59.9B of Chinese and 23.4B of Spanish,
    # against the 83.6B a 1.7B draws and the 47.2B a 1B draws. So no L2 build
    # can feed a 1.7B at all — scheme A's own L2 build is 72.8B, not 92B, for
    # this reason. Chinese is clean through the 1B rung (0.79 epochs).
    # Spanish is clean only through 350M and repeats 2.0x at 1B; capped at 1B
    # anyway (decided 2026-09-10) — record that repetition wherever L2-ES is
    # compared against the other L2 schemes.
    "ZH": dict(label="-ZH", subdir="ZH", langs={2},
               max_size={2: "1B"}, temp=1.0, sets="ZH", seeds="single",
               arches=("deep",)),
    "ES": dict(label="-ES", subdir="ES", langs={2},
               max_size={2: "1B"}, temp=1.0, sets="ES", seeds="single",
               arches=("deep",)),
}

# Seeds. Every cell runs 1904; the columns below run three. The triple is
# per SIZE because two people are filling it: 175M and 600M are this project's
# (64, 313, 1904), and the 1B column is aromanou's, already on disk with
# (28, 1797, 1904) at L in {1, 2, 30}. The grid has to name the seeds that
# EXIST — under the wrong triple the launcher submits two more runs per cell
# and the watcher never evaluates the ones already trained. L50 was added to
# the 1B triple on 2026-09-10 to match the other x3 columns; those two cells
# are new and train under the 40-checkpoint regime, unlike aromanou's runs
# (20 saves, every 2287 iters to 45740 — see pretrain/CLAUDE.md).
SEED_SINGLE = [1904]
SEED_TRIPLES = {
    "175M": ([64, 313, 1904], {1, 2, 50}),
    "600M": ([64, 313, 1904], {1, 2, 50}),
    "1B": ([28, 1797, 1904], {1, 2, 30, 50}),
}


def scheme_sizes(scheme: str, L: int) -> list[str]:
    """Ladder rungs a scheme trains at one setting — everything up to its
    per-setting cap. ZH/ES stop at 1B: no L2 source can feed a 1.7B (see
    DATA_SCHEMES), so their reference is the 1B rung, which still leaves a
    five-rung ladder under it."""
    sizes = [s for s in LADDER if L in SIZE_LANG_SETTINGS[s]]
    cap = DATA_SCHEMES[scheme]["max_size"].get(L)
    return sizes[: sizes.index(cap) + 1] if cap else sizes


def cell_languages(L: int, scheme: str = "A") -> set[str]:
    """Canonical language codes a cell trains on: English plus its setting's
    FineWeb-2 languages, mapped through the `fineweb_iso2` table in
    configs/languages.json (covers the full 100-language set: iso639-1 where
    one exists, the FineWeb iso3 otherwise, Arabic dialects folded into ar).
    tasks.json tags its tasks with the same codes, so the auto-eval watchers
    intersect the two to pick each cell's benchmark tasks."""
    langs = {"en"}
    if L == 1:
        return langs
    iso3_to_code = json.loads(
        (SCRIPT_DIR.parent.parent / "configs" / "languages.json").read_text()
    )["fineweb_iso2"]
    sets_ = json.loads((SCRIPT_DIR / "data" /
                        f"language_sets_scheme{DATA_SCHEMES[scheme]['sets']}"
                        ".json").read_text())["sets"]
    for code in sets_[f"FW_L{L}"]:
        mapped = iso3_to_code.get(code.split("_")[0])
        if mapped:
            langs.add(mapped)
    return langs

# CSCS smoke test: smallest size, one mid setting, 50 steps.
TEST_SIZE = "90M"
TEST_LANGS = 8
TEST_SEED = 1904
TEST_STEPS = 50
TEST_WARMUP = 10
TEST_DECAY = 20


def seeds_for(size: str, L: int, scheme: str = "A") -> list[int]:
    """Seeds for one cell: three on the columns the plan marks x3, one
    everywhere else. The extra data schemes run a single seed — they are
    intervention levels, not part of the seed-noise estimate."""
    if DATA_SCHEMES[scheme]["seeds"] == "single":
        return SEED_SINGLE
    triple, langs = SEED_TRIPLES.get(size, (SEED_SINGLE, set()))
    return triple if L in langs else SEED_SINGLE


def predictivity_cells(schemes: Optional[list] = None) -> list[dict]:
    """Every run in the grid as {size, L, seed, scheme}, in
    scheme -> size -> L -> seed order. Defaults to EVERY scheme so the
    progress, models.json and auto-eval tools see the whole sweep; the
    launcher narrows it with --scheme."""
    cells = []
    for scheme in (schemes or list(DATA_SCHEMES)):
        for size in LADDER:
            for L in sorted(DATA_SCHEMES[scheme]["langs"]):
                if size not in scheme_sizes(scheme, L):
                    continue
                for seed in seeds_for(size, L, scheme):
                    cells.append({"size": size, "L": L, "seed": seed,
                                  "scheme": scheme})
    return cells


def schedule_for(cfg: dict) -> tuple[int, int, int]:
    """Per-size predictivity schedule from the config's own "predictivity"
    block (D = 100 x N tokens at 504 x 4096 tokens/iter, ~4% warmup, ~20% WSD
    decay — see "predictivity_schedule" in the file's global section). The
    top-level train_iters belongs to the fixed-token SNR sweeps and is
    ignored here."""
    p = cfg["predictivity"]
    return p["train_iters"], p["lr_warmup_iters"], p["lr_wsd_decay_iters"]


def n_checkpoints(train_iters: int) -> int:
    """Evenly spaced checkpoints per run: 20, but 40 at the 1B rung and 60 at
    the 1.7B rung — the reference models get denser sampling (every 2.5 % /
    1.7 % of training) while 40 and 60, being multiples of 20, keep every
    size on the shared k/20 grid (1B: every 2nd checkpoint, 1.7B: every 3rd).
    The thresholds sit in the wide gaps between the ~29k, ~46k and ~81k
    schedules. The generators round each schedule to this grid."""
    return 20 if train_iters < 30000 else 40 if train_iters < 60000 else 60


def save_interval(train_iters: int) -> int:
    """Per-size checkpoint interval = train_iters / n_checkpoints — exact
    division by construction, so the final checkpoint is on-grid, checkpoint
    k sits at k/n of training at every size, and the 1xC operating point
    (train_iters/5) is always checkpoint n/5 (4, 8 or 12) — the plan's
    "checkpointing at defined token counts" requirement, index-aligned across
    sizes for the SNR analysis."""
    return train_iters // n_checkpoints(train_iters)


def run_interval(saved: list[int]) -> int:
    """The save interval a run ACTUALLY used, read off its sorted saved iters:
    the most common gap between consecutive saves (the first save alone when
    there is only one; ties go to the larger gap, since off-grid saves from a
    SIGUSR2 exit only ever shorten a gap). Runs trained before the per-size
    40/60 rule — aromanou's 1B cells, 20 saves every 2287 iters — keep their
    own grid, and this is how every consumer learns it."""
    gaps = [b - a for a, b in zip(saved, saved[1:])]
    return max(set(gaps), key=lambda g: (gaps.count(g), g)) if gaps else saved[0]


def due_iters(saved: list[int], target: int, every: int = 2) -> list[int]:
    """The saved iters to evaluate: every `every`-th checkpoint of the SIZE's
    grid, expressed on the run's OWN grid so the evaluated fractions of
    training are the same whatever density the run saved at, plus its final
    save. A 40-save 1B run yields every 2nd save and a 20-save 1B run every
    save — the same k/20 points, comparable checkpoint for checkpoint. Saves
    off the run's grid (a SIGUSR2 exit) are never due; the final save is,
    whatever its iter (aromanou's runs end at 45740, the grid at 45720)."""
    if not saved:
        return []
    step = run_interval(saved)
    n_run = max(1, round(target / step))    # saves the run makes over the budget
    n_size = n_checkpoints(target)          # saves the size's grid makes
    # Save j of the run sits at fraction j/n_run; it is due when that is a
    # multiple of every/n_size, i.e. when j*n_size is divisible by n_run*every.
    # Exact for any `every` (no rounding, no division by zero), and identical
    # to the old i % (every*save_interval) rule whenever n_run == n_size.
    # The final save: the target itself when the run saved it; otherwise the
    # first save past it (a run on the old 45740 schedule never writes 45720),
    # and nothing while the run is still short of the target.
    final = target if target in saved else saved[-1] if saved[-1] > target else target
    return [i for i in saved
            if (i % step == 0 and (i // step) * n_size % (n_run * every) == 0)
            or i == final]


def mix_label(L: int, arch: str = "deep", scheme: str = "A") -> str:
    """Scheme label for EXP_NAME: language setting, data scheme (empty for
    the scheme-A baseline) and the arch — always explicit, e.g. `L8-deep`,
    `L8-schemeB-deep`, `L50-AT3-shallow`, `L2-ZH-deep`."""
    return f"L{L}{DATA_SCHEMES[scheme]['label']}-{arch}"


def exp_name(size: str, L: int, arch: str, seed: int, scheme: str = "A") -> str:
    """Canonical model/cell name (e.g. `lm-90M-L8-deep-seed1904`) — the
    checkpoint dir under Meg-Runs/<project>/, the W&B run id/name, and the
    prefix of eval result ids. `lm` not `apertus`: the architecture has
    diverged from Apertus. pretrain_progress.py parses this format; job
    names derive from it via job_name()."""
    return f"lm-{size}-{mix_label(L, arch, scheme)}-seed{seed}"


def job_name(kind: str, exp: str) -> str:
    """Slurm/Azure job display name: `<kind>-<cell sans the lm- prefix>`,
    e.g. `pretrain-90M-L8-deep-seed1904`, `eval-90M-L8-deep-seed1904-iter425`.
    The model name keeps its `lm-` prefix everywhere else (dirs, W&B,
    models.json); jobs drop it for the kind prefix instead."""
    return f"{kind}-{exp.removeprefix('lm-')}"


def data_blend(english: str, fineweb: str, L: int) -> str:
    """Megatron --data-path value blending English (DCLM) and FineWeb-2.

    L = 1: English only (weight 1.0). L >= 2: English and the setting's
    FineWeb-2 each at the fixed 50% share. The prefixes are the .bin/.idx
    prefixes written by data/build_data_mixtures.py — cluster paths on CSCS,
    AML input mounts on Azure.
    """
    if L == 1:
        return f"1.0 {english}"
    return f"{EN_SHARE / 100:.2f} {english} {(100 - EN_SHARE) / 100:.2f} {fineweb}"


SEQ_LEN = 4096          # megatron_args.sh; tokens per sample
BYTES_PER_TOKEN = 4     # Megatron .bin element size for the 131k vocab
# The durable master every build writes, with a <prefix>.plan.json per build:
# the per-language token estimates the builder allocated from.
DATA_MASTER = Path("/capstor/store/cscs/swissai/infra01/"
                   "multilingual_data_mixtures/predictivity-data")


def undersized_build(prefix: str, L: int, scheme: str, run_tokens: int) -> Optional[str]:
    """Why the FineWeb-2 build at `prefix` must not feed a run of `run_tokens`
    total tokens, or None when it may.

    A run drawing no more than the build holds is always fine. One drawing
    more repeats data, which is accepted only when the SOURCE is the limit —
    the build already realizes what its current target can (scheme A's L2
    Russian, ES's Spanish, AT3 L100 at T=3). A build smaller than that is a
    stale one the grid has since outgrown (the 52B A-L15/A-L50/B-L15 copies,
    once the 1.7B row gained those settings), and training on it would repeat
    data only because nobody swapped the full build in."""
    have = Path(f"{prefix}.bin").stat().st_size // BYTES_PER_TOKEN
    draw = run_tokens * (100 - EN_SHARE) // 100
    if draw <= have:
        return None
    # Per-language token estimates, from ANY build plan on the master: the
    # builder measures a language the same way whatever build it is in, and the
    # oldest builds (scheme A's L2) predate plan files altogether.
    langs = json.loads((SCRIPT_DIR / "data" / f"language_sets_scheme"
                        f"{DATA_SCHEMES[scheme]['sets']}.json").read_text())["sets"][f"FW_L{L}"]
    est: dict[str, int] = {}
    for p in sorted(DATA_MASTER.glob("fineweb_L*.plan.json")) + sorted(
            DATA_MASTER.glob("*/fineweb_L*.plan.json")):
        try:
            sources = json.loads(p.read_text())["sources"]
        except (OSError, ValueError, KeyError):
            continue
        for lang in langs:
            if f"fineweb_{lang}" in sources:
                est.setdefault(lang, sources[f"fineweb_{lang}"]["estimated_tokens"])
    unknown = [lang for lang in langs if lang not in est]
    if unknown:
        return (f"draws {draw / 1e9:.1f}B from a {have / 1e9:.1f}B build, and no "
                f"build plan under {DATA_MASTER} estimates {unknown}, so a source "
                f"limit cannot be told apart from a stale build")
    # What the builder can realize at the CURRENT target: temperature
    # allocation p^(1/T), each language capped at its estimate (it never repeats).
    sys.path.insert(0, str(SCRIPT_DIR / "data"))
    from build_data_mixtures import fineweb_target_tokens  # lazy: it imports this module
    temp, total = DATA_SCHEMES[scheme]["temp"], sum(est.values())
    w = {k: (v / total) ** (1 / temp) for k, v in est.items()}
    z, target = sum(w.values()), fineweb_target_tokens(L, scheme)
    can = sum(min(target * w[k] / z, est[k]) for k in est)
    if have >= 0.98 * can:
        return None
    return (f"draws {draw / 1e9:.1f}B ({draw / have:.2f} epochs) from a "
            f"{have / 1e9:.1f}B build the grid now sizes at {can / 1e9:.1f}B — "
            f"stage the full build first")


# Width-scaled init anchor: 1/sqrt(hidden_size) scaling that keeps the
# reviewed 0.008944 exactly at the 1B width (d=1792), so the init is
# consistent across the 768..3072 ladder instead of one fixed value.
INIT_STD_ANCHOR = 0.008944
INIT_STD_ANCHOR_WIDTH = 1792


def init_std(hidden_size: int) -> float:
    return round(INIT_STD_ANCHOR * (INIT_STD_ANCHOR_WIDTH / hidden_size) ** 0.5, 6)


def cell_env(
    cfg: dict,
    size: str,
    seed: int,
    exp: str,
    blend: str,
    training_steps: Optional[int] = None,
    lr_warmup_iters: Optional[int] = None,
    lr_wsd_decay_iters: Optional[int] = None,
    mbs: Optional[int] = None,
    lr: Optional[float] = None,
    beta3_factor: Optional[float] = None,
    gbs: Optional[int] = None,
) -> dict:
    """The env-var dict megatron_args.sh consumes — the platform-independent
    description of one run. Identical on CSCS and Azure by construction.

    `lr`, `beta3_factor` and `gbs` are DIAGNOSTIC overrides and default to None, in
    which case this returns exactly what the already-pretrained cells used —
    the ladder's comparability rests on that. Callers that pass either get a
    `diag-` run name forced on them (see main())."""
    iters, warmup, decay = schedule_for(cfg)
    # Out-of-range factors do not fail loudly, they train garbage for hours:
    # F * iters <= 1 gives beta3 <= 0, and a negative F gives beta3 > 1.
    beta3 = None if beta3_factor is None else 1 - 1 / (beta3_factor * iters)
    if beta3 is not None and not 0 < beta3 < 1:
        raise ValueError(
            f"--ademamix-beta3-factor {beta3_factor} gives beta3={beta3:.6g} "
            f"over {iters} iters; need 0 < beta3 < 1, i.e. factor > {1/iters:.3g}")
    return {
        "MODEL_SIZE": size,
        "NUM_LAYERS": cfg["n_layers"],
        "HIDDEN_SIZE": cfg["hidden_size"],
        "FFN_HIDDEN_SIZE": cfg["ffn_hidden_size"],
        "NUM_ATTENTION_HEADS": cfg["num_attention_heads"],
        "NUM_QUERY_GROUPS": cfg["num_query_groups"],
        "MBS": mbs if mbs is not None else cfg["micro_batch_size"],
        "TRAINING_STEPS": training_steps if training_steps is not None else iters,
        "LR": lr if lr is not None else cfg["lr"],
        "LR_WARMUP_ITERS": lr_warmup_iters if lr_warmup_iters is not None else warmup,
        "LR_WSD_DECAY_ITERS": (
            lr_wsd_decay_iters if lr_wsd_decay_iters is not None else decay
        ),
        # AdEMAMix alpha/beta3 warm up over the cell's FULL schedule — always
        # the target iters, never a capped resume's --training-steps, so every
        # (re)submission runs the identical optimizer schedule.
        "ADEMAMIX_WARMUP": iters,
        # beta3's ENDPOINT is fixed at 0.9999 for every grid cell; only the
        # warmup above scales with run length. Emitted ONLY for diagnostic
        # runs, so a normal launch's dict stays byte-identical to what the
        # trained cells used and megatron_args.sh keeps its own default —
        # which is what makes "unchanged by default" checkable with a diff.
        # 8 dp, not 6: at the long rungs 1-beta3 is ~6e-5, and 6 dp would
        # round the timescale off by ~0.5%.
        **({"ADEMAMIX_BETA3": f"{beta3:.8f}"} if beta3 is not None else {}),
        # Same rule as ADEMAMIX_BETA3: emitted ONLY when overridden, so a
        # normal launch's dict stays byte-identical to what the trained
        # cells used and megatron_args.sh keeps its own GBS=504.
        **({"GBS": gbs} if gbs is not None else {}),
        "SAVE_INTERVAL": save_interval(iters),
        "INIT_STD": init_std(cfg["hidden_size"]),
        "SEED": seed,
        "EXP_NAME": exp,
        "DATA_BLEND": blend,
        "TOKENIZER_MODEL": TOKENIZER_MODEL,
        "PROJECT_NAME": PROJECT_NAME,
    }


# --- CSCS (sbatch) -----------------------------------------------------------

GBS = 504  # global batch size (megatron_args.sh) — fixed across the sweep

# Cluster scale (nodes) lives in the deep hyperparams file; the shallow ladder
# shares the same non-embedding sizes, so it uses the same node counts.
NODES_BY_SIZE = {
    size: cfg["nodes"]
    for size, cfg in json.loads(HYPERPARAMS["deep"].read_text())["configs"].items()
}


def cscs_mbs(nodes: int, mbs: int, gbs: int = GBS) -> int:
    """Largest micro-batch <= the memory-tuned value that divides the cluster
    layout (DP = 4 GPUs x nodes; Megatron requires GBS % (DP x MBS) == 0) —
    the same resolution launch_pretraining_azure.sh does against its GPU
    count. A no-op for the deep ladder (its values are already valid); the
    shallow ladder's generator-suggested MBS (24/14/8/4/...) needs it."""
    dp = 4 * nodes
    while gbs % (dp * mbs) != 0:
        mbs -= 1
        if mbs < 1:
            raise ValueError(
                f"no micro-batch divides GBS={gbs} over DP={dp}; "
                f"pick a --gbs that is a multiple of {dp}")
    return mbs


# Per-size steady-state iter time (ms), for walltime sizing, keyed by arch
# then size. Values carry ~20% over the measurement: under-estimating walls a
# job and costs a whole queue cycle, over-estimating only lengthens the
# request. (At 600M and above the 11:59:59 cap dominates anyway.)
#
# [m] = median over 500k+ logged iterations from the 2026-08-21..27 runs, the
# window AFTER the training data moved to iopsstor. Sampled mid-run (first 20
# and last 10 iterations dropped) so neither the cold start nor the end-of-run
# async-checkpoint flush is in the number, and pooled across L: iteration cost
# is the same to within 4% at every language setting, because tokens per
# iteration and sequence length do not change with L. Only the batch content
# does. p10..p90 sits within 1-3% of the median at every measured rung, which
# is the band that says compute-bound rather than I/O-bound.
#
# The values these replaced were fitted during the capstor period and were
# inflated by dataloader stalls (CLAUDE.md #8): deep 175M read 2176 against
# 844 now, shallow 90M 4072 against 1154. The power-of-2 GEMM aliasing
# hypothesis attached to them was refuted by a standalone benchmark (flat
# 553-633 TFLOP/s at every ladder shape) — do not reinstate it.
#
# 1B and 1.7B have NO run on this sweep yet, so they are left at their older
# estimates rather than given a fabricated precision. Both need >11h at any
# plausible rate, so auto_time clamps them to the cap either way and the
# walltime is insensitive to the value; the BUDGET is not (they are ~84% of
# the sweep's node-hours), so measure them on the first real run.
ITER_MS = {
    "deep":    {"90M": 1500,   # [m] 1248
                "175M": 1000,  # [m]  844
                "350M": 750,   # [m]  604
                "600M": 660,   # [m]  548
                "1B": 2400, "1.7B": 3200},        # not measured
    "shallow": {"90M": 1400,   # [m] 1154
                "175M": 1000,  # [m]  810
                "350M": 700,   # [m]  567
                "600M": 650,   # [m]  539
                "1B": 2600, "1.7B": 3200},        # not measured
}
TIME_MARGIN_SEC = 9000   # 2h30m: 1h SIGUSR2 grace + cold-start + buffer
TIME_MIN_SEC = 5400      # 1h30m
TIME_MAX_SEC = 43199     # 11:59:59 (slurm normal queue cap)


def auto_time(size: str, remaining_iters: int, arch: str = "deep") -> str:
    """Walltime for a run with `remaining_iters` to go, rounded up to 15 min."""
    total = remaining_iters * ITER_MS[arch].get(size, 2400) // 1000 + TIME_MARGIN_SEC
    total = (total + 899) // 900 * 900
    total = min(max(total, TIME_MIN_SEC), TIME_MAX_SEC)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def active_slurm_jobs() -> set[str]:
    """All queued/running Slurm job names, ANY user (empty off-cluster).

    Deliberately not `--me`, matching auto_evals_cscs.active_jobs(): cells are
    trained into one shared tree, so a collaborator's in-flight pretrain job
    owns that cell's checkpoint dir just as much as ours would. With `--me` we
    could not see it, would read the cell as `resume`, and would submit a
    second Megatron job writing the same --save dir.
    """
    try:
        out = subprocess.run(["squeue", "-h", "--format=%j"],
                             capture_output=True, text=True, timeout=30)
        return set(out.stdout.split()) if out.returncode == 0 else set()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()


def rewind_marker(ckpt_dir: Path, want: int, dry_run: bool) -> bool:
    """Point latest_checkpointed_iteration.txt at `want` (the last VALID
    checkpoint) when it is ahead of it — e.g. after an async-save left a
    shell iter dir the marker already references. Refuses (False) if `want`
    isn't a loadable checkpoint; True when the marker is correct/rewound."""
    marker_file = ckpt_dir / "latest_checkpointed_iteration.txt"
    try:
        current = int(marker_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return True  # no marker (fresh) or unreadable — Megatron will complain
    if current <= want:
        return True
    # Validity first, dry-run second: the preview must refuse exactly what
    # the real run refuses, or it claims a resume that will not happen.
    from pretrain_progress import is_valid_iter_dir  # lazy: avoids import cycle
    if not is_valid_iter_dir(ckpt_dir / f"iter_{want:07d}"):
        print(f"    !! refusing to rewind marker: iter_{want:07d} is not a "
              f"valid checkpoint", file=sys.stderr)
        return False
    if dry_run:
        print(f"    (dry-run) would rewind marker: {current} -> {want}")
        return True
    marker_file.write_text(f"{want}\n")
    print(f"    rewound marker: {current} -> {want}")
    return True


def submit_cscs(env: dict, dry_run: bool, nodes: Optional[int] = None,
                time: Optional[str] = None, account: Optional[str] = None,
                dependency: Optional[str] = None) -> None:
    export_vars = ",".join(f"{k}={v}" for k, v in env.items())
    # PRETRAIN_DIR: sbatch spools the wrapper, so it can't find megatron_args.sh
    # from $0 — pass the real checkout dir here.
    cmd = ["sbatch", f"--job-name={job_name('pretrain', env['EXP_NAME'])}",
           f"--export=ALL,PRETRAIN_DIR={SCRIPT_DIR},{export_vars}"]
    if nodes is not None:
        cmd.append(f"--nodes={nodes}")
    if time is not None:
        cmd.append(f"--time={time}")
    if account is not None:
        cmd.append(f"--account={account}")
    if dependency is not None:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(CSCS_SUBMIT_SCRIPT))

    print(f"  job:    {job_name('pretrain', env['EXP_NAME'])}"
          + (f"  (nodes: {nodes})" if nodes else ""))
    print(f"  export: {export_vars}")
    if dry_run:
        print("  - skipped (dry-run)\n")
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n{result.stdout.strip()}\n")


# --- Azure ML (az ml job create) ---------------------------------------------

def az_args(size: str) -> tuple[str, list[str]]:
    var = "AZ_ML_ARGS_CA" if size in ND_SIZES else "AZ_ML_ARGS_ES"
    try:
        return var, os.environ[var].split()
    except KeyError:
        sys.exit(f"{var} not set — run `source azure/env.sh` first.")


def active_azure_jobs(ws_args: list[str]) -> set[str]:
    """Display names of queued/running jobs in one workspace (empty on any
    az failure — the check is best-effort)."""
    states = {"NotStarted", "Queued", "Starting", "Preparing", "Running", "Finalizing"}
    try:
        # --all-results: the newest-200 default window includes completed
        # jobs, so the eval watcher's convert/eval churn can push a live
        # multi-day training out of it — and a blind dedup check would
        # resubmit the training.
        out = subprocess.run(
            ["az", "ml", "job", "list", "--all-results", "true", *ws_args,
             "--query", "[].{d:display_name,s:status}", "--output", "json"],
            capture_output=True, text=True, timeout=120)
        jobs = json.loads(out.stdout) if out.returncode == 0 else []
        return {j["d"] for j in jobs if j["s"] in states}
    except Exception:
        return set()


def submit_azure(env: dict, cell: dict, dry_run: bool,
                 data_root: Optional[str] = None,
                 compute: Optional[str] = None) -> None:
    size, L, exp = cell["size"], cell["L"], env["EXP_NAME"]
    ws_var, ws_args = az_args(size)
    data_root = data_root or f"{DATASTORE}/data"
    # --compute overrides the size->cluster default. Needed whenever the
    # planned SKU is unavailable: H100 cannot use low-priority at all and has
    # 0 dedicated quota, so the A100 clusters (gpu-nc96-a100-lp /
    # gpu-nc96-a100-ded) are the fallback. Jobs are single-node either way
    # (torchrun --standalone), so only the per-node GPU count changes; the
    # wrapper re-resolves MBS against it.
    overrides = {
        "display_name": job_name("pretrain", exp),
        "compute": compute if compute else (
            "azureml:gpu-nd96-spot" if size in ND_SIZES
            else "azureml:gpu-nc80-lp"),
        # L=1 has no FineWeb blend (data_blend uses inputs.english alone), but
        # the job spec still declares the input and downloads it — pointing it
        # at english_dclm again fetched the 686 GB build twice per monolingual
        # cell. validation/ (2 GB) is the smallest folder every workspace has.
        "inputs.fineweb.path":
            f"{data_root}/{'validation' if L == 1 else f'fineweb_L{L}'}",
        **{f"outputs.{o}.path": f"{DATASTORE}/runs/{exp}/{o}"
           for o in ("checkpoints", "logs", "cache")},
        **{f"environment_variables.{k}": v for k, v in env.items()},
    }
    if os.environ.get("WANDB_API_KEY"):
        overrides["environment_variables.WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    cmd = ["az", "ml", "job", "create", "--file", str(AZURE_JOB_YML), *ws_args]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={v}"]

    print(f"  job: {job_name('pretrain', exp)}  [{ws_var}]  "
          f"({env['TRAINING_STEPS']} iters)")
    if dry_run:
        print("  " + " ".join(c if "WANDB_API_KEY" not in c else
                              "environment_variables.WANDB_API_KEY=***" for c in cmd) + "\n")
    else:
        subprocess.run(cmd, check=True)


# --- Driver ------------------------------------------------------------------

def run_test(data: dict, arch: str, data_dir: str, dry_run: bool) -> None:
    cfg = data["configs"][TEST_SIZE]
    exp = f"lm-test-{TEST_SIZE.lower()}-{mix_label(TEST_LANGS, arch)}"
    blend = data_blend(f"{data_dir}/english_dclm",
                       f"{data_dir}/fineweb_L{TEST_LANGS}", TEST_LANGS)
    print(f"=== Test run: {TEST_SIZE} | {mix_label(TEST_LANGS, arch)} | "
          f"seed {TEST_SEED} | {TEST_STEPS} steps ===\n")
    env = cell_env(cfg, TEST_SIZE, TEST_SEED, exp, blend,
                   training_steps=TEST_STEPS,
                   lr_warmup_iters=TEST_WARMUP, lr_wsd_decay_iters=TEST_DECAY)
    # Save twice inside the short run. cell_env sizes SAVE_INTERVAL for the
    # full schedule (225 at 90M), which TEST_STEPS never reaches — so the
    # async torch_dist save, CLAUDE.md failure mode #2, went unexercised by
    # the very run meant to smoke it out.
    env["SAVE_INTERVAL"] = TEST_STEPS // 2
    submit_cscs(env, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("platform", choices=["cscs", "azure"],
                        help="Where to submit: cscs (sbatch) or azure (az ml)")
    parser.add_argument("--no-auto-evals", action="store_true",
                        help="CSCS only: do NOT start the auto-eval watcher. "
                             "By default a launch also starts it for the arch "
                             "being submitted, so trained cells get converted "
                             "and evaluated without a separate step.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print submit commands without running them")
    parser.add_argument("--arch", choices=["deep", "shallow"], default="deep",
                        help="Architecture family: deep (baseline) or shallow "
                             "(the model-depth intervention level)")
    parser.add_argument("--scheme", choices=list(DATA_SCHEMES), default="A",
                        help="Data scheme to submit: A (resource-ranked, "
                             "T=1 — the baseline), AT3 (same lists at T=3; "
                             "L50 as the temperature intervention and L100, "
                             "which exists only at T=3), B (diversity-first "
                             "lists at L in {8, 15, 30}), ZH / ES (L2 with "
                             "Chinese / Spanish instead of Russian). Each "
                             "reads its own <data_dir>/<subdir> and carries "
                             "its own name label, so schemes never collide.")
    parser.add_argument("--size", metavar="SIZES",
                        help="Filter by size — one or a comma-separated list "
                             "(e.g. '600M' or '350M,175M'). Default: all sizes.")
    parser.add_argument("--langs", metavar="L", type=int,
                        help=f"Filter by language setting (one of {LANG_SETTINGS})")
    parser.add_argument("--seed", metavar="SEED", type=int, help="Filter by seed")
    # CSCS-only knobs
    parser.add_argument("--data_dir", default=CSCS_DEFAULT_DATA_DIR,
                        help="CSCS only: dir holding english_dclm.* and "
                             f"fineweb_L*.* (default: {CSCS_DEFAULT_DATA_DIR})")
    parser.add_argument("--time", metavar="HH:MM:SS",
                        help="CSCS only: override sbatch --time")
    parser.add_argument("--account", metavar="ACCOUNT",
                        help="CSCS only: override sbatch --account (e.g. a139)")
    parser.add_argument("--dependency", metavar="DEP",
                        help="CSCS only: pass-through to sbatch --dependency")
    parser.add_argument("--training-steps", metavar="N", type=int,
                        help="CSCS only: manually cap --train-iters (the "
                             "idempotent resume logic sets this itself)")
    parser.add_argument("--test", action="store_true",
                        help=f"CSCS only: smoke test ({TEST_SIZE}, L{TEST_LANGS}, "
                             f"{TEST_STEPS} steps)")
    parser.add_argument("--compute", metavar="CLUSTER",
                        help="Azure only: override the size->cluster default, "
                             "e.g. gpu-nc96-a100-lp or gpu-nc96-a100-ded when "
                             "no H100 is obtainable (accepts a bare name or "
                             "azureml:<name>)")
    # Diagnostic overrides. Every grid cell must keep the config the trained
    # cells used, so these are opt-in, never defaults, and any run that sets
    # one is renamed diag-* below — it can then never land in a grid cell's
    # checkpoint dir or W&B run id. See plan/90M-rung-anomaly.md.
    parser.add_argument("--lr", metavar="LR", type=float,
                        help="CSCS only, DIAGNOSTIC: override the per-size peak "
                             "LR. Forces a diag- run name — never a grid cell.")
    parser.add_argument("--ademamix-beta3-factor", metavar="F", type=float,
                        help="CSCS only, DIAGNOSTIC: set beta3 = 1 - 1/(F * iters) "
                             "instead of the ladder's fixed 0.9999, i.e. put the "
                             "slow-EMA timescale at F of the run. Forces a diag- "
                             "run name — never a grid cell.")
    parser.add_argument("--gbs", metavar="N", type=int,
                        help="CSCS only, DIAGNOSTIC: override the global batch "
                             "size (ladder value 504). Must be a multiple of "
                             "DP = 4 x nodes. Forces a diag- run name — never a "
                             "grid cell.")
    args = parser.parse_args()

    if args.platform != "cscs":
        # not-in-(None, False), not truthiness: --lr 0 is a mistake worth
        # reporting, not a value to silently drop.
        for flag in ("time", "account", "dependency", "training_steps", "test",
                     "lr", "ademamix_beta3_factor", "gbs"):
            if getattr(args, flag) not in (None, False):
                parser.error(f"--{flag.replace('_', '-')} is CSCS-only")
    elif args.compute:
        parser.error("--compute is Azure-only")

    diag = {k: v for k, v in (("lr", args.lr),
                              ("beta3f", args.ademamix_beta3_factor),
                              ("gbs", args.gbs))
            if v is not None}
    if diag:
        if any(v <= 0 for v in diag.values()):
            parser.error("--lr / --ademamix-beta3-factor / --gbs must be positive")
        # Without a filter one diagnostic flag would fan a non-standard
        # config across every fresh cell in the grid.
        if not (args.size or args.langs or args.seed):
            parser.error("a diagnostic run must be narrowed with "
                         "--size/--langs/--seed")
        # A diag- cell has no models.json entry, so the watcher could not
        # evaluate it anyway.
        args.no_auto_evals = True

    size_filter = args.size.split(",") if args.size else None
    valid_sizes = list(SIZE_LANG_SETTINGS)
    bad_sizes = [s for s in (size_filter or []) if s not in valid_sizes]
    if bad_sizes:
        parser.error(f"--size {bad_sizes} not valid. Choose from: {valid_sizes}")
    if args.gbs:
        # Fail here, not 200 lines later inside cscs_mbs: Megatron needs
        # GBS % (DP x MBS) == 0, and MBS >= 1 makes GBS % DP == 0 the real
        # constraint. DP = 4 GPUs x the size's node count.
        for sz in (size_filter or valid_sizes):
            dp = 4 * NODES_BY_SIZE[sz]
            if args.gbs % dp:
                parser.error(f"--gbs {args.gbs} is not a multiple of DP={dp} "
                             f"({sz} runs on {NODES_BY_SIZE[sz]} nodes); "
                             f"nearest valid: {args.gbs // dp * dp} or "
                             f"{(args.gbs // dp + 1) * dp}")
    if args.langs and args.langs not in LANG_SETTINGS:
        parser.error(f"--langs '{args.langs}' not valid. Choose from: {LANG_SETTINGS}")

    # Accept "gpu-x" or "azureml:gpu-x"; the job yml wants the azureml: form.
    az_compute = (args.compute if args.compute is None
                  or args.compute.startswith("azureml:")
                  else f"azureml:{args.compute}")

    config = HYPERPARAMS[args.arch]
    data = json.loads(config.read_text())
    print(f"Platform: {args.platform} | Config: {config.name} (arch: {args.arch})")
    if args.dry_run:
        print("(dry-run — submit commands will be printed but not executed)")
    print()

    if args.test:
        run_test(data, args.arch, args.data_dir, args.dry_run)
        return

    if args.arch not in DATA_SCHEMES[args.scheme]["arches"]:
        print(f"Scheme {args.scheme} is not trained in the {args.arch} "
              f"architecture (only {', '.join(DATA_SCHEMES[args.scheme]['arches'])}).")
        return
    cells = [
        c for c in predictivity_cells([args.scheme])
        if (size_filter is None or c["size"] in size_filter)
        and (args.langs is None or c["L"] == args.langs)
        and (args.seed is None or c["seed"] == args.seed)
    ]
    if not cells:
        print("No cells match the given filters.")
        return
    print(f"=== {len(cells)} matching cells ===\n")

    # Idempotency state, computed once per invocation.
    if args.platform == "cscs":
        from pretrain_progress import CKPT_ROOT, cell_action  # lazy: no cycle
        active = active_slurm_jobs()
    else:
        active = set() if args.dry_run else set().union(
            *(active_azure_jobs(az_args(s)[1])
              for s in {c["size"] for c in cells}))

    for c in cells:
        cfg = data["configs"][c["size"]]
        # Every non-baseline scheme keeps its FineWeb-2 build in its own
        # subdir of the same layout (the english build and the validation
        # manifest are symlinked in — see data/launch_builds.sh).
        # predictivity_cells() only emits a scheme at the settings it actually
        # defines, so unlike the old two-scheme code there is no "fall back to
        # A" case to guess at here: a cell's scheme IS its data.
        subdir = DATA_SCHEMES[c["scheme"]]["subdir"]
        cell_dir = args.data_dir + (f"/{subdir}" if subdir else "")
        exp = exp_name(c["size"], c["L"], args.arch, c["seed"], c["scheme"])
        if diag:
            # Rename BEFORE anything keys off it. `diag-...` matches neither
            # pretrain_progress.NAME_RE nor ladder_report.LOG_RE, and
            # sync_models_json builds its keys from exp_name(), so a run with
            # a non-standard config is structurally unable to be mistaken for
            # a ladder rung. Not optional, for that reason.
            tag = "".join(f"-{k}{v:g}" for k, v in diag.items())
            exp = f"diag-{exp.removeprefix('lm-')}{tag}"
        target = schedule_for(cfg)[0]

        if job_name("pretrain", exp) in active:
            print(f"  skip [active]: {exp} already queued/running")
            continue

        if args.platform == "cscs":
            action, a, b = cell_action(CKPT_ROOT / exp, target)
            if action == "done":
                print(f"  skip [done]: {exp} (valid checkpoint at target {target})")
                continue
            if action == "corrupt":
                print(f"  *** corrupt: {exp} — {a} iter dir(s) on disk but none "
                      f"valid; SKIPPING (manual review)")
                continue
            # The data must be on the stage BEFORE nodes are allocated: a
            # missing prefix fails every rank at dataset build, minutes into
            # the job (2026-09-11: all six ZH/ES cells, whose builds had never
            # reached the stage). is_file() follows symlinks, so a dangling
            # english_dclm link counts as missing too.
            prefixes = [f"{cell_dir}/english_dclm"] + (
                [f"{cell_dir}/fineweb_L{c['L']}"] if c["L"] > 1 else [])
            missing = [p for p in prefixes
                       if not all(Path(f"{p}.{ext}").is_file() for ext in ("bin", "idx"))]
            if missing:
                print(f"  skip [no data]: {exp} — not staged: {', '.join(missing)}")
                continue
            # A build that exists but is smaller than the grid now sizes it
            # (the 52B A-L15/A-L50/B-L15 copies, once the 1.7B row gained those
            # settings) must not feed a cell that draws more than it holds:
            # Megatron silently repeats it.
            if c["L"] > 1:
                short = undersized_build(f"{cell_dir}/fineweb_L{c['L']}", c["L"],
                                         c["scheme"], target * (args.gbs or GBS) * SEQ_LEN)
                if short:
                    print(f"  skip [data undersized]: {exp} — {short}")
                    continue
            # A run started on another checkpoint grid (aromanou's 1B cells: every
            # 2287 to 45740) must not be resumed from this checkout, which would
            # save every save_interval(target) from here on: the run ends on no
            # single grid, and due_iters() drops its early checkpoints.
            if action == "resume":
                from pretrain_progress import ITER_RE, is_valid_iter_dir  # lazy: cycle
                ckdir = CKPT_ROOT / exp / "checkpoints"
                saved = sorted(int(m.group(1)) for e in ckdir.iterdir()
                               if (m := ITER_RE.match(e.name)) and is_valid_iter_dir(e))
                if len(saved) >= 2 and run_interval(saved) != save_interval(target):
                    print(f"  skip [foreign schedule]: {exp} saved every "
                          f"{run_interval(saved)} iters; this checkout would continue "
                          f"every {save_interval(target)} — resume it from the "
                          f"checkout that started it")
                    continue
            load_iter, tgt = (0, a) if action == "fresh" else (a, b)
            if action == "resume" and not rewind_marker(
                    CKPT_ROOT / exp / "checkpoints", load_iter, args.dry_run):
                continue
            blend = data_blend(f"{cell_dir}/english_dclm",
                               f"{cell_dir}/fineweb_L{c['L']}", c["L"])
            print(f"  [{action}] {exp}: iters {load_iter} -> {tgt}")
            nodes = cfg.get("nodes", NODES_BY_SIZE[c["size"]])
            submit_cscs(
                cell_env(cfg, c["size"], c["seed"], exp, blend,
                         training_steps=args.training_steps or tgt,
                         mbs=cscs_mbs(nodes, cfg["micro_batch_size"],
                                      args.gbs or GBS),
                         lr=args.lr, beta3_factor=args.ademamix_beta3_factor,
                         gbs=args.gbs),
                dry_run=args.dry_run, nodes=nodes,
                time=args.time or auto_time(c["size"], tgt - load_iter, args.arch),
                account=args.account, dependency=args.dependency,
            )
        else:
            # $ENGLISH_DIR/$FINEWEB_DIR, not ${{inputs.*}}: binding expressions
            # are substituted only inside the job yml's `command`, never in an
            # environment_variables value (which is what --set writes). The yml
            # exports the mounts under these names and the wrapper expands them.
            blend = data_blend("$ENGLISH_DIR/english_dclm",
                               f"$FINEWEB_DIR/fineweb_L{c['L']}", c["L"])
            submit_azure(
                cell_env(cfg, c["size"], c["seed"], exp, blend),
                cell=c, dry_run=args.dry_run,
                data_root=(f"{DATASTORE}/data/{subdir}" if subdir else None),
                compute=az_compute,
            )

    if args.platform == "cscs" and not args.no_auto_evals and not args.dry_run:
        # ONE watcher for the whole grid, not one per --arch. The watcher
        # covers every arch and scheme in a pass, so a launch of the deep
        # ladder no longer leaves the shallow and scheme-B cells waiting for
        # a watcher nobody remembered to start.
        watcher = "auto_evals_cscs.py --watch"
        if subprocess.run(["pgrep", "-f", watcher], capture_output=True).returncode:
            AUTO_EVAL_LOGS.mkdir(parents=True, exist_ok=True)
            log = open(AUTO_EVAL_LOGS / "auto_evals.log", "a")
            subprocess.Popen([sys.executable, "auto_evals_cscs.py",
                              "--watch", "1800"], cwd=str(SCRIPT_DIR),
                             stdout=log, stderr=log, start_new_session=True)
            print("started auto-evals (all archs/schemes); opt out with "
                  "--no-auto-evals")
        else:
            print("(auto-evals already watching)")

    # Refresh the training-side figures (plan, simple, detailed) and the
    # generated grid block in README.md / the plan doc. plan_table and
    # sync_docs are derived from the constants in THIS file, so a grid edit
    # reaches the docs on the next launch instead of leaving stale numbers
    # behind. eval_progress.png is NOT redrawn here: it tracks the eval state
    # the auto-eval watcher changes, so the watcher refreshes it each pass.
    # Best-effort: a plotting problem must never fail a submission.
    if args.platform == "cscs" and not args.dry_run:
        try:
            from pretrain_progress import plan_table, sync_docs, update_plots
            update_plots()
            plan_table()
            sync_docs()
        except Exception as e:  # e.g. matplotlib missing off-cluster
            print(f"(progress plots not refreshed: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
