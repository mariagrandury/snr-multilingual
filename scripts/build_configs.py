"""One-time bootstrap: build configs/models.json, configs/tasks.json and
configs/hf_wandb.json from the existing models_*.txt / tasks_*.txt files in
src/evals/configs/signal_to_ratio/.

After this runs, the JSONs are the source of truth — this script is
kept for traceability and re-run only when the canonical data below
changes (it's faithful: a re-run reproduces the JSONs exactly).

models.json schema (per model):
  source, family, size, params, [hyperparams_key, mix_en, mix_fw2, seed],
  checkpoint_kind, backends,
  stages: { <phase>: { tokens, num_iters, tokens_per_iter, checkpoints } }
where <phase> ∈ {pretraining, midtraining, posttraining}. checkpoints holds
`final` + the subset lists (all / 10_ckpts / da_ckpts / dense_tail / full_eval)
— ints for megatron_iter models, branch-name strings for hf_branch models.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
EVALS_CONFIGS = REPO / "src" / "evals" / "configs" / "signal_to_ratio"
HYPERPARAMS_DEEP = REPO / "src" / "pretrain" / "hyperparams_deep.json"


# --- Stage helpers ----------------------------------------------------------

def _meg_tokens_per_iter() -> int:
    """global_batch_size × seq_len from hyperparams_deep.json — the canonical
    megatron tokens-per-iter (504 × 4096 = 2064384)."""
    g = json.loads(HYPERPARAMS_DEEP.read_text())["global"]
    return g["global_batch_size"] * g["seq_len"]


MEG_TOKENS_PER_ITER = _meg_tokens_per_iter()


def _branch_step(branch: str) -> int | None:
    """Step number embedded in an HF branch name — handles `stepN`,
    `step-N`, `stageK-stepN`, `stageK-step-N`, `stepN-tokensXXX`. None for
    `main` (no step)."""
    m = re.search(r"step-?(\d+)", branch)
    return int(m.group(1)) if m else None


def _ckpt_step(ckpt) -> int | None:
    """Step/iteration VALUE of a checkpoint, whether it's a megatron iter int
    or an HF branch-name string. None for stepless branches (e.g. `main`)."""
    if isinstance(ckpt, int):
        return ckpt
    return _branch_step(ckpt)


def _branch_sort_key(branch: str):
    """Order HF branches by step; `main` (no step) sorts last."""
    s = _branch_step(branch)
    return (s is None, s or 0)


def _ckpt_sort_key(ckpt):
    """Order checkpoints (ints or branch strings) by step; stepless sorts last."""
    s = _ckpt_step(ckpt)
    return (s is None, s or 0)


def _da_grid(all_ckpts: list, last_step: int, prev_last: int = 0) -> list:
    """Decision-accuracy checkpoints: the entries of ``all_ckpts`` nearest to
    10/20/30/40/50/100 % of training, measured by step VALUE relative to
    ``last_step`` (NOT list index). 0 % corresponds to ``prev_last`` — 0 for a
    from-scratch first stage, the previous stage's last step for a continuation
    stage. Ties resolve to the higher step. Dedup preserves order.

    Only stepped checkpoints are candidates (stepless branches like `main`
    are skipped)."""
    cands = [c for c in all_ckpts if _ckpt_step(c) is not None]
    span = last_step - prev_last
    picks = []
    for pct in (10, 20, 30, 40, 50, 100):
        target = prev_last + span * pct / 100.0
        best = min(cands, key=lambda c: (abs(_ckpt_step(c) - target), -_ckpt_step(c)))
        picks.append(best)
    seen = set()
    out = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _full_eval(dense_tail: list, da_ckpts: list, key=None) -> list:
    """Canonical eval set = sorted(da_ckpts ∪ dense_tail), stepless dropped."""
    u = {c for c in (set(dense_tail) | set(da_ckpts))
         if _ckpt_step(c) is not None}
    return sorted(u, key=key or _ckpt_sort_key)


def _meg_stage(schedule: dict, prev_last: int = 0) -> dict:
    """A `stages.<phase>` entry for a megatron-iter model.

    `schedule`: {final, all, dense_tail, 10_ckpts} of iter ints. `da_ckpts`
    is recomputed from `all` via the 10/20/30/40/50/100 % da-grid (any
    `da_ckpts` in `schedule` is ignored). Derives `full_eval`;
    tokens/num_iters/tokens_per_iter from the iter count.
    """
    all_ck = schedule.get("all", [])
    num_iters = schedule.get("final")
    ck = {
        "final": num_iters,
        "all": all_ck,
        "dense_tail": schedule.get("dense_tail", []),
        "10_ckpts": schedule.get("10_ckpts", []),
        "da_ckpts": _da_grid(all_ck, num_iters, prev_last) if all_ck else [],
    }
    ck["full_eval"] = _full_eval(ck["dense_tail"], ck["da_ckpts"])
    return {
        "tokens": num_iters * MEG_TOKENS_PER_ITER if num_iters else None,
        "num_iters": num_iters,
        "tokens_per_iter": MEG_TOKENS_PER_ITER,
        "checkpoints": ck,
    }


def _hf_stage(tokens: int | None, ckpt_data: dict, prev_last: int = 0) -> dict:
    """A `stages.<phase>` entry for an hf_branch model.

    `ckpt_data` is either:
      - {final, 10_ckpts, dense_tail} → derives all / da_ckpts / full_eval
      - {final, all}                 → an explicit branch list with no subsets
        (a `dense_tail`/`10_ckpts` may also be supplied, in which case
        da_ckpts/full_eval are derived from the explicit `all` list)
    tokens_per_iter = tokens / num_iters (num_iters = max branch step); the
    per-branch token count is recovered at load time from the branch name
    (`-tokensXXX`) or `step × tokens_per_iter`, so it isn't stored here.

    `prev_last`: 0 % anchor for the da-grid (the previous stage's last step for
    a continuation stage such as midtraining; 0 for a from-scratch stage).
    """
    if "all" in ckpt_data and "10_ckpts" not in ckpt_data:
        all_b = sorted(ckpt_data["all"], key=_branch_sort_key)
        ck = {"final": ckpt_data["final"], "all": all_b,
              "full_eval": list(all_b)}
    else:
        ten, tail = ckpt_data["10_ckpts"], ckpt_data["dense_tail"]
        if "all" in ckpt_data:
            all_b = sorted(ckpt_data["all"], key=_branch_sort_key)
        else:
            all_b = sorted(set(ten) | set(tail), key=_branch_sort_key)
        steps = [s for s in (_branch_step(b) for b in all_b) if s is not None]
        last = max(steps) if steps else 0
        da = _da_grid(all_b, last, prev_last)
        ck = {
            "final": ckpt_data["final"],
            "all": all_b,
            "dense_tail": tail,
            "10_ckpts": ten,
            "da_ckpts": da,
            "full_eval": _full_eval(tail, da, key=_branch_sort_key),
        }
    steps = [s for s in (_branch_step(b) for b in all_b) if s is not None]
    num_iters = max(steps) if steps else None
    return {
        "tokens": tokens,
        "num_iters": num_iters,
        "tokens_per_iter": tokens / num_iters if (tokens and num_iters) else None,
        "checkpoints": ck,
    }


def _main_only_stage(tokens: int | None) -> dict:
    """A `stages.<phase>` entry for a model published only at `main`
    (no intermediate checkpoints)."""
    return {
        "tokens": tokens,
        "num_iters": None,
        "tokens_per_iter": None,
        "checkpoints": {"final": "main", "all": ["main"]},
    }


# --- Apertus custom pretrains (36 cells) ------------------------------------

SIZES = ["175M", "350M", "600M", "1B"]
SIZE_PARAMS = {"175M": 175_000_000, "350M": 350_000_000,
               "600M": 600_000_000, "1B": 1_000_000_000}
# (mix_en, mix_fw2): percentages of FineWeb-Edu and FineWeb-2 in the
# pretraining data mix. The model name embeds them as
# `fwEdu{mix_en}-fw2{mix_fw2}` — the literal `2` is the dataset name
# (FineWeb-2), no separator before the percentage.
MIXES = [(30, 70), (60, 40), (90, 10)]
SEEDS = [28, 1797, 1904]

# Per-seed checkpoint schedules.
#
#   `all`        — every iter actually produced by training.
#   `dense_tail` — last 5 iters (used for trailing-N noise).
#   `10_ckpts`   — 10 evenly-spaced iters (history/progress views).
#   `da_ckpts`   — 5 iters for decision-accuracy computation.
#   `full_eval`  — derived = sorted(set(dense_tail) | set(da_ckpts));
#                  the canonical eval sweep that snr_progress.csv covers.
CKPT_SCHEDULES = {
    1904: {
        "final": 50000,
        "all": [2000, 6000, 12000, 18000, 22000, 28000, 34000, 38000,
                42000, 44000, 46000, 48000, 50000],
        "dense_tail": [42000, 44000, 46000, 48000, 50000],
        "10_ckpts": [6000, 12000, 18000, 22000, 28000, 34000,
                     38000, 42000, 46000, 50000],
        "da_ckpts": [6000, 12000, 22000, 28000, 50000],
    },
    1797: {
        "final": 50000,
        "all": list(range(2000, 50001, 2000)),
        "dense_tail": [42000, 44000, 46000, 48000, 50000],
        "10_ckpts": [6000, 10000, 16000, 20000, 26000, 30000,
                     36000, 40000, 46000, 50000],
        "da_ckpts": [6000, 10000, 20000, 30000, 50000],
    },
    28: {
        "final": 50000,
        "all": list(range(2000, 50001, 2000)),
        "dense_tail": [42000, 44000, 46000, 48000, 50000],
        "10_ckpts": [6000, 10000, 16000, 20000, 26000, 30000,
                     36000, 40000, 46000, 50000],
        "da_ckpts": [6000, 10000, 20000, 30000, 50000],
    },
}

MEG_BASE = "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small"
HF_LOCAL_BASE = "/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints"


def custom_pretrain_entries() -> dict:
    out = {}
    for size in SIZES:
        for mix_en, mix_fw2 in MIXES:
            for seed in SEEDS:
                name = f"apertus-{size}-fwEdu{mix_en}-fw2{mix_fw2}-seed{seed}"
                family = f"apertus-fwEdu{mix_en}-fw2{mix_fw2}-seed{seed}"
                out[name] = {
                    "source": "snr-pretraining-custom",
                    "family": family,
                    "size": size,
                    "params": SIZE_PARAMS[size],
                    "hyperparams_key": size,
                    "mix_en": mix_en,
                    "mix_fw2": mix_fw2,
                    "seed": seed,
                    "checkpoint_kind": "megatron_iter",
                    "backends": {
                        "megatron": f"{MEG_BASE}/{name}/checkpoints/",
                        "hf_local": f"{HF_LOCAL_BASE}/{name}/",
                    },
                    "stages": {
                        "pretraining": _meg_stage(CKPT_SCHEDULES[seed]),
                    },
                }
    return out


# --- a06 pretrains (2 cells) -----------------------------------------------

A06_BASE = "/capstor/store/cscs/swissai/infra01/main_run_megatron/main_run_megatron_ahgele/Megatron-LM/logs/Meg-Runs/main-runs-v1"
HF_STAGING_BASE = "/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints"


def a06_pretrain_entries() -> dict:
    a06_1b_iters = [20000, 40000, 60000, 80000, 100000, 120000, 140000,
                    160000, 180000, 200000, 220000, 240000, 260000, 280000,
                    300000, 320000, 340000, 360000, 370000, 380000]
    a06_3b_iters = [15000, 30000, 45000, 60000, 75000, 90000, 105000,
                    120000, 135000, 150000, 155000, 160000, 165000]
    return {
        "apertus3-1b-21-nodes": {
            "source": "snr-pretraining-a06",
            "family": "apertus3-a06",
            "size": "1B",
            "params": 1_000_000_000,
            "checkpoint_kind": "megatron_iter",
            "backends": {
                "megatron": f"{A06_BASE}/apertus3-1b-21-nodes/checkpoints/",
                "hf_local": f"{HF_STAGING_BASE}/apertus3-1b-21-nodes/"
            },
            "stages": {
                "pretraining": _meg_stage({
                    "final": 380000,
                    "all": a06_1b_iters,
                    "dense_tail": [320000, 340000, 360000, 370000, 380000],
                    "10_ckpts": [20000, 60000, 100000, 140000, 180000,
                                 220000, 260000, 300000, 340000, 380000],
                    # da picks at 10%, 33%, 50%, 66%, 100% of training
                    # (× 380k, snapped to the 20k iter grid).
                    "da_ckpts": [40000, 120000, 200000, 260000, 380000],
                }),
            },
        },
        "apertus3-3b-64-nodes": {
            "source": "snr-pretraining-a06",
            "family": "apertus3-a06",
            "size": "3B",
            "params": 3_000_000_000,
            "checkpoint_kind": "megatron_iter",
            "backends": {
                "megatron": f"{A06_BASE}/apertus3-3b-64-nodes/checkpoints/",
                "hf_local": f"{HF_STAGING_BASE}/apertus3-3b-64-nodes/"
            },
            "stages": {
                "pretraining": _meg_stage({
                    "final": 165000,
                    "all": a06_3b_iters,
                    "dense_tail": [135000, 150000, 155000, 160000, 165000],
                    "10_ckpts": [15000, 30000, 45000, 60000, 75000, 90000,
                                 105000, 120000, 135000, 150000, 165000],
                    # da picks at 10%, 33%, 50%, 66%, 100% of training
                    # (× 165k, snapped to the 15k iter grid).
                    "da_ckpts": [15000, 60000, 90000, 105000, 165000],
                }),
            },
        },
    }


# --- HF reference (pretraining + midtraining + posttraining) ----------------

# Multi-checkpoint HF repos: {model: {phase: ckpt_data}}. ckpt_data is either
# {tokens, final, 10_ckpts, dense_tail} (subsets derived) or {tokens, final,
# all} (an explicit short branch list). Per the SNR convention, `pretraining`
# is the repo's stage1 and `midtraining` is its stage2/3.
HF_CKPTS = {
    "Apertus-8B-2509": {
        "pretraining": {
            "tokens": 15_000_000_000_000,
            "final": "main",
            "10_ckpts": [
                "step400000-tokens1680B", "step750000-tokens3150B",
                "step1194000-tokens5014B", "step1432000-tokens6014B",
                "step1750000-tokens7652B", "step1900000-tokens8912B",
                "step2100000-tokens10592B", "step2250000-tokens11852B",
                "step2450000-tokens13532B", "step2627139-tokens15T",
            ],
            "dense_tail": [
                "step2500000-tokens13952B", "step2550000-tokens14372B",
                "step2600000-tokens14792B", "main",
            ],
        },
    },
    "Olmo-3-1025-7B": {
        "pretraining": {
            "tokens": 6_000_000_000_000,
            "final": "stage1-step1413814",
            "10_ckpts": [
                "stage1-step141000", "stage1-step283000", "stage1-step424000",
                "stage1-step566000", "stage1-step707000", "stage1-step848000",
                "stage1-step990000", "stage1-step1131000",
                "stage1-step1272000", "stage1-step1413814",
            ],
            "dense_tail": [
                "stage1-step1273000", "stage1-step1308000",
                "stage1-step1343000", "stage1-step1379000",
            ],
        },
    },
    "SmolLM3-3B-checkpoints": {
        "pretraining": {
            "tokens": 7_200_000_000_000,
            "final": "stage1-step-3440000",
            "10_ckpts": [
                "stage1-step-360000", "stage1-step-720000",
                "stage1-step-1040000", "stage1-step-1400000",
                "stage1-step-1720000", "stage1-step-2080000",
                "stage1-step-2400000", "stage1-step-2760000",
                "stage1-step-3080000", "stage1-step-3440000",
            ],
            "dense_tail": [
                "stage1-step-3120000", "stage1-step-3200000",
                "stage1-step-3280000", "stage1-step-3360000",
            ],
        },
        # stage2 + stage3 mid-training, every 40000 steps. A continuation of
        # the stage1 pretraining run, so the da-grid's 0 % anchor is stage1's
        # last step (3440000), set via `prev_last` in hf_entry.
        "midtraining": {
            "tokens": 9_900_000_000_000,
            "final": "stage3-step-4720000",
            "all": [
                "stage2-step-3480000", "stage2-step-3520000",
                "stage2-step-3560000", "stage2-step-3600000",
                "stage2-step-3640000", "stage2-step-3680000",
                "stage2-step-3720000", "stage2-step-3760000",
                "stage2-step-3800000", "stage2-step-3840000",
                "stage2-step-3880000", "stage2-step-3920000",
                "stage2-step-3960000", "stage2-step-4000000",
                "stage2-step-4040000", "stage2-step-4080000",
                "stage2-step-4120000", "stage2-step-4160000",
                "stage2-step-4200000", "stage3-step-4240000",
                "stage3-step-4280000", "stage3-step-4320000",
                "stage3-step-4360000", "stage3-step-4400000",
                "stage3-step-4440000", "stage3-step-4480000",
                "stage3-step-4520000", "stage3-step-4560000",
                "stage3-step-4600000", "stage3-step-4640000",
                "stage3-step-4680000", "stage3-step-4720000",
            ],
            "10_ckpts": [
                "stage2-step-3600000", "stage2-step-3720000",
                "stage2-step-3840000", "stage2-step-3960000",
                "stage2-step-4120000", "stage3-step-4240000",
                "stage3-step-4360000", "stage3-step-4480000",
                "stage3-step-4600000", "stage3-step-4720000",
            ],
            "dense_tail": [
                "stage3-step-4560000", "stage3-step-4600000",
                "stage3-step-4640000", "stage3-step-4680000",
                "stage3-step-4720000",
            ],
        },
        # Post-training soup variants, published as named branches. `main` is
        # DROPPED — its config fails vLLM ModelConfig validation (the it-* branch
        # checkpoints load fine; only `main` is broken).
        "posttraining": {
            "tokens": None,
            "final": "it-soup-APO",
            "all": ["it-mid-training", "it-SFT", "it-soup-APO",
                    "it-LC-expert"],
        },
    },
}


def hf_entry(name, hf_url, size, params, family, phase,
             source="huggingface-reference", main_tokens=None):
    """Build one hf_branch model entry. Models in HF_CKPTS get their
    multi-stage checkpoint lists; everything else is a single-stage
    `phase` model published only at `main` (with `main_tokens` if known)."""
    if name in HF_CKPTS:
        stages = {}
        prev_last = 0
        for ph, cd in HF_CKPTS[name].items():
            stages[ph] = _hf_stage(cd["tokens"], cd, prev_last=prev_last)
            ni = stages[ph]["num_iters"]
            if ni is not None:
                prev_last = ni
    else:
        stages = {phase: _main_only_stage(main_tokens)}
    return {
        "source": source,
        "family": family,
        "size": size,
        "params": params,
        "checkpoint_kind": "hf_branch",
        "backends": {"hf": hf_url},
        "stages": stages,
    }


def hf_entries() -> dict:
    # (name, url, size, params, family, phase, source, main_tokens)
    rows = [
        # ---------- Swiss-AI reference: pretraining ----------
        ("Apertus-8B-2509", "https://huggingface.co/swiss-ai/Apertus-8B-2509",
         "8B", 8_000_000_000, "Apertus-2509", "pretraining",
         "swiss-ai-reference", None),
        ("Apertus-70B-2509", "https://huggingface.co/swiss-ai/Apertus-70B-2509",
         "70B", 70_000_000_000, "Apertus-2509", "pretraining",
         "swiss-ai-reference", 15_000_000_000_000),

        # ---------- HF reference: pretraining ----------
        ("Qwen3-0.6B-Base", "https://huggingface.co/Qwen/Qwen3-0.6B-Base",
         "0.6B", 600_000_000, "Qwen3-Base", "pretraining"),
        ("Qwen3-1.7B-Base", "https://huggingface.co/Qwen/Qwen3-1.7B-Base",
         "1.7B", 1_700_000_000, "Qwen3-Base", "pretraining"),
        ("Qwen3-4B-Base", "https://huggingface.co/Qwen/Qwen3-4B-Base",
         "4B", 4_000_000_000, "Qwen3-Base", "pretraining"),
        ("Qwen3-8B-Base", "https://huggingface.co/Qwen/Qwen3-8B-Base",
         "8B", 8_000_000_000, "Qwen3-Base", "pretraining"),
        ("Qwen3-14B-Base", "https://huggingface.co/Qwen/Qwen3-14B-Base",
         "14B", 14_000_000_000, "Qwen3-Base", "pretraining"),
        ("gemma-3-1b-pt", "https://huggingface.co/google/gemma-3-1b-pt",
         "1B", 1_000_000_000, "gemma-3-pt", "pretraining"),
        ("gemma-3-4b-pt", "https://huggingface.co/google/gemma-3-4b-pt",
         "4B", 4_000_000_000, "gemma-3-pt", "pretraining"),
        ("gemma-3-12b-pt", "https://huggingface.co/google/gemma-3-12b-pt",
         "12B", 12_000_000_000, "gemma-3-pt", "pretraining"),
        ("gemma-3-27b-pt", "https://huggingface.co/google/gemma-3-27b-pt",
         "27B", 27_000_000_000, "gemma-3-pt", "pretraining"),
        ("Olmo-3-1025-7B", "https://huggingface.co/allenai/Olmo-3-1025-7B",
         "7B", 7_000_000_000, "Olmo-3-base", "pretraining"),
        ("SmolLM3-3B-checkpoints",
         "https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints",
         "3B", 3_000_000_000, "SmolLM3-checkpoints", "pretraining"),
        ("tiny-aya-base", "https://huggingface.co/CohereLabs/tiny-aya-base",
         "3B", 3_349_200_000, "tiny-aya-base", "pretraining"),

        # ---------- HF reference: posttraining ----------
        ("Apertus-1.7B-it800000-SFT",
         "https://huggingface.co/daslab-testing/Apertus-1.7B-it800000-SFT",
         "1.7B", 1_700_000_000, "Apertus-it-SFT-1", "posttraining",
         "daslab-testing"),
        ("Apertus-70B-Instruct-2509",
         "https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509",
         "70B", 70_000_000_000, "Apertus-Instruct-2509", "posttraining",
         "swiss-ai-reference"),
        ("Apertus-8B-Instruct-2509",
         "https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509",
         "8B", 8_000_000_000, "Apertus-Instruct-2509", "posttraining",
         "swiss-ai-reference"),
        ("Ministral-3-14B-Instruct-2512",
         "https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512",
         "14B", 14_000_000_000, "Ministral-3-Instruct-2512", "posttraining",
         "huggingface-reference"),
        ("Ministral-3-3B-Instruct-2512",
         "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512",
         "3B", 3_000_000_000, "Ministral-3-Instruct-2512", "posttraining",
         "huggingface-reference"),
        ("Ministral-3-8B-Instruct-2512",
         "https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512",
         "8B", 8_000_000_000, "Ministral-3-Instruct-2512", "posttraining",
         "huggingface-reference"),
        ("Mistral-Small-3.2-24B-Instruct-2506",
         "https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506",
         "24B", 24_000_000_000, "Mistral-Small-3.2-Instruct-2506", "posttraining",
         "huggingface-reference"),
        ("Olmo-3-7B-Instruct",
         "https://huggingface.co/allenai/Olmo-3-7B-Instruct",
         "7B", 7_000_000_000, "Olmo-3-Instruct", "posttraining"),
        ("Olmo-3-7B-Instruct-DPO",
         "https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO",
         "7B", 7_000_000_000, "Olmo-3-Instruct-DPO", "posttraining"),
        ("Olmo-3-7B-Instruct-SFT",
         "https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT",
         "7B", 7_000_000_000, "Olmo-3-Instruct-SFT", "posttraining"),
        ("Qwen3-0.6B", "https://huggingface.co/Qwen/Qwen3-0.6B",
         "0.6B", 600_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-14B", "https://huggingface.co/Qwen/Qwen3-14B",
         "14B", 14_000_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-1.7B", "https://huggingface.co/Qwen/Qwen3-1.7B",
         "1.7B", 1_700_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-30B-A3B", "https://huggingface.co/Qwen/Qwen3-30B-A3B",
         "30B-A3B", 30_000_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-4B", "https://huggingface.co/Qwen/Qwen3-4B",
         "4B", 4_000_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-8B", "https://huggingface.co/Qwen/Qwen3-8B",
         "8B", 8_000_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3.5-0.8B", "https://huggingface.co/Qwen/Qwen3.5-0.8B",
         "0.8B", 800_000_000, "Qwen3.5-it", "posttraining"),
        ("Qwen3.5-122B-A10B", "https://huggingface.co/Qwen/Qwen3.5-122B-A10B",
         "122B-A10B", 122_000_000_000, "Qwen3.5-it", "posttraining"),
        ("Qwen3.5-27B", "https://huggingface.co/Qwen/Qwen3.5-27B",
         "27B", 27_000_000_000, "Qwen3.5-it", "posttraining"),
        ("Qwen3.5-2B", "https://huggingface.co/Qwen/Qwen3.5-2B",
         "2B", 2_000_000_000, "Qwen3.5-it", "posttraining"),
        ("Qwen3.5-35B-A3B", "https://huggingface.co/Qwen/Qwen3.5-35B-A3B",
         "35B-A3B", 35_000_000_000, "Qwen3.5-it", "posttraining"),
        ("Qwen3.5-4B", "https://huggingface.co/Qwen/Qwen3.5-4B",
         "4B", 4_000_000_000, "Qwen3.5-it", "posttraining"),
        ("Qwen3.5-9B", "https://huggingface.co/Qwen/Qwen3.5-9B",
         "9B", 9_000_000_000, "Qwen3.5-it", "posttraining"),
        ("SmolLM3-3B", "https://huggingface.co/HuggingFaceTB/SmolLM3-3B",
         "3B", 3_000_000_000, "SmolLM3-it", "posttraining"),
        ("aya-expanse-32b",
         "https://huggingface.co/CohereLabs/aya-expanse-32b",
         "32B", 32_000_000_000, "aya-expanse", "posttraining"),
        ("aya-expanse-8b",
         "https://huggingface.co/CohereLabs/aya-expanse-8b",
         "8B", 8_000_000_000, "aya-expanse", "posttraining"),
        ("gemma-3-12b-it", "https://huggingface.co/google/gemma-3-12b-it",
         "12B", 12_000_000_000, "gemma-3-it", "posttraining", "huggingface-reference"),
        ("gemma-3-1b-it", "https://huggingface.co/google/gemma-3-1b-it",
         "1B", 1_000_000_000, "gemma-3-it", "posttraining", "huggingface-reference"),
        ("gemma-3-270m-it", "https://huggingface.co/google/gemma-3-270m-it",
         "270M", 270_000_000, "gemma-3-it", "posttraining", "huggingface-reference"),
        ("gemma-3-27b-it", "https://huggingface.co/google/gemma-3-27b-it",
         "27B", 27_000_000_000, "gemma-3-it", "posttraining", "huggingface-reference"),
        ("gemma-3-4b-it", "https://huggingface.co/google/gemma-3-4b-it",
         "4B", 4_000_000_000, "gemma-3-it", "posttraining", "huggingface-reference"),
        ("tiny-aya-earth",
         "https://huggingface.co/CohereLabs/tiny-aya-earth",
         "3B", 3_349_200_000, "tiny-aya", "posttraining"),
        ("tiny-aya-fire",
         "https://huggingface.co/CohereLabs/tiny-aya-fire",
         "3B", 3_349_200_000, "tiny-aya", "posttraining"),
        ("tiny-aya-global",
         "https://huggingface.co/CohereLabs/tiny-aya-global",
         "3B", 3_349_200_000, "tiny-aya", "posttraining"),
        ("tiny-aya-water",
         "https://huggingface.co/CohereLabs/tiny-aya-water",
         "3B", 3_349_200_000, "tiny-aya", "posttraining"),

        # ---------- OLMo-2 reference (base + SFT/DPO/Instruct) ----------
        ("OLMo-2-0425-1B", "https://huggingface.co/allenai/OLMo-2-0425-1B",
         "1B", 1_485_000_000, "OLMo-2-base", "pretraining"),
        ("OLMo-2-0425-1B-SFT", "https://huggingface.co/allenai/OLMo-2-0425-1B-SFT",
         "1B", 1_485_000_000, "OLMo-2-Instruct-SFT", "posttraining"),
        ("OLMo-2-0425-1B-DPO", "https://huggingface.co/allenai/OLMo-2-0425-1B-DPO",
         "1B", 1_485_000_000, "OLMo-2-Instruct-DPO", "posttraining"),
        ("OLMo-2-0425-1B-Instruct",
         "https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct",
         "1B", 1_485_000_000, "OLMo-2-Instruct", "posttraining"),
        ("OLMo-2-1124-7B", "https://huggingface.co/allenai/OLMo-2-1124-7B",
         "7B", 7_299_000_000, "OLMo-2-base", "pretraining"),
        ("OLMo-2-1124-7B-SFT", "https://huggingface.co/allenai/OLMo-2-1124-7B-SFT",
         "7B", 7_299_000_000, "OLMo-2-Instruct-SFT", "posttraining"),
        ("OLMo-2-1124-7B-DPO", "https://huggingface.co/allenai/OLMo-2-1124-7B-DPO",
         "7B", 7_299_000_000, "OLMo-2-Instruct-DPO", "posttraining"),
        ("OLMo-2-1124-7B-Instruct",
         "https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct",
         "7B", 7_299_000_000, "OLMo-2-Instruct", "posttraining"),
        ("OLMo-2-1124-13B", "https://huggingface.co/allenai/OLMo-2-1124-13B",
         "13B", 13_716_000_000, "OLMo-2-base", "pretraining"),
        ("OLMo-2-1124-13B-SFT", "https://huggingface.co/allenai/OLMo-2-1124-13B-SFT",
         "13B", 13_716_000_000, "OLMo-2-Instruct-SFT", "posttraining"),
        ("OLMo-2-1124-13B-DPO", "https://huggingface.co/allenai/OLMo-2-1124-13B-DPO",
         "13B", 13_716_000_000, "OLMo-2-Instruct-DPO", "posttraining"),
        ("OLMo-2-1124-13B-Instruct",
         "https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct",
         "13B", 13_716_000_000, "OLMo-2-Instruct", "posttraining"),
        ("OLMo-2-0325-32B", "https://huggingface.co/allenai/OLMo-2-0325-32B",
         "32B", 32_234_000_000, "OLMo-2-base", "pretraining"),
        ("OLMo-2-0325-32B-SFT", "https://huggingface.co/allenai/OLMo-2-0325-32B-SFT",
         "32B", 32_234_000_000, "OLMo-2-Instruct-SFT", "posttraining"),
        ("OLMo-2-0325-32B-DPO", "https://huggingface.co/allenai/OLMo-2-0325-32B-DPO",
         "32B", 32_234_000_000, "OLMo-2-Instruct-DPO", "posttraining"),
        ("OLMo-2-0325-32B-Instruct",
         "https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct",
         "32B", 32_234_000_000, "OLMo-2-Instruct", "posttraining"),

        # ---------- Olmo-3 reference (32B base + Think + 3.1 Instruct) ----------
        ("Olmo-3-1125-32B", "https://huggingface.co/allenai/Olmo-3-1125-32B",
         "32B", 32_234_000_000, "Olmo-3-base", "pretraining"),
        ("Olmo-3-7B-Think", "https://huggingface.co/allenai/Olmo-3-7B-Think",
         "7B", 7_298_000_000, "Olmo-3-Think", "posttraining"),
        ("Olmo-3-7B-Think-SFT",
         "https://huggingface.co/allenai/Olmo-3-7B-Think-SFT",
         "7B", 7_298_000_000, "Olmo-3-Think-SFT", "posttraining"),
        ("Olmo-3-7B-Think-DPO",
         "https://huggingface.co/allenai/Olmo-3-7B-Think-DPO",
         "7B", 7_298_000_000, "Olmo-3-Think-DPO", "posttraining"),
        ("Olmo-3-32B-Think", "https://huggingface.co/allenai/Olmo-3-32B-Think",
         "32B", 32_234_000_000, "Olmo-3-Think", "posttraining"),
        ("Olmo-3-32B-Think-SFT",
         "https://huggingface.co/allenai/Olmo-3-32B-Think-SFT",
         "32B", 32_234_000_000, "Olmo-3-Think-SFT", "posttraining"),
        ("Olmo-3-32B-Think-DPO",
         "https://huggingface.co/allenai/Olmo-3-32B-Think-DPO",
         "32B", 32_234_000_000, "Olmo-3-Think-DPO", "posttraining"),
        ("Olmo-3.1-32B-Instruct",
         "https://huggingface.co/allenai/Olmo-3.1-32B-Instruct",
         "32B", 32_234_000_000, "Olmo-3-Instruct", "posttraining"),
        ("Olmo-3.1-32B-Instruct-SFT",
         "https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-SFT",
         "32B", 32_234_000_000, "Olmo-3-Instruct-SFT", "posttraining"),
        ("Olmo-3.1-32B-Instruct-DPO",
         "https://huggingface.co/allenai/Olmo-3.1-32B-Instruct-DPO",
         "32B", 32_234_000_000, "Olmo-3-Instruct-DPO", "posttraining"),

        # ---------- gemma-4 reference (pt + it) ----------
        ("gemma-4-E2B", "https://huggingface.co/google/gemma-4-E2B",
         "E2B", 2_000_000_000, "gemma-4-pt", "pretraining"),
        ("gemma-4-E2B-it", "https://huggingface.co/google/gemma-4-E2B-it",
         "E2B", 2_000_000_000, "gemma-4-it", "posttraining", "posttraining"),
        ("gemma-4-E4B", "https://huggingface.co/google/gemma-4-E4B",
         "E4B", 8_000_000_000, "gemma-4-pt", "pretraining"),
        ("gemma-4-E4B-it", "https://huggingface.co/google/gemma-4-E4B-it",
         "E4B", 8_000_000_000, "gemma-4-it", "posttraining", "posttraining"),
        ("gemma-4-26B-A4B", "https://huggingface.co/google/gemma-4-26B-A4B",
         "26B-A4B", 26_540_000_000, "gemma-4-pt", "pretraining"),
        ("gemma-4-26B-A4B-it", "https://huggingface.co/google/gemma-4-26B-A4B-it",
         "26B-A4B", 26_540_000_000, "gemma-4-it", "posttraining", "posttraining"),
        ("gemma-4-31B", "https://huggingface.co/google/gemma-4-31B",
         "31B", 32_680_000_000, "gemma-4-pt", "pretraining"),
        ("gemma-4-31B-it", "https://huggingface.co/google/gemma-4-31B-it",
         "31B", 32_680_000_000, "gemma-4-it", "posttraining", "posttraining"),

        # ---------- misc reference pretrains ----------
        ("gemma-3-270m", "https://huggingface.co/google/gemma-3-270m",
         "270M", 268_000_000, "gemma-3-pt", "pretraining"),
        ("Qwen3.5-4B-Base", "https://huggingface.co/Qwen/Qwen3.5-4B-Base",
         "4B", 4_000_000_000, "Qwen3.5-Base", "pretraining"),
    ]
    out = {}
    for row in rows:
        name, url, size, params, family, phase = row[:6]
        source = row[6] if len(row) > 6 else "huggingface-reference"
        main_tokens = row[7] if len(row) > 7 else None
        out[name] = hf_entry(name, url, size, params, family, phase,
                             source=source, main_tokens=main_tokens)
    return out


# --- Distillation (treated as pretraining for SNR-style eval) ---------------
#
# Megatron checkpoints from the swissai distillation runs (ap-from8b-TOP256).
# Both models are trained to iter 800000 with checkpoints every 20000 iters.
# 0.6b starts at iter 20000 (40 regular iters); 1b starts at iter 120000
# (35 regular iters — the early ones aren't on disk). Both have a handful of
# non-canonical save points (e.g. iter_0125067) that we ignore for eval.
DISTILL_BASE = "/capstor/store/cscs/swissai/infra01/distillation/checkpoints/distill"

DISTILL_SCHEDULES = {
    "apertus-0.6b-from8b-TOP256-long": {
        # 40 iters in [20000, 800000] step 20000
        "final": 800000,
        "all": list(range(20000, 800001, 20000)),
        "dense_tail": [720000, 740000, 760000, 780000, 800000],
        # 10 evenly spaced, every 80k from 80k to 800k
        "10_ckpts": [80000, 160000, 240000, 320000, 400000, 480000,
                     560000, 640000, 720000, 800000],
        # da picks at 10%, 33%, 50%, 66%, 100% of training (snapped to 20k grid)
        "da_ckpts": [80000, 260000, 400000, 520000, 800000],
    },
    "apertus-1b-from8b-TOP256-long": {
        # 35 iters in [120000, 800000] step 20000
        "final": 800000,
        "all": list(range(120000, 800001, 20000)),
        "dense_tail": [720000, 740000, 760000, 780000, 800000],
        # 10 evenly spaced from 120k to 800k (every ~75k, snapped to 20k grid)
        "10_ckpts": [120000, 160000, 240000, 320000, 400000, 480000,
                     560000, 640000, 720000, 800000],
        # da picks at 10%, 33%, 50%, 66%, 100% of training (snapped to 20k grid);
        # 10% would be 80k but 1b's earliest on-disk iter is 120k, so use 120k.
        "da_ckpts": [120000, 260000, 400000, 520000, 800000],
    },
}


def distill_entries() -> dict:
    rows = [
        ("apertus-0.6b-from8b-TOP256-long", "0.6B", 600_000_000,
         "ap0.6b-from8b-TOP256-long"),
        ("apertus-1b-from8b-TOP256-long", "1B", 1_000_000_000,
         "ap1b-from8b-TOP256-long"),
    ]
    return {
        name: {
            "source": "distillation",
            "family": "ap-from8b-TOP256",
            "size": size,
            "params": params,
            "checkpoint_kind": "megatron_iter",
            "backends": {
                "megatron": f"{DISTILL_BASE}/{dir_name}/checkpoints/",
                "hf_local": f"{HF_LOCAL_BASE}/{name}/",
            },
            "stages": {
                "pretraining": _meg_stage(DISTILL_SCHEDULES[name]),
            },
        }
        for name, size, params, dir_name in rows
    }


# --- Tokenizer-lm (Clara, sizes TBD) ----------------------------------------

TOKLM_BASE = "/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/hf-checkpoints"


def tokenizer_lm_entries() -> dict:
    rows = ["tiny", "small", "pilot", "full"]
    return {
        f"tokenizer-lm-{r}-128k-apertus": {
            "source": "tokenizer-lm",
            "family": "tokenizer-lm-128k-apertus",
            "size": "TBD",
            "params": None,
            "checkpoint_kind": "hf_local",
            "backends": {"hf_local": f"{TOKLM_BASE}/{r}-128k-apertus"},
            "stages": {"pretraining": _main_only_stage(None)},
        }
        for r in rows
    }


# --- Pools ------------------------------------------------------------------

# Non-custom reference pools are evaluated on ALL three task groups regardless
# of the model's nominal stage (comprehensive cross-stage SNR over the
# reference models). The custom seeds_* pools keep pretraining_full only.
ALL_EVAL_GROUPS = ["pretraining_full", "midtraining", "posttraining"]

POOLS = {
    "seeds_1904": {
        "description": "Single-seed Apertus baseline (3 mixes × 1 seed). "
                       "Held-out test for the framework-generalization story.",
        "stage": "pretraining",
        "members": [{"source": "snr-pretraining-custom", "seeds": [1904]}],
        "include_external": True,
    },
    "seeds_28_1797": {
        "description": "Train pool for framework generalization (3 mixes × 2 seeds).",
        "stage": "pretraining",
        "members": [{"source": "snr-pretraining-custom", "seeds": [28, 1797]}],
        "include_external": True,
    },
    "seeds_28_1797_1904": {
        "description": "Pooled all Apertus seeds (3 mixes × 3 seeds = 9 "
                       "model_families per size). Recommended for downstream "
                       "allenai_comparison / benchmark_creation work.",
        "stage": "pretraining",
        "members": [{"source": "snr-pretraining-custom",
                     "seeds": [28, 1797, 1904]}],
        "include_external": True,
        "eval_groups": ["pretraining_full"],
    },
    "pretraining_a06": {
        "description": "Apertus3 a06 main-run checkpoints (1B and 3B).",
        "stage": "pretraining",
        "members": [{"source": "snr-pretraining-a06"}],
        "include_external": False,
        "eval_groups": ALL_EVAL_GROUPS,
    },
    "pretraining_distill": {
        "description": "Distillation checkpoints (ap-from8b-TOP256, 0.6B + 1B). "
                       "Treated as pretraining-style runs for SNR evaluation.",
        "stage": "pretraining",
        "members": [{"source": "distillation"}],
        "include_external": False,
        "eval_groups": ALL_EVAL_GROUPS,
    },
    "pretraining_hf_reference": {
        "description": "All HF / Swiss-AI reference pretraining ckpts.",
        "stage": "pretraining",
        "members": [
            {"source": "huggingface-reference"},
            {"source": "swiss-ai-reference"},
        ],
        "eval_groups": ALL_EVAL_GROUPS,
    },
    "posttraining_hf_reference": {
        "description": "All HF / Swiss-AI reference posttraining ckpts.",
        "stage": "posttraining",
        "members": [
            {"source": "huggingface-reference"},
            {"source": "swiss-ai-reference"},
        ],
        "eval_groups": ALL_EVAL_GROUPS,
    },
    "midtraining_hf_reference": {
        "description": "HF / Swiss-AI reference models with a midtraining "
                       "stage (currently SmolLM3-3B-checkpoints stage2/stage3).",
        "stage": "midtraining",
        "members": [
            {"source": "huggingface-reference"},
            {"source": "swiss-ai-reference"},
        ],
        "eval_groups": ALL_EVAL_GROUPS,
    },
    # --- Analysis-only model-set tiers (ported from refactor/shared_config).
    # Consumed by the pool-driven signal-and-noise analysis, NOT the eval
    # launchers. `external` spans pretraining+posttraining (stage 'all') over
    # every non-custom model, all tasks, no stage/name filtering. ---
    "external": {
        "description": "Model-set tier: every non-custom model pooled across "
                       "all four external parquets (reference_hf + a06 + "
                       "distillation + posttraining), all models and all tasks, "
                       "with no stage/name filtering. Cross-model signal/noise "
                       "and within-family scaling DA over the external ladder. "
                       "Stage 'all' (spans pretraining + posttraining).",
        "stage": "all",
        "members": [
            {"source": "huggingface-reference"},
            {"source": "swiss-ai-reference"},
            {"source": "snr-pretraining-a06"},
            {"source": "distillation"},
            {"source": "posttraining"},
        ],
        "load_all_external": True,
    },
    "custom_swissai_hf": {
        "description": "Model-set tier: all 3 custom seeds + every external "
                       "pretraining-checkpoint model (swiss-ai-reference + "
                       "huggingface-reference + a06 + distillation, filtered to "
                       "the pretraining stage). Maximum statistical power and "
                       "the >1B scaling ladder.",
        "stage": "pretraining",
        "members": [{"source": "snr-pretraining-custom", "seeds": [28, 1797, 1904]}],
        "include_external": True,
        "eval_groups": ["pretraining_full"],
    },
}

# Pools whose members get stage-level `eval_groups` written onto the built
# models. Excludes the seeds_1904 / seeds_28_1797 sub-pools (covered by the
# pooled seeds_28_1797_1904). Order is irrelevant — assignment is a union.
EVAL_GROUP_POOLS = [
    "seeds_28_1797_1904", "pretraining_a06", "pretraining_distill",
    "pretraining_hf_reference", "posttraining_hf_reference",
    "midtraining_hf_reference", "custom_swissai_hf",
]

# Default stage → eval-group when a pool has no explicit `eval_groups`.
DEFAULT_STAGE_EVAL_GROUP = {
    "pretraining": "pretraining_full",
    "midtraining": "midtraining",
    "posttraining": "posttraining",
}

# Model-name prefixes whose archs are multimodal / unsupported by the eval
# harness — they never receive eval_groups.
EVAL_GROUP_EXCLUDE_PREFIXES = ("Qwen3.5", "gemma-4")


def assign_eval_groups(models: dict) -> None:
    """Set stage-level `eval_groups` on built models in place, driven by pool
    membership. For each in-scope pool, expand its members (match by `source`,
    and `seed` when the member dict carries `seeds`) and union the pool's
    eval-groups onto the stage matching the pool's `stage`. Models whose name
    starts with an excluded prefix get no eval_groups."""
    for pool_name in EVAL_GROUP_POOLS:
        pool = POOLS[pool_name]
        stage = pool["stage"]
        groups = pool.get("eval_groups",
                          [DEFAULT_STAGE_EVAL_GROUP[stage]])
        for name, model in models.items():
            if name.startswith(EVAL_GROUP_EXCLUDE_PREFIXES):
                continue
            if stage not in model["stages"]:
                continue
            for member in pool["members"]:
                if member["source"] != model["source"]:
                    continue
                if "seeds" in member and model.get("seed") not in member["seeds"]:
                    continue
                st = model["stages"][stage]
                existing = st.get("eval_groups", [])
                merged = list(existing)
                for g in groups:
                    if g not in merged:
                        merged.append(g)
                st["eval_groups"] = merged
                break


# --- Sources → parquet split ------------------------------------------------
#
# Every model's `source` maps to exactly one parquet split in the published
# `multilingual-snr/multilingual-snr-eval-results` dataset. `split: null`
# means the source is evaluated but not published (build_hf_dataset.py skips
# its rows). The parquet filename is `{split}-00000-of-00001.parquet` (see
# HF_WANDB["parquet_pattern"]).
SOURCES = {
    "snr-pretraining-custom": {"split": "pretraining_custom"},
    "snr-pretraining-a06":    {"split": "pretraining_a06"},
    "swiss-ai-reference":     {"split": "reference_hf"},
    "huggingface-reference":  {"split": "reference_hf"},
    "posttraining":           {"split": "posttraining"},
    "distillation":           {"split": "distillation"},
    "daslab-testing":         {"split": None},
    "tokenizer-lm":           {"split": None},
}


# --- SNR analysis parameters ------------------------------------------------
#
# SNR is always computed on the custom 4-size ladder (175M/350M/600M/1B);
# decision accuracy uses those plus larger reference models. These are
# global — there is no per-pool / per-family override.
SNR = {
    "small_sizes": ["175M", "350M", "600M"],
    "target_size": "1B",
    "plotted_mixes": ["fwEdu30", "fwEdu60", "fwEdu90"],
    "da_early_steps": [6000, 18000, 28000],
    "last_n": 5,
}


# --- HF / W&B infra config (configs/hf_wandb.json) --------------------------

HF_WANDB = {
    "repo_id": "multilingual-snr/multilingual-snr-eval-results",
    "parquet_pattern": "{split}-00000-of-00001.parquet",
    "wandb": {
        "entity": "mariagrandury-epflnlp",
        "project": "snr-experiments",
    },
    "multilingual_evals": {
        # raw/<bench>/<model_dir>/.../results_*.json sources to merge from.
        # epfl-nlp/multilingual-evals is intentionally omitted (private
        # storage over quota → 403); the multilingual-snr repo is a superset.
        "raw_source_repos": ["multilingual-snr/multilingual-snr-eval-results"],
        # remote raw/ model-dir name → configs/models.json key.
        "model_dirs": {
            "apertus-8b-2509": "Apertus-8B-2509",
            "olmo-3-1025-7b": "Olmo-3-1025-7B",
            "smollm3-3b-checkpoints": "SmolLM3-3B-checkpoints",
        },
    },
}


# --- Tasks ------------------------------------------------------------------

# Language token → ISO code mapping mirrors multilingual/analyze_snr_variants.py
# Covers iso2 + iso3 + full English name forms so subtask names like
# `_arabic`, `_korean`, `_portuguese` auto-derive correctly. Project-scoped
# canonical codes are defined in configs/languages.json.
LANG_MAP = {
    "ar": "ar", "arb": "ar", "ara": "ar", "arabic": "ar",
    "de": "de", "deu": "de", "ger": "de", "german": "de",
    "en": "en", "eng": "en", "english": "en",
    "es": "es", "spa": "es", "spanish": "es",
    "eu": "eu", "eus": "eu", "baq": "eu", "basque": "eu",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr",
    "hi": "hi", "hin": "hi", "hindi": "hi",
    "ja": "ja", "jp": "ja", "jpn": "ja", "japanese": "ja",
    "ko": "ko", "kor": "ko", "korean": "ko",
    "pt": "pt", "por": "pt", "portuguese": "pt",
    "ru": "ru", "rus": "ru", "russian": "ru",
    "sw": "sw", "swh": "sw", "swa": "sw", "swahili": "sw",
    "te": "te", "tel": "te", "telugu": "te",
    "th": "th", "tha": "th", "thai": "th",
    "tr": "tr", "tur": "tr", "turkish": "tr",
    "uk": "uk", "ukr": "uk", "ukrainian": "uk",
    "vi": "vi", "vie": "vi", "vietnamese": "vi",
    "zh": "zh", "zho": "zh", "cmn": "zh", "chinese": "zh", "mandarin": "zh",
}


def _benchmark_of(task: str) -> str:
    """Strip language/script suffix to get the benchmark family name."""
    if task in {"arc_challenge", "arc_easy"}:
        return "arc"
    # Suite-prefix shortcuts: collapse per-language siblings under one
    # benchmark family even when the language suffix is a full name
    # (e.g. include_base_44_tamil → include_base_44).
    for prefix in ("include_base_44_",):
        if task.startswith(prefix):
            return prefix.rstrip("_")
    parts = task.split("_")
    out = []
    for p in parts:
        if p in LANG_MAP:
            break
        out.append(p)
    return "_".join(out) if out else parts[0]


def _language_of(task: str) -> str:
    eng_only = {"arc_challenge", "arc_easy", "commonsense_qa", "hellaswag",
                "mmlu", "openbookqa", "piqa", "truthfulqa_mc1"}
    if task in eng_only:
        return "en"
    for tok in task.split("_"):
        if tok in LANG_MAP:
            return LANG_MAP[tok]
    return "??"


def _read_list(path: Path) -> list[str]:
    """Read one task per line. Drop blanks, `#` comments, `===` headers,
    indented annotations, and `.py` filename leftovers — `tasks_pretraining_extra.txt`
    has a free-form 'datasets with a .py loader' note block we don't want
    in the task list."""
    out = []
    for raw in path.read_text().splitlines():
        s = raw.rstrip()
        if not s or not s.strip():
            continue
        if s.lstrip().startswith("#"):
            continue
        if s != s.lstrip():
            continue  # indented annotation
        if "===" in s or s.endswith(".py"):
            continue
        out.append(s.strip())
    return out


def build_tasks_json(merge_existing: bool = True) -> dict:
    """Build the {tasks, groups} dict.

    When ``merge_existing`` is True (default) and configs/tasks.json
    already exists, every task entry's ``language`` field is preserved
    from the on-disk copy so manual annotations (e.g. ``"multi"`` for
    multilingual benchmarks, language tags the auto-derivation can't
    infer) are not clobbered on re-build.
    """
    group_files = {
        "pretraining_full":         EVALS_CONFIGS / "tasks_pretraining_full.txt",
        "pretraining_classic":      EVALS_CONFIGS / "tasks_pretraining_classic.txt",
        "pretraining_extra":        EVALS_CONFIGS / "tasks_pretraining_extra.txt",
        "pretraining_random":       EVALS_CONFIGS / "tasks_pretraining_random.txt",
        "pretraining_allenai":      EVALS_CONFIGS / "tasks_pretraining_allenai.txt",
        "midtraining":              EVALS_CONFIGS / "tasks_midtraining.txt",
        "posttraining":             EVALS_CONFIGS / "tasks_posttraining.txt",
        # Posttraining tasks that need an external LLM-as-judge (CSCS_SERVING_API).
        # Split out of `posttraining` so the main sweep doesn't hang on
        # judge-call timeouts when the serving key isn't configured.
        "posttraining_llm_judge":   EVALS_CONFIGS / "tasks_posttraining_llm_judge.txt",
        "include_base_44":          EVALS_CONFIGS / "tasks_include.txt",
        "test":                     EVALS_CONFIGS / "tasks_test.txt",
    }
    groups = {g: _read_list(p) for g, p in group_files.items() if p.exists()}

    # task → set of groups it belongs to
    task_to_groups: dict[str, set[str]] = {}
    for g, tasks in groups.items():
        for t in tasks:
            task_to_groups.setdefault(t, set()).add(g)

    # task → set of stages derived from group memberships
    GROUP_TO_STAGE = {
        "pretraining_full": "pretraining",
        "pretraining_classic": "pretraining",
        "pretraining_extra": "pretraining",
        "pretraining_random": "pretraining",
        "pretraining_allenai": "pretraining",
        "include_base_44": "pretraining",
        "midtraining": "midtraining",
        "posttraining": "posttraining",
        "posttraining_llm_judge": "posttraining",
        "test": "pretraining",
    }

    # Manual `language` annotations from the existing tasks.json
    # (preserves "multi", and language tags the auto-derivation
    # doesn't infer, e.g. "cn"/"jp" instead of the canonical "zh"/"ja").
    existing_lang: dict[str, str] = {}
    existing_path = CONFIGS / "tasks.json"
    if merge_existing and existing_path.exists():
        try:
            prev = json.loads(existing_path.read_text())
            for t, e in prev.get("tasks", {}).items():
                if isinstance(e, dict) and "language" in e:
                    existing_lang[t] = e["language"]
        except Exception:
            pass

    tasks_section = {}
    for t, gs in sorted(task_to_groups.items()):
        stages = sorted({GROUP_TO_STAGE[g] for g in gs
                         if g in GROUP_TO_STAGE})
        # Manual annotation wins over auto-derivation (so "multi" / "cn" /
        # etc. survive re-builds). Auto-derive is the fallback for newly
        # added tasks. `??` is the failure marker — not a real annotation —
        # so it's re-derived (lets LANG_MAP extensions take effect on a
        # rebuild for previously-unresolved tokens).
        prev = existing_lang.get(t)
        lang = prev if prev and prev != "??" else _language_of(t)
        tasks_section[t] = {
            "language": lang,
            "benchmark": _benchmark_of(t),
            "stages": stages,
        }
    # Synthetic launch group: union of the three stage groups, so ONE job per
    # checkpoint covers pretraining + midtraining + posttraining in a single
    # BATCH_TASKS=1 lm_eval call (avoids the same-NAME collision of launching
    # the three groups as separate jobs). Dedup preserves first-seen order.
    groups["all_stages"] = list(dict.fromkeys(
        groups.get("pretraining_full", [])
        + groups.get("midtraining", [])
        + groups.get("posttraining", [])))

    return {"tasks": tasks_section, "groups": groups}


# --- Driver -----------------------------------------------------------------

def build_models() -> dict:
    """Build the full {models, pools, sources, snr} dict, including the
    stage-level `eval_groups` assignment. Pure / side-effect-free so callers
    can diff it against the on-disk configs/models.json without writing."""
    models = {}
    models.update(custom_pretrain_entries())
    models.update(a06_pretrain_entries())
    models.update(hf_entries())
    models.update(distill_entries())
    models.update(tokenizer_lm_entries())
    assign_eval_groups(models)
    return {"models": models, "pools": POOLS, "sources": SOURCES, "snr": SNR}


def main():
    models_json = build_models()
    models = models_json["models"]

    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / "models.json").write_text(
        json.dumps(models_json, indent=2, ensure_ascii=True) + "\n"
    )
    print(f"Wrote configs/models.json — {len(models)} models, "
          f"{len(POOLS)} pools, {len(SOURCES)} sources")

    (CONFIGS / "hf_wandb.json").write_text(
        json.dumps(HF_WANDB, indent=2) + "\n"
    )
    print(f"Wrote configs/hf_wandb.json — repo {HF_WANDB['repo_id']}")

    tasks_json = build_tasks_json()
    (CONFIGS / "tasks.json").write_text(
        json.dumps(tasks_json, indent=2) + "\n"
    )
    print(f"Wrote configs/tasks.json — {len(tasks_json['tasks'])} tasks, "
          f"{len(tasks_json['groups'])} groups")


if __name__ == "__main__":
    main()
