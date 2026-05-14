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


def _branch_sort_key(branch: str):
    """Order HF branches by step; `main` (no step) sorts last."""
    s = _branch_step(branch)
    return (s is None, s or 0)


def _pick5(lst: list) -> list:
    """5 ~evenly-spaced picks from a list (HF da_ckpts). Returns the list
    unchanged if it has ≤ 5 entries."""
    if len(lst) <= 5:
        return list(lst)
    idx = sorted({round(i * (len(lst) - 1) / 4) for i in range(5)})
    return [lst[i] for i in idx]


def _full_eval(dense_tail: list, da_ckpts: list, key=None) -> list:
    """Canonical eval set = sorted(dense_tail ∪ da_ckpts)."""
    u = set(dense_tail) | set(da_ckpts)
    return sorted(u, key=key)


def _meg_stage(schedule: dict) -> dict:
    """A `stages.<phase>` entry for a megatron-iter model.

    `schedule`: {final, all, dense_tail, 10_ckpts, da_ckpts} of iter ints.
    Derives `full_eval`; tokens/num_iters/tokens_per_iter from the iter count.
    """
    ck = {
        "final": schedule.get("final"),
        "all": schedule.get("all", []),
        "dense_tail": schedule.get("dense_tail", []),
        "10_ckpts": schedule.get("10_ckpts", []),
        "da_ckpts": schedule.get("da_ckpts", []),
    }
    ck["full_eval"] = _full_eval(ck["dense_tail"], ck["da_ckpts"])
    num_iters = ck["final"]
    return {
        "tokens": num_iters * MEG_TOKENS_PER_ITER if num_iters else None,
        "num_iters": num_iters,
        "tokens_per_iter": MEG_TOKENS_PER_ITER,
        "checkpoints": ck,
    }


def _hf_stage(tokens: int | None, ckpt_data: dict) -> dict:
    """A `stages.<phase>` entry for an hf_branch model.

    `ckpt_data` is either:
      - {final, 10_ckpts, dense_tail} → derives all / da_ckpts / full_eval
      - {final, all}                 → an explicit short branch list (e.g.
                                       a midtraining stage with no subsets)
    tokens_per_iter = tokens / num_iters (num_iters = max branch step); the
    per-branch token count is recovered at load time from the branch name
    (`-tokensXXX`) or `step × tokens_per_iter`, so it isn't stored here.
    """
    if "all" in ckpt_data:
        all_b = sorted(ckpt_data["all"], key=_branch_sort_key)
        ck = {"final": ckpt_data["final"], "all": all_b}
    else:
        ten, tail = ckpt_data["10_ckpts"], ckpt_data["dense_tail"]
        da = _pick5(ten)
        all_b = sorted(set(ten) | set(tail), key=_branch_sort_key)
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
                "megatron": f"{A06_BASE}/apertus3-1b-21-nodes/checkpoints/"
            },
            "stages": {
                "pretraining": _meg_stage({
                    "final": 380000,
                    "all": a06_1b_iters,
                    "dense_tail": [320000, 340000, 360000, 370000, 380000],
                    "10_ckpts": [20000, 60000, 100000, 140000, 180000,
                                 220000, 260000, 300000, 340000, 380000],
                    "da_ckpts": [20000, 60000, 100000, 140000, 180000,
                                 220000, 260000, 300000, 340000, 380000],
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
                "megatron": f"{A06_BASE}/apertus3-3b-64-nodes/checkpoints/"
            },
            "stages": {
                "pretraining": _meg_stage({
                    "final": 165000,
                    "all": a06_3b_iters,
                    "dense_tail": [135000, 150000, 155000, 160000, 165000],
                    "10_ckpts": [15000, 30000, 45000, 60000, 75000, 90000,
                                 105000, 120000, 135000, 150000, 165000],
                    "da_ckpts": [15000, 30000, 45000, 60000, 75000, 90000,
                                 105000, 120000, 135000, 150000, 165000],
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
        # stage2 + stage3 finals only — no intermediate checkpoints published.
        "midtraining": {
            "tokens": 9_900_000_000_000,
            "final": "stage3-step-4720000",
            "all": ["stage2-step-4200000", "stage3-step-4720000"],
        },
    },
}


def hf_entry(name, hf_url, size, params, family, phase,
             source="huggingface-reference", main_tokens=None):
    """Build one hf_branch model entry. Models in HF_CKPTS get their
    multi-stage checkpoint lists; everything else is a single-stage
    `phase` model published only at `main` (with `main_tokens` if known)."""
    if name in HF_CKPTS:
        stages = {ph: _hf_stage(cd["tokens"], cd)
                  for ph, cd in HF_CKPTS[name].items()}
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
        ("Qwen3-4B-Base", "https://huggingface.co/Qwen/Qwen3.5-4B-Base",
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
        ("SmolLM3-3B-Base", "https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base",
         "3B", 3_000_000_000, "SmolLM3-Base", "pretraining",
         "huggingface-reference", 11_200_000_000_000),
        ("SmolLM3-3B-checkpoints",
         "https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints",
         "3B", 3_000_000_000, "SmolLM3-checkpoints", "pretraining"),
        ("Olmo-3-1025-7B", "https://huggingface.co/allenai/Olmo-3-1025-7B",
         "7B", 7_000_000_000, "Olmo-3-1025", "pretraining"),

        # ---------- HF reference: posttraining ----------
        ("Qwen3-0.6B", "https://huggingface.co/Qwen/Qwen3-0.6B",
         "0.6B", 600_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-1.7B", "https://huggingface.co/Qwen/Qwen3-1.7B",
         "1.7B", 1_700_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-4B", "https://huggingface.co/Qwen/Qwen3.5-4B",
         "4B", 4_000_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-8B", "https://huggingface.co/Qwen/Qwen3-8B",
         "8B", 8_000_000_000, "Qwen3-it", "posttraining"),
        ("Qwen3-14B", "https://huggingface.co/Qwen/Qwen3-14B",
         "14B", 14_000_000_000, "Qwen3-it", "posttraining"),
        ("gemma-3-1b-it", "https://huggingface.co/google/gemma-3-1b-it",
         "1B", 1_000_000_000, "gemma-3-it", "posttraining"),
        ("gemma-3-4b-it", "https://huggingface.co/google/gemma-3-4b-it",
         "4B", 4_000_000_000, "gemma-3-it", "posttraining"),
        ("gemma-3-12b-it", "https://huggingface.co/google/gemma-3-12b-it",
         "12B", 12_000_000_000, "gemma-3-it", "posttraining"),
        ("gemma-3-27b-it", "https://huggingface.co/google/gemma-3-27b-it",
         "27B", 27_000_000_000, "gemma-3-it", "posttraining"),
        ("SmolLM3-3B", "https://huggingface.co/HuggingFaceTB/SmolLM3-3B",
         "3B", 3_000_000_000, "SmolLM3-it", "posttraining"),
        ("Apertus-1.7B-it800000-SFT",
         "https://huggingface.co/daslab-testing/Apertus-1.7B-it800000-SFT",
         "1.7B", 1_700_000_000, "Apertus-it-SFT-1", "posttraining",
         "daslab-testing"),
        ("Olmo-3-7B-Instruct-SFT",
         "https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT",
         "7B", 7_000_000_000, "Olmo-3-Instruct-SFT", "posttraining"),
        ("Olmo-3-7B-Instruct-DPO",
         "https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO",
         "7B", 7_000_000_000, "Olmo-3-Instruct-DPO", "posttraining"),
        ("Olmo-3-7B-Instruct",
         "https://huggingface.co/allenai/Olmo-3-7B-Instruct",
         "7B", 7_000_000_000, "Olmo-3-Instruct", "posttraining"),
        ("Apertus-8B-Instruct-2509",
         "https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509",
         "8B", 8_000_000_000, "Apertus-Instruct-2509", "posttraining",
         "swiss-ai-reference"),
        ("Apertus-70B-Instruct-2509",
         "https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509",
         "70B", 70_000_000_000, "Apertus-Instruct-2509", "posttraining",
         "swiss-ai-reference"),
    ]
    out = {}
    for row in rows:
        name, url, size, params, family, phase = row[:6]
        source = row[6] if len(row) > 6 else "huggingface-reference"
        main_tokens = row[7] if len(row) > 7 else None
        out[name] = hf_entry(name, url, size, params, family, phase,
                             source=source, main_tokens=main_tokens)
    return out


# --- Posttraining Megatron (distillation) -----------------------------------

DISTILL_BASE = "/capstor/store/cscs/swissai/infra01/distillation/checkpoints/distill"


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
                "megatron": f"{DISTILL_BASE}/{dir_name}/checkpoints/"
            },
            "stages": {
                "posttraining": _meg_stage({"final": None, "all": []}),
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
    },
    "pretraining_a06": {
        "description": "Apertus3 a06 main-run checkpoints (1B and 3B).",
        "stage": "pretraining",
        "members": [{"source": "snr-pretraining-a06"}],
        "include_external": False,
    },
    "pretraining_hf_reference": {
        "description": "All HF / Swiss-AI reference pretrains.",
        "stage": "pretraining",
        "members": [
            {"source": "huggingface-reference"},
            {"source": "swiss-ai-reference"},
        ],
    },
}


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
    "daslab-testing":         {"split": None},
    "distillation":           {"split": None},
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
            "smollm3-3b-base": "SmolLM3-3B-Base",
        },
    },
}


# --- Tasks ------------------------------------------------------------------

# Language token → ISO code mapping mirrors multilingual/analyze_snr_variants.py
LANG_MAP = {
    "ar": "ar", "arb": "ar", "de": "de", "es": "es", "spa": "es",
    "eu": "eu", "eus": "eu", "fr": "fr", "hi": "hi", "hin": "hi",
    "ru": "ru", "rus": "ru", "vi": "vi", "vie": "vi",
    "zh": "zh", "zho": "zh", "cmn": "zh", "ja": "ja", "jp": "ja", "jpn": "ja",
    "sw": "sw", "swh": "sw", "th": "th", "tha": "th",
    "tr": "tr", "tur": "tr", "en": "en", "eng": "en", "te": "te",
}


def _benchmark_of(task: str) -> str:
    """Strip language/script suffix to get the benchmark family name."""
    if task in {"arc_challenge", "arc_easy"}:
        return "arc"
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
        "pretraining_full":     EVALS_CONFIGS / "tasks_pretraining_full.txt",
        "pretraining_classic":  EVALS_CONFIGS / "tasks_pretraining_classic.txt",
        "pretraining_extra":    EVALS_CONFIGS / "tasks_pretraining_extra.txt",
        "pretraining_random":   EVALS_CONFIGS / "tasks_pretraining_random.txt",
        "pretraining_allenai":  EVALS_CONFIGS / "tasks_pretraining_allenai.txt",
        "midtraining":          EVALS_CONFIGS / "tasks_midtraining.txt",
        "posttraining":         EVALS_CONFIGS / "tasks_posttraining.txt",
        "include_base_44":      EVALS_CONFIGS / "tasks_include.txt",
        "test":                 EVALS_CONFIGS / "tasks_test.txt",
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
        # added tasks.
        lang = existing_lang.get(t) or _language_of(t)
        tasks_section[t] = {
            "language": lang,
            "benchmark": _benchmark_of(t),
            "stages": stages,
        }
    return {"tasks": tasks_section, "groups": groups}


# --- Driver -----------------------------------------------------------------

def main():
    models = {}
    models.update(custom_pretrain_entries())
    models.update(a06_pretrain_entries())
    models.update(hf_entries())
    models.update(distill_entries())
    models.update(tokenizer_lm_entries())

    models_json = {"models": models, "pools": POOLS,
                   "sources": SOURCES, "snr": SNR}
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / "models.json").write_text(
        json.dumps(models_json, indent=2) + "\n"
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
