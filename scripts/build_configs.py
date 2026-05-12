"""One-time bootstrap: build configs/models.json and configs/tasks.json
from the existing models_*.txt / tasks_*.txt files in
src/evals/configs/signal_to_ratio/.

After this runs, the JSONs are the source of truth — this script is
kept for traceability but not re-run automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
EVALS_CONFIGS = REPO / "src" / "evals" / "configs" / "signal_to_ratio"

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
# Derive `full_eval` from dense_tail ∪ da_ckpts so it can't drift.
for _seed, _sched in CKPT_SCHEDULES.items():
    _sched["full_eval"] = sorted(set(_sched["dense_tail"])
                                  | set(_sched["da_ckpts"]))

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
                    "stage": "pretraining",
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
                    "checkpoints": CKPT_SCHEDULES[seed],
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
            "stage": "pretraining",
            "size": "1B",
            "params": 1_000_000_000,
            "checkpoint_kind": "megatron_iter",
            "backends": {
                "megatron": f"{A06_BASE}/apertus3-1b-21-nodes/checkpoints/"
            },
            "checkpoints": {
                "final": 380000,
                "all": a06_1b_iters,
                "dense_tail": [320000, 340000, 360000, 370000, 380000],
                "10_ckpts": [20000, 60000, 100000, 140000, 180000,
                             220000, 260000, 300000, 340000, 380000],
                "da_ckpts": [20000, 60000, 100000, 140000, 180000,
                             220000, 260000, 300000, 340000, 380000],
            },
        },
        "apertus3-3b-64-nodes": {
            "source": "snr-pretraining-a06",
            "family": "apertus3-a06",
            "stage": "pretraining",
            "size": "3B",
            "params": 3_000_000_000,
            "checkpoint_kind": "megatron_iter",
            "backends": {
                "megatron": f"{A06_BASE}/apertus3-3b-64-nodes/checkpoints/"
            },
            "checkpoints": {
                "final": 165000,
                "all": a06_3b_iters,
                "dense_tail": [135000, 150000, 155000, 160000, 165000],
                "10_ckpts": [15000, 30000, 45000, 60000, 75000, 90000,
                             105000, 120000, 135000, 150000, 165000],
                "da_ckpts": [15000, 30000, 45000, 60000, 75000, 90000,
                             105000, 120000, 135000, 150000, 165000],
            },
        },
    }


# --- HF reference (pretraining + midtraining + posttraining) ----------------

def hf_entry(name, hf_url, size, params, family, stage,
             checkpoints=None, source="huggingface-reference"):
    if checkpoints is None:
        checkpoints = {"all": [{"branch": "main", "tokens": None}]}
    return {
        "source": source,
        "family": family,
        "stage": stage,
        "size": size,
        "params": params,
        "checkpoint_kind": "hf_branch",
        "backends": {"hf": hf_url},
        "checkpoints": checkpoints,
    }


def hf_entries() -> dict:
    # (model_key, hf_url, size, params, family, stage, source[, checkpoints])
    rows = [
        # ---------- Swiss-AI reference: pretraining ----------
        ("Apertus-8B-2509", "https://huggingface.co/swiss-ai/Apertus-8B-2509",
         "8B", 8_000_000_000, "Apertus-2509", "pretraining", "swiss-ai-reference",
         {"all": [{"branch": "main", "tokens": 15_000_000_000_000}]}),
        ("Apertus-70B-2509", "https://huggingface.co/swiss-ai/Apertus-70B-2509",
         "70B", 70_000_000_000, "Apertus-2509", "pretraining", "swiss-ai-reference",
         {"all": [{"branch": "main", "tokens": 15_000_000_000_000}]}),

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
         "3B", 3_000_000_000, "SmolLM3-Base", "pretraining"),
        ("SmolLM3-3B-checkpoints",
         "https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints",
         "3B", 3_000_000_000, "SmolLM3-checkpoints", "pretraining",
         "huggingface-reference",
         {"all": [
             {"branch": "stage1-step-3440000", "tokens": 7_200_000_000_000},
             {"branch": "stage2-step-4200000", "tokens": 8_800_000_000_000},
             {"branch": "stage3-step-4720000", "tokens": 9_900_000_000_000},
         ]}),

        # ---------- HF reference: midtraining ----------
        ("Olmo-3-1025-7B", "https://huggingface.co/allenai/Olmo-3-1025-7B",
         "7B", 7_000_000_000, "Olmo-3-1025", "midtraining"),

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
        if len(row) == 6:
            name, url, size, params, family, stage = row
            out[name] = hf_entry(name, url, size, params, family, stage)
        elif len(row) == 7:
            name, url, size, params, family, stage, source = row
            out[name] = hf_entry(name, url, size, params, family, stage,
                                 source=source)
        elif len(row) == 8:
            name, url, size, params, family, stage, source, ckpts = row
            out[name] = hf_entry(name, url, size, params, family, stage,
                                 checkpoints=ckpts, source=source)
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
            "stage": "posttraining",
            "size": size,
            "params": params,
            "checkpoint_kind": "megatron_iter",
            "backends": {
                "megatron": f"{DISTILL_BASE}/{dir_name}/checkpoints/"
            },
            "checkpoints": {"final": None, "all": []},
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
            "stage": "pretraining",
            "size": "TBD",
            "params": None,
            "checkpoint_kind": "hf_local",
            "backends": {"hf_local": f"{TOKLM_BASE}/{r}-128k-apertus"},
            "checkpoints": {"all": [{"branch": "main", "tokens": None}]},
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

    models_json = {"models": models, "pools": POOLS}
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / "models.json").write_text(
        json.dumps(models_json, indent=2) + "\n"
    )
    print(f"Wrote configs/models.json — {len(models)} models, "
          f"{len(POOLS)} pools")

    tasks_json = build_tasks_json()
    (CONFIGS / "tasks.json").write_text(
        json.dumps(tasks_json, indent=2) + "\n"
    )
    print(f"Wrote configs/tasks.json — {len(tasks_json['tasks'])} tasks, "
          f"{len(tasks_json['groups'])} groups")


if __name__ == "__main__":
    main()
