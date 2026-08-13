#!/usr/bin/env python3
"""
Drive create_data_mixture.py over the full small-to-large predictivity sweep.

The experiment builds, once each:
  - one fixed validation set (all 199 FineWeb-2 languages + English),
  - one English (DCLM) training dataset,
  - one FineWeb-2 training dataset per multilingual language setting,
and then blends English + FineWeb-2 50/50 at *training* time with the Megatron
data loader's blend weights (the 1-language setting is 100% English). See
.claude-shared/plans/small-to-large-predictivity-training-plan.md.

This wrapper turns the FW_Lx language lists in language_sets_scheme{A,B}.json
into `--languages` arguments and shells out to create_data_mixture.py with the
right per-build token target, mirroring the Commands section of the plan. It
adds no tokenization logic of its own; create_data_mixture.py owns that.

Build-token sizing:
  A built dataset must be large enough for the *largest* sample any run draws
  from it. The largest run at a setting is its biggest model's full token
  budget D(N) = 5 x Chinchilla, and at training time the multilingual half is
  50% of that. Hence:
    fineweb_target(L) = largest_budget(L) * ML_SHARE * (1 + HEADROOM)
    english_target    = max_english_need  * (1 + HEADROOM)
  where largest_budget(L) is 170B where the 1.7B model trains (L in 1,8,30,
  100,200) and 100B otherwise, ML_SHARE = 0.50, and the English need is
  bounded by the 1-language setting (100% English at 170B). This reproduces
  the plan's 55B / 93.5B FineWeb-2 builds and 187B English build.

Usage:
  # Everything for scheme A into ./outputs (validation, English, all settings)
  python build_data_mixtures.py --scheme A --output_dir outputs

  # Just print the create_data_mixture.py commands without running them
  python build_data_mixtures.py --scheme A --output_dir outputs --dry_run

  # Only the FineWeb-2 builds for two settings, scheme B
  python build_data_mixtures.py --scheme B --output_dir outputs \
      --stage fineweb --settings 8,30
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CREATE_SCRIPT = SCRIPT_DIR / "create_data_mixture.py"

# Per-size token budget D(N) = 5 x Chinchilla = 100 x N (non-embedding N),
# in billions, keyed by the nominal size label. Used only to size builds, so
# the nominal (rounded) value with headroom is deliberate — see module docstring.
SIZE_BUDGET_B = {"90M": 9, "175M": 17.5, "350M": 35, "600M": 60, "1B": 100, "1.7B": 170}

# The largest model trained at each language setting (from the Models table in
# the plan): the 1.7B reference exists only at L in {1, 8, 30, 100, 200}; the
# 1B model is the largest everywhere else. This sets each setting's largest
# single-run token budget.
LARGEST_SIZE_PER_SETTING = {
    1: "1.7B", 2: "1B", 8: "1.7B", 15: "1B",
    30: "1.7B", 50: "1B", 100: "1.7B", 200: "1.7B",
}

# Multilingual fraction of a run at training time (fixed 50/50 English /
# FineWeb-2). The FineWeb-2 builds are sized to this share of the largest run.
ML_SHARE = 0.50
HEADROOM = 0.10  # extra build margin so a build is never the binding constraint

VALIDATION_PREFIX = "validation"
ENGLISH_PREFIX = "english_dclm"
FINEWEB_PREFIX_FMT = "fineweb_L{L}"  # e.g. fineweb_L8


def round_to_b(tokens_b: float) -> int:
    """Round a billions-of-tokens target up to the nearest 0.5B, in absolute
    tokens. Half-billion granularity keeps the printed plan readable while
    never sizing a build below its computed need. The small epsilon absorbs
    float error so a target already on a 0.5B boundary (e.g. 93.5B) is not
    bumped to the next half-billion."""
    half_billions = math.ceil(tokens_b * 2 - 1e-6)
    return half_billions * 500_000_000


def english_target_tokens() -> int:
    """Token target for the single English (DCLM) build.

    Bounded by the 1-language setting, which is 100% English at the largest
    budget that trains there (170B, the 1.7B model), plus headroom.
    """
    largest_b = SIZE_BUDGET_B[LARGEST_SIZE_PER_SETTING[1]]
    return round_to_b(largest_b * (1 + HEADROOM))


def fineweb_target_tokens(setting: int) -> int:
    """Token target for one setting's FineWeb-2 build.

    Sized to the largest multilingual draw at that setting: the biggest model's
    budget times the fixed multilingual share (50%), plus headroom.
    """
    largest_b = SIZE_BUDGET_B[LARGEST_SIZE_PER_SETTING[setting]]
    return round_to_b(largest_b * ML_SHARE * (1 + HEADROOM))


def load_scheme(scheme: str) -> dict:
    """Return the {FW_Lx: [lang_script, ...]} sets for scheme A or B."""
    path = SCRIPT_DIR / f"language_sets_scheme{scheme}.json"
    return json.loads(path.read_text())["sets"]


def fineweb_languages(sets: dict, setting: int) -> str:
    """Comma-separated `{lang}_{script}` list for an L-setting (no spaces).

    The FineWeb-2 list for an L-setting has L-1 entries (English is the DCLM
    half and is supplied separately), so it is keyed FW_L{L} in the JSON.
    """
    return ",".join(sets[f"FW_L{setting}"])


def run(cmd: list, dry_run: bool) -> None:
    """Print a create_data_mixture.py command and (unless dry_run) execute it,
    aborting the whole sweep on the first non-zero exit."""
    printable = " \\\n    ".join(cmd)
    print(f"\n$ {printable}\n")
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"create_data_mixture.py failed (exit {result.returncode}); stopping.")


def build_validation(out: Path, all_langs: str, args) -> None:
    """Step 1: the fixed validation set, once, over every FineWeb-2 language
    (English is added automatically by create_data_mixture.py)."""
    run([
        sys.executable, str(CREATE_SCRIPT),
        "--build_validation",
        "--languages", all_langs,
        "--val_tokens_per_language", str(args.val_tokens_per_language),
        "--val_max_fraction", str(args.val_max_fraction),
        "--output_prefix", str(out / VALIDATION_PREFIX),
    ], args.dry_run)


def build_english(out: Path, manifest: Path, args) -> None:
    """Step 2: the single English (DCLM) dataset, validation rows excluded."""
    run([
        sys.executable, str(CREATE_SCRIPT),
        "--target_tokens", str(english_target_tokens()),
        "--fineweb_pct", "0", "--dclm_pct", "100",
        "--validation_manifest", str(manifest),
        "--output_prefix", str(out / ENGLISH_PREFIX),
    ], args.dry_run)


def build_fineweb(out: Path, manifest: Path, sets: dict, setting: int, args) -> None:
    """Step 3: one setting's FineWeb-2 dataset (T=1 allocation), validation
    rows excluded. The 1-language setting has no FineWeb-2 build."""
    if setting == 1:
        print("\n[L=1] English-only setting — no FineWeb-2 build; trains on the English dataset alone.")
        return
    prefix = FINEWEB_PREFIX_FMT.format(L=setting)
    run([
        sys.executable, str(CREATE_SCRIPT),
        "--target_tokens", str(fineweb_target_tokens(setting)),
        "--fineweb_pct", "100", "--dclm_pct", "0",
        "--languages", fineweb_languages(sets, setting),
        "--temperature", str(args.temperature),
        "--validation_manifest", str(manifest),
        "--output_prefix", str(out / prefix),
    ], args.dry_run)


def print_build_plan(settings: list) -> None:
    """Show the token target every build will use, before running anything."""
    print("=" * 64)
    print("Build plan (token targets)")
    print("=" * 64)
    print(f"  validation : per-language budget (see --val_tokens_per_language)")
    print(f"  {ENGLISH_PREFIX:14s}: {english_target_tokens()/1e9:6.1f}B  (100% EN, bounds the L=1 run)")
    for L in settings:
        if L == 1:
            continue
        largest = LARGEST_SIZE_PER_SETTING[L]
        tgt = fineweb_target_tokens(L)
        print(f"  fineweb_L{L:<5d}: {tgt/1e9:6.1f}B  "
              f"(largest run {largest} @ {SIZE_BUDGET_B[largest]}B x "
              f"{ML_SHARE:.0%} ml x {1+HEADROOM:.0%})")
    print()


def main():
    all_settings = sorted(LARGEST_SIZE_PER_SETTING)  # 1,2,8,15,30,50,100,200

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scheme", choices=["A", "B"], required=True,
        help="Language-set scheme: A (resource-ranked) or B (diversity-first). "
             "Picks language_sets_scheme{A,B}.json.",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory for the .bin/.idx artifacts and the validation manifest.",
    )
    parser.add_argument(
        "--stage", choices=["all", "validation", "english", "fineweb"],
        default="all",
        help="Which artifacts to build (default: all). The english and fineweb "
             "stages require the validation manifest to already exist.",
    )
    parser.add_argument(
        "--settings", type=str, default=None,
        help="Comma-separated subset of language settings to build FineWeb-2 "
             f"for (default: all of {all_settings}).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Per-language allocation temperature within FineWeb-2 (default 1.0, "
             "proportional to estimated tokens). Passed through to create_data_mixture.py.",
    )
    parser.add_argument(
        "--val_tokens_per_language", type=int, default=5_000_000,
        help="Validation tokens per language (default 5,000,000).",
    )
    parser.add_argument(
        "--val_max_fraction", type=float, default=0.3,
        help="Cap validation at this fraction of each language's first file (default 0.3).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the create_data_mixture.py commands without running them.",
    )
    args = parser.parse_args()

    settings = (
        [int(s) for s in args.settings.split(",")] if args.settings else all_settings
    )
    bad = [s for s in settings if s not in all_settings]
    if bad:
        sys.exit(f"Unknown settings {bad}; valid: {all_settings}")

    sets = load_scheme(args.scheme)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"{VALIDATION_PREFIX}.manifest.json"
    all_langs = fineweb_languages(sets, max(all_settings))  # FW_L200: all 199

    print(f"Scheme {args.scheme} | output_dir {out} | stage {args.stage}")
    print_build_plan(settings)

    if args.stage in ("all", "validation"):
        build_validation(out, all_langs, args)
    if args.stage in ("all", "english"):
        build_english(out, manifest, args)
    if args.stage in ("all", "fineweb"):
        for L in settings:
            build_fineweb(out, manifest, sets, L, args)


if __name__ == "__main__":
    main()
