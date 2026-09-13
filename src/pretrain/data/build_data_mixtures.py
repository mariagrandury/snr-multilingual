#!/usr/bin/env python3
"""
Drive create_data_mixture.py over the full small-to-large predictivity sweep.

The experiment builds, once each:
  - one fixed validation set (the FW_L100 languages + English),
  - one English (DCLM) training dataset,
  - one FineWeb-2 training dataset per multilingual language setting,
and then blends English + FineWeb-2 50/50 at *training* time with the Megatron
data loader's blend weights (the 1-language setting is 100% English). See
plan/small-to-large-predictivity-training-plan.md.

This wrapper turns the FW_Lx language lists in language_sets_scheme{A,B,ZH,ES}
.json into `--languages` arguments and shells out to create_data_mixture.py
with the right per-build token target, mirroring the Commands section of the
plan. It adds no tokenization logic of its own; create_data_mixture.py owns
that. Which lists, which settings and which sampling temperature a build uses
all come from the scheme's entry in launch_trainings.DATA_SCHEMES — there is no
`--temperature` flag, because a temperature that disagrees with the grid would
silently produce a dataset no cell can use.

Build-token sizing:
  A built dataset must be large enough for the *largest* sample any run draws
  from it. The largest run at a setting is its biggest model's full token
  budget D(N) = 5 x Chinchilla, and at training time the multilingual half is
  50% of that. Hence:
    fineweb_target(L) = largest_budget(L, scheme) * ML_SHARE * (1 + HEADROOM)
    english_target    = max_english_need          * (1 + HEADROOM)
  where largest_budget comes from the grid (largest_size() -> scheme_sizes()),
  ML_SHARE = 0.50, and the English need is bounded by the 1-language setting
  (100% English at ~167B). With the reviewed deep-ladder budgets (read from
  hyperparams_deep.json) this yields a ~184B English build, ~92B FineWeb-2
  builds wherever the 1.7B rung trains — which since 2026-09-10 is every
  scheme-A setting and both AT3 settings — and ~52B for the schemes capped at
  the 1B rung (ZH and ES at L2).

  The builder NEVER repeats data: a source that runs out before its target
  prints a shortfall and processing continues. So a target the corpus cannot
  reach yields a SMALLER build, not a flatter one — which is why the realized
  size, not the target, is what has to cover the largest run.

Usage:
  # Everything for scheme A into ./outputs (validation, English, all settings)
  python build_data_mixtures.py --scheme A --output_dir outputs

  # Just print the create_data_mixture.py commands without running them
  python build_data_mixtures.py --scheme A --output_dir outputs --dry_run

  # Only the FineWeb-2 builds for two settings, scheme B (its own output_dir)
  python build_data_mixtures.py --scheme B --output_dir outputs/schemeB \
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

# The grid owns the data schemes — which settings each one builds, at what
# temperature, and the largest rung that trains it. Importing them is what
# keeps a build's token target tied to the models that will read it: when the
# 1.7B row gained L15 and L50 (2026-09-10) those two builds had to grow from
# 52B to 92B, and a hardcoded table here would not have noticed.
sys.path.insert(0, str(SCRIPT_DIR.parent))
from launch_trainings import DATA_SCHEMES, scheme_sizes  # noqa: E402

# Per-size token budget D(N) = 5 x Chinchilla = 100 x N (non-embedding N), in
# billions, read from the reviewed deep ladder's per-size predictivity schedule
# (hyperparams_deep.json `predictivity.train_tokens`). The shallow variants
# match within a few %, so the same builds cover both arch families.
def _load_size_budget_b() -> dict:
    configs = json.loads((SCRIPT_DIR.parent / "hyperparams" / "hyperparams_deep.json").read_text())["configs"]
    return {size: c["predictivity"]["train_tokens"] / 1e9 for size, c in configs.items()}


SIZE_BUDGET_B = _load_size_budget_b()

def largest_size(setting: int, scheme: str = "A") -> str:
    """The biggest model that trains this scheme at this setting — the run
    that sizes the build. Read from the grid rather than tabulated here, so a
    rung added to a setting grows its build instead of leaving the reference
    model to loop over a dataset smaller than its own budget."""
    return scheme_sizes(scheme, setting)[-1]

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
    return round_to_b(SIZE_BUDGET_B[largest_size(1)] * (1 + HEADROOM))


def fineweb_target_tokens(setting: int, scheme: str = "A") -> int:
    """Token target for one setting's FineWeb-2 build.

    Sized to the largest multilingual draw at that setting: the biggest model's
    budget times the fixed multilingual share (50%), plus headroom.
    """
    largest_b = SIZE_BUDGET_B[largest_size(setting, scheme)]
    return round_to_b(largest_b * ML_SHARE * (1 + HEADROOM))


def load_scheme(scheme: str) -> dict:
    """Return the {FW_Lx: [lang_script, ...]} sets a scheme builds from."""
    path = SCRIPT_DIR / f"language_sets_scheme{DATA_SCHEMES[scheme]['sets']}.json"
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


def already_built(prefix: Path) -> bool:
    """True once a create_data_mixture build has finished for this prefix.

    create_data_mixture writes the .idx only in finalize() and removes its
    .checkpoint.json only after that. So `.idx present and no .checkpoint.json`
    marks a complete build; a partial (preempted) one still has the checkpoint.
    This lets the sweep be resubmitted past the 12h wall: finished builds are
    skipped instead of rebuilt from scratch (which would overwrite them).
    """
    return Path(f"{prefix}.idx").exists() and not Path(f"{prefix}.checkpoint.json").exists()


def build_validation(out: Path, all_langs: str, args) -> None:
    """Step 1: the fixed validation set, once, over every FineWeb-2 language
    (English is added automatically by create_data_mixture.py)."""
    manifest = out / f"{VALIDATION_PREFIX}.manifest.json"
    if manifest.exists():
        # The language lists are generated (generate_language_sets.py) and can
        # gain languages; a manifest that lacks any of them must be rebuilt —
        # the per-language carve-out is deterministic (first file, leading
        # rows), so existing languages' val_doc_count and the training builds
        # that skip them are unchanged by the rebuild.
        have = set(json.loads(manifest.read_text()))
        missing = [l for l in all_langs.split(",") if f"fineweb_{l}" not in have]
        if not missing:
            print(f"\n[validation] manifest present ({manifest}) — skipping.")
            return
        print(f"\n[validation] manifest lacks {missing} — rebuilding over the full list.")
    if manifest.is_symlink():
        # A scheme dir links the ONE shared manifest; rebuilding through the
        # link would overwrite it with bin paths inside this scheme dir.
        sys.exit(f"[validation] {manifest} links the shared manifest; rebuild it "
                 f"in the data root it points to (--scheme A --output_dir <root> "
                 f"--stage validation), not in a scheme dir.")
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
    prefix = out / ENGLISH_PREFIX
    if already_built(prefix):
        print(f"\n[{ENGLISH_PREFIX}] already built ({prefix}.idx present) — skipping.")
        return
    run([
        sys.executable, str(CREATE_SCRIPT),
        "--target_tokens", str(english_target_tokens()),
        "--fineweb_pct", "0", "--dclm_pct", "100",
        "--validation_manifest", str(manifest),
        "--output_prefix", str(prefix),
    ], args.dry_run)


def build_fineweb(out: Path, manifest: Path, sets: dict, setting: int, args) -> None:
    """Step 3: one setting's FineWeb-2 dataset at its scheme's allocation
    temperature, validation rows excluded. The 1-language setting has no
    FineWeb-2 build."""
    if setting == 1:
        print("\n[L=1] English-only setting — no FineWeb-2 build; trains on the English dataset alone.")
        return
    temp = DATA_SCHEMES[args.scheme]["temp"]
    prefix = FINEWEB_PREFIX_FMT.format(L=setting)
    langs = fineweb_languages(sets, setting)
    # The language lists are generated and can change (e.g. the 2026-08-21
    # L100 swap); a finished build records its list in a sidecar so a build
    # made from an older list is refused instead of silently reused. Line 2
    # records the temperature: AT3 has byte-identical language lists to A, so
    # the lists alone cannot tell the two apart, and a sidecar written before
    # schemes existed simply has no second line (checked, not assumed).
    sidecar = out / f"{prefix}.languages"
    if already_built(out / prefix):
        if sidecar.exists():
            lines = sidecar.read_text().splitlines()
            if lines and lines[0].strip() != langs:
                sys.exit(f"[{prefix}] built from a DIFFERENT language list ({sidecar}); "
                         f"move the old build aside before rebuilding.")
            # No T= line means the sidecar predates schemes, when T=1 was the
            # only temperature there was.
            built_t = next((l.split("=", 1)[1] for l in lines[1:]
                            if l.startswith("T=")), "1.0")
            if float(built_t or "nan") != temp:
                sys.exit(f"[{prefix}] built at T={built_t}, this scheme wants "
                         f"T={temp} ({sidecar}); build it in its own "
                         f"--output_dir instead of overwriting.")
        elif temp != 1.0:
            # Every unrecorded build predates schemes, so every one is T=1:
            # accepting it here would train a flattened scheme on the T=1 mix.
            sys.exit(f"[{prefix}] finished build with no record of its "
                     f"temperature; scheme {args.scheme} wants T={temp:g} and "
                     f"unrecorded builds are T=1 — build it in its own --output_dir.")
        else:
            print(f"\n[{prefix}] WARNING: no language record next to the build — "
                  f"cannot verify it matches the current list.")
        print(f"\n[{prefix}] already built ({out / prefix}.idx present) — skipping.")
        return
    # Every language must already have its validation rows carved out, or the
    # build trains on exactly the rows its BPB is later measured on (the
    # leading rows of its first file), and extending validation afterwards
    # cannot undo that. The shared manifest predates the 2026-08-21 L100 swap
    # and lacks 8 of the L100 languages.
    if manifest.is_file():
        have = set(json.loads(manifest.read_text()))
        absent = [l for l in langs.split(",") if f"fineweb_{l}" not in have]
        if absent:
            sys.exit(f"[{prefix}] validation manifest {manifest} has no rows for "
                     f"{absent}; extend it first, in the data root: "
                     f"build_data_mixtures.py --scheme A --output_dir <root> "
                     f"--stage validation")
    run([
        sys.executable, str(CREATE_SCRIPT),
        "--target_tokens", str(fineweb_target_tokens(setting, args.scheme)),
        "--fineweb_pct", "100", "--dclm_pct", "0",
        "--languages", langs,
        "--temperature", str(temp),
        "--validation_manifest", str(manifest),
        "--output_prefix", str(out / prefix),
    ], args.dry_run)
    if not args.dry_run:
        sidecar.write_text(f"{langs}\nT={temp}\n")


def print_build_plan(settings: list, scheme: str = "A") -> None:
    """Show the token target every build will use, before running anything."""
    temp = DATA_SCHEMES[scheme]["temp"]
    print("=" * 64)
    print(f"Build plan (token targets) — scheme {scheme}, T={temp:g}")
    print("=" * 64)
    print(f"  validation : per-language budget (see --val_tokens_per_language)")
    print(f"  {ENGLISH_PREFIX:14s}: {english_target_tokens()/1e9:6.1f}B  (100% EN, bounds the L=1 run)")
    for L in settings:
        if L == 1:
            continue
        largest = largest_size(L, scheme)
        tgt = fineweb_target_tokens(L, scheme)
        print(f"  fineweb_L{L:<5d}: {tgt/1e9:6.1f}B  "
              f"(largest run {largest} @ {SIZE_BUDGET_B[largest]}B x "
              f"{ML_SHARE:.0%} ml x {1+HEADROOM:.0%})")
    print()


def main():
    # Which settings exist at all, across every scheme — the per-scheme
    # subset is resolved once --scheme is known.
    all_settings = sorted({L for v in DATA_SCHEMES.values() for L in v["langs"]})

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scheme", choices=list(DATA_SCHEMES), required=True,
        help="Data scheme to build (launch_trainings.DATA_SCHEMES): its "
             "language lists, its allocation temperature and the settings it "
             "covers all come from there. Give each scheme its own "
             "--output_dir; they share the english build and the validation "
             "manifest by symlink (data/launch_builds.sh).",
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
             "for (default: every setting the --scheme defines).",
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

    # A scheme only builds the settings it defines: AT3 is L50 and L100, ZH
    # and ES are L2 alone. Asking for one it does not define is a mistake worth
    # reporting — silently building it would put a file in the scheme's
    # directory that no cell will ever read.
    mine = sorted(DATA_SCHEMES[args.scheme]["langs"])
    settings = (
        [int(s) for s in args.settings.split(",")] if args.settings else mine
    )
    bad = [s for s in settings if s not in mine]
    if bad:
        sys.exit(f"Scheme {args.scheme} does not define settings {bad}; "
                 f"it covers {mine}")

    sets = load_scheme(args.scheme)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"{VALIDATION_PREFIX}.manifest.json"
    # Validation covers the largest trained setting's languages (FW_L100 + EN)
    # — the 200-language list was dropped along with the training setting. It
    # is scheme A's L100 list whatever scheme is being built: one validation
    # set is shared by every model, which is what makes their BPB comparable.
    all_langs = fineweb_languages(load_scheme("A"), max(all_settings))

    print(f"Scheme {args.scheme} | output_dir {out} | stage {args.stage}")
    print_build_plan(settings, args.scheme)

    if args.stage in ("all", "validation"):
        build_validation(out, all_langs, args)
    if args.stage in ("all", "english"):
        build_english(out, manifest, args)
    if args.stage in ("all", "fineweb"):
        for L in settings:
            build_fineweb(out, manifest, sets, L, args)


if __name__ == "__main__":
    main()
