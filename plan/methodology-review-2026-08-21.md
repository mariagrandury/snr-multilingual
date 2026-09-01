# Methodology review — predictivity sweep (2026-08-21)

Scope: `plan/` (training plan, compute budget, model sheet) and `src/pretrain/`
(hyperparams, language sets, data builds, launchers, auto-eval chain), plus the
benchmark configuration they drive (`configs/tasks.json`, `configs/languages.json`).
Findings ordered by severity. "FIXED" = implemented; "DECISION" = needs your
call before anyone implements it. Status 2026-08-21: findings 1, 2, 6, 7 and 8
are committed (79102bd, 9b19801, 8be0eac, 2f69d74, d58394a — the sweep is
being retrained from scratch on their account), as is the rename of every
model/run to `lm-<size>-L<L>[-schemeB]-<deep|shallow>-seed<seed>` with
`pretrain-`/`eval-`/`convert-` job prefixes (9df9cf8); findings 3, 4, 5 and 9
are implemented but uncommitted, pending review.

Verification notes: every architecture's non-embedding parameter count was
recomputed from shapes and matches the JSONs exactly (deep and shallow, all six
sizes); the predictivity schedules (iters, warmup ≈ 4 %, WSD decay ≈ 20 %) match
D = 100·N at GBS 504 × 4096; the scheme-A lists are exactly byte-ranked top-k
(see finding 5 for the two undocumented exclusions); the validation carve-out
logic in `create_data_mixture.py` (leading rows of each source's first file,
manifest-driven skip at train-build time) is correct and leak-free.

---

## 1. Azure evals and conversions use the wrong tokenizer — FIXED

`azure/eval.sh` defaults `TOKENIZER=alehc/swissai-tokenizer` and
`azure/convert.sh` defaults `HF_TOKENIZER=alehc/swissai-tokenizer` (both
inherited from the 36-model sweep). The predictivity models are trained with
`swiss-ai/Apertus-70B-2509` (`launch_trainings.TOKENIZER_MODEL`), and the two
tokenizers are different files on the Hub (tokenizer.json 17,078,368 vs
17,078,480 bytes; tokenizer_config.json differs too). The CSCS watcher
(`auto_evals_cscs.py`) correctly overrides both; the Azure watcher
(`auto_evals_azure.py`) overrides **neither**, so every Azure benchmark score
would be produced by tokenizing eval prompts with a tokenizer that doesn't
match the trained embedding table.

**Fix applied:** `auto_evals_azure.py` now passes
`environment_variables.TOKENIZER` (eval) and `environment_variables.HF_TOKENIZER`
(convert) = `TOKENIZER_MODEL`, mirroring the CSCS watcher. Any Azure eval
results produced before this fix should be discarded and re-run.

## 2. The checkpoint grid cannot support the plan's own analysis — FIXED

The plan requires (a) a checkpoint at the Chinchilla-optimal point 1×C = 20·N
(= `train_iters / 5`, since D = 5×C) for every size, and (b) a dense final
window for the checkpoint-noise estimate ("the last 30, spaced about 1000
steps"). The implementation saved every fixed 2000 iters, which gives
90M/175M/350M/600M only **3/5/9/15** checkpoints total, never saves the 1×C
iters (900/1700/3340/5760), and at 90M the nearest saved checkpoint (2000) is
2.2× the 1×C token count. Checkpoint noise over a "final window" is
meaningless with 3 points.

**Fix applied:** per-size `SAVE_INTERVAL = train_iters // n` with n = 20
checkpoints per run, 40 at the 1B and 60 at the 1.7B rung (team decision
2026-08-21: denser references; multiples of 20 keep every size on the shared
k/20 grid), set by `launch_trainings.cell_env` (both platforms) and
mirrored in `sync_models_json.save_points` and the CSCS watcher's due rule.
The generators round each schedule to the checkpoint grid (nearest multiple
of 20/40 instead of 100 — which is also slightly *closer* to the exact
D = 100·N targets), so the division is exact for every size, deep and
shallow: n evenly spaced checkpoints per run, checkpoint k at k/n of
training, the final checkpoint on-grid, and the 1×C point exactly checkpoint
n/5 (4, 8 or 12) everywhere — index-aligned operating points across
sizes for the SNR analysis (e.g. "50 % of training" = checkpoint 10 at every
size). Auto-evals ("every 2nd checkpoint") scale accordingly: ~10 aligned
eval points per run instead of 1–3 for the small rungs.

Caveat: cells already mid-training (the L2 launches of 2026-08-19) keep their
existing 2000-grid checkpoints; resumes will save on the new grid from the
next multiple onward. The due rule keys on the new grid, so a few early
2000-grid checkpoints of those cells simply won't be auto-evaluated (the
final checkpoint always is).

## 3. The documented grid disagrees with the implemented grid — FIXED (docs)

The code (after commit e8dea0e, 2026-08-19) trains **56 runs per level**:
×3 seeds at 175M/1B for L ∈ {1, **2**, 30, 100}, and 1.7B at
L ∈ {1, **2**, 8, 30, 100}. The documents still describe older grids:
`plan/models.md` says 51 runs with no 1.7B at L2 and ×3 only at {1, 30, 100};
`plan/compute-budget.md`'s headline says 153 runs = 51×3 (its own CSCS
appendix says 52); the launcher/sync docstrings said "52 jobs". The budget of
record is therefore stale by +2 × 175M, +2 × 1B, +1 × 1.7B per level
≈ +3.4e21 FLOPs/level (+17 %): ≈ 2.32e22 FLOPs/level, 6.95e22 total,
≈ 2,015 H100-days, fast-mix ≈ **$130k** (headroom ≈ $70k).

**Fix applied:** models.md grid/table/run counts, launcher + sync docstrings,
pretrain CLAUDE.md, plan run count and ×3 note, and a dated update note at the
top of compute-budget.md with the corrected totals (tables below it left as
the 2026-08-14 snapshot). Also fixed models.md's stale L2 build row (the L2
FineWeb-2 build is 92 B for the 1.7B run — commit f7a2bb5 — not 52 B).

## 4. Benchmark coverage of the trained languages — PARTLY FIXED

The auto-eval task selection intersects the `auto` benchmark group with each
cell's trained languages. Three problems, in decreasing severity:

a) **28 tasks were tagged `language: "??"` and could never be selected** —
   including 26 `include_base_44_*` languages whose identity is obvious
   (persian, polish, serbian, tagalog, …). A cell training Polish silently
   lost its INCLUDE-Polish task. **Fixed:** tags filled in `configs/tasks.json`
   (and `scripts/build_configs.py`'s LANG_MAP extended so rebuilds agree);
   `include_base_44` (the aggregate) → `multi`, `ceval-valid` → `zh`.

b) **`global_piqa` was not in the `auto` group** even though its
   completion-format tasks (including `_bos_latn` and `_nno_latn`) are
   already wired in tasks.json, and it is the single widest-coverage
   pretraining-stage benchmark in the harness (117 languages). **Fixed:**
   added `global_piqa` to `groups.auto` — this alone closes the only L50 gap
   (Bosnian) plus Nynorsk, and adds a second benchmark to many mid-list
   languages.

   Reproducibility footnote (pre-existing, not fixed): tasks.json has been
   hand-curated well past what `scripts/build_configs.py` can regenerate —
   194 `benchmark` fields would drift on a from-scratch rebuild (only
   `language` fields are merge-preserved). Treat the JSON as the source of
   truth and rebuilds as unsafe until the generator is reconciled.

c) **Residual gaps — DECIDED 2026-08-21 (option ii below, with a ≥ 2-family
   condition on the fill-ins): the eight are swapped for kin, jav, xho, hat,
   fao, zul, ibo, sot; the lists are now generated by
   `src/pretrain/data/generate_language_sets.py`, every available harness task
   is wired by `scripts/wire_harness_tasks.py`, and the rationale lives in
   `plan/benchmark_selection.md`. The L100 build must be redone.** Original
   finding: after (a)+(b), every L50 language has ≥ 1
   benchmark; in L100, eight languages have *nothing* in the harness:
   `epo_Latn`, `ltz_Latn`, `tat_Cyrl`, `div_Thaa`, `hif_Latn`, and the three
   noise subsets `gmh_Latn` (Middle High German), `nrm_Latn`, `bew_Latn`.
   They are covered by the primary metric (per-language BPB) but not by any
   benchmark. Options: (i) accept — BPB is the outcome metric, benchmarks are
   secondary; (ii) add a "≥ 1 benchmark exists" condition to the L100
   selection and replace the eight with the next byte-ranked covered
   languages (gla, kin, jav, xho, ceb, yue, hat, mri are all
   belebele-covered) — but this **changes the L100 data build**, so decide
   before that build is consumed by real runs. I did not change the lists.

   Worth knowing for later expansion (not implemented, cost/scope decision):
   the harness also has belebele in 122 languages (only 13 wired in
   tasks.json), global_mmlu full in 42 (11 wired), okapi m_arc/m_hellaswag
   (~31), and afrimmlu/afrixnli/afrimgsm if African languages ever enter the
   lists.

## 5. Language-list rationale is under-documented (but sound) — FIXED (generator)

The scheme-A lists are exactly "top-k FineWeb-2 subsets by `utf8_bytes`
(train split) from `fineweb2-language-distribution.csv`" **plus two
undocumented conditions**: (1) `und_*` (unidentified-language) subsets are
excluded; (2) `hau_Latn` is excluded because it is absent from the swiss-ai
filtered dataset dir the builds read (commit f57a4b7 "language code did not
exist"). Verified: with those two conditions the FW_L2…FW_L100 lists are
exact prefixes of the byte ranking; scheme B's S7/S14/S29 are exactly
"first language of each not-yet-seen script (then family), in byte order,
within the top-49". Two stale statements contradicted this: the plan claimed
the L30 list is "the TokEval set minus English" (it is the resource-ranked
top-29), and `hau_Latn` is still present in FW_L200 (harmless — the setting
was dropped — but inconsistent with its exclusion from FW_L100).

**Fix applied:** the lists are now *generated* (`generate_language_sets.py`,
reproducing every pre-existing list exactly, L100 excepted per finding 4c) with
the conditions in the JSONs' `description` fields and in the plan; TokEval claim
corrected; plan's stale
"validation covers all 199 languages" statements corrected to FW_L100 + English
(also in `build_data_mixtures.py`'s docstring — the code was already right).

## 6. Peak LR is computed for a fixed 100 B-token budget, not each run's D = 100·N — FIXED (2026-08-21, with the retrain)

`set_lr_and_bs` computes `lr = 0.3118·C^(−1/8)` with
`C = 6·N·desired_tokens` and `desired_tokens = 100e9` for **every** size,
while the runs actually train D = 100·N (9.3 B → 167 B). With each run's true
compute (C = 600·N²), 90M's LR would be ≈ 1.43e-3 (35 % above the configured
1.061e-3) and 1.7B's ≈ 6.93e-4 (6 % below). The 2026-08-14 revision
deliberately moved *from* the actual-budget LRs *to* the fixed-100B ones
("Peak LRs dropped to the reviewed generators' 6ND law") — but the fitted law
is a function of the run's own compute, so the current values are the less
standard choice. Consequences: every cell and both arch levels share the same
rule, so within-size rankings (the decision-accuracy analysis) are unaffected;
across sizes, each rung sits at a slightly different distance from its LR
optimum, which mildly bends the size-scaling fit.

**Fix applied** (the finding-2 retrain removed the consistency objection):
both generators now write `lr = 0.3118·(600·N²)^(−1/8)` — the law at the
run's own compute. New deep peak LRs: 1.428e-3 / 1.217e-3 / 1.029e-3 /
8.976e-4 / 7.996e-4 / 6.931e-4 (90M → 1.7B); shallow analogous from its own
N. The convention is recorded in both JSONs' `predictivity_schedule` note.

## 7. The "model depth" intervention is confounded — FIXED (2026-08-21, with the retrain)

The shallow ladder differed from the deep one not only in width/depth (128 vs
64) but also in per-size FFN multiplier (4,6,6,4,6,4 vs uniform 4) and GQA
ratio (2,3,2,3,4,2 vs uniform 4), because the architecture search optimized
them freely to hit each target N. The plan sells depth as "the most
controlled option (same data, same tokenizer, same size)" — as implemented it
was aspect ratio + FFN width + KV-head count changing together.

**Fix applied:** the shallow search now pins `ffw_multiplier = 4` and
`gqa_ratio = 4` (deep's uniform values), so every shallow layer has exactly
the deep layer structure and only the aspect ratio varies. Exact d = 128·L
had to be relaxed (with both knobs pinned, N moves in ~L³ steps and misses
targets by up to 20%): d_model is a multiple of 256 with width/depth in
[96, 160], and among candidates within 4% of the target the ratio closest to
128 wins. New ladder: L8·d1024 (−5.2%), L10·d1280 (−2.3%), L14·d1536
(+0.8%), L14·d2048 (+3.7%), L17·d2304 (+0.4%), L20·d2816 (−0.4%); ratios
110–146, ~2× deep's 51–77. Schedules, LRs, and the per-size save grid all
regenerate consistently (1×C still on-grid ≤600M).

## 8. The hyperparameter generators destroy the reviewed configs — FIXED

Running the checked-in generators would silently clobber the reviewed source
of truth: `find_hyperparams_shallow.py`'s `__main__` still targets the *old*
4-size ladder (100M/300M/500M/1B) and overwrites `hyperparams_shallow.json`;
`find_hyperparams_deep.py` drops the audited `nodes` field and resets the
hand-capped micro-batches (90M MBS 7 → 24, commit 308660d). Verified the
actual shallow shapes are exactly reproduced by the corrected search
(d_model = 128 × n_layers, head_dim 64, ffw ∈ {3,4,6}, GQA ∈ {2,3,4},
minimize |N − target| against the deep ladder's six non-embedding sizes —
all six match, no 1B pin needed).

**Fix applied:** shallow `__main__` retargeted to the six deep-ladder sizes
with the corrected constraint set; both generators now carry over existing
`nodes`/`micro_batch_size` values instead of resetting them. Regenerating
both files now produces a byte-identical no-op (verified with `git diff`).

## 9. Assorted stale documentation — FIXED

models.md: shallow 90M micro-batch (24 → 7), the fixed "Init std 0.008944"
row (init is width-scaled, 0.008944·√(1792/d)), the "1B shape pinned by
DECISION note" claim (the corrected search reproduces it unpinned), the
checkpoint row (now per-size interval), and the L2 data-build row (52 B →
92 B, largest run 1.7B). Pretrain CLAUDE.md "52 cells" → 56. Plan: run count,
×3 rows, validation coverage, TokEval claim, and the data table's L2 row
(55 B → the 1.7B-sized build; its FW_L100 build note now names settings
{2, 8, 30, 100}).

---

## Not findings (checked and sound)

- **Architectures:** all six deep and six shallow shapes recompute to their
  stated non-embedding counts; head_dim 64, GQA divisibility, tied
  embeddings, FFN non-gated ×2 factor all consistent with the S&N /
  OLMo-ladder non-embedding convention. Deep width/depth is exactly 64 for
  175M–1B (90M: 51.2, 1.7B: 76.8 — acceptable endpoint compromises to hit N).
- **Schedules:** predictivity iters/warmup/decay match D = 100·N at
  2,064,384 tokens/iter with the documented 4 %/20 % rounding; AdEMAMix
  α/β3 warmup = full run length per cell (uniform across sizes) is the right
  call for the size-scaling fit; width-scaled init anchored at d = 1792 is
  consistent across the ladder.
- **Data methodology:** 50/50 blend composed at training time from one
  English build + per-setting FineWeb-2 builds; per-language allocation at
  T = 1 with parquet-footer-corrected token estimation; per-source token
  targets counted exactly during the write; validation carve-out is
  row-exact, manifest-driven, and covers every trained language + English;
  build sizing (92 B / 52 B / 184 B + 10 % headroom) matches the 1.7B-included
  settings {1, 2, 8, 30, 100}.
- **Launcher/idempotency:** the cell-env contract is platform-identical; the
  resume path (marker rewind, capped train-iters +
  `--use-checkpoint-opt_param-scheduler`) preserves the LR trajectory.
