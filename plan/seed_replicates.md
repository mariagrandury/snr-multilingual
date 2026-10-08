# What the seed replicates are for

*2026-09-21. Decision doc: what to do with the ×3 seed cells already trained,
and whether to add seed replicates at the 1.7B reference. Numbers from
`analysis/rq03_noise_and_snr/pretraining/predictivity_all/effect_vs_noise.csv`
(benchmark rows) — see the provenance note at the end.*

## What exists

**20** trained non-1904 cells, every one of them deep (verified against the
checkpoint root):

| column | L | replicate seeds |
| --- | --- | --- |
| 175M, scheme A | 1, 2, 50 | 64, 313 |
| 600M, scheme A | 1, 2, 50 | 64, 313 |
| 1B, scheme A | 1, 2, 30 | 28, 1797 |
| 1B, scheme B (deep) | 30 | 28, 1797 |

Gaps: **no replicates at 350M and none at 1.7B**. The 1B × L50 pair was in
`SEED_TRIPLES` and never trained; it was **removed from the grid on
2026-09-21** along with every planned shallow replicate — see "What changed"
at the end.

**The triples use different seeds at different sizes** — 175M/600M are this
project's (64, 313, 1904), the 1B column is aromanou's (28, 1797, 1904), and
the grid names the seeds that exist rather than the seeds we would have picked
(`launch_trainings.py:258-272`). This matters more than it looks: a *family* is
`(L, arch, scheme, seed)` (`ladder.py:181`), and DA pairs only families present
at both the proxy and the reference. So a seed-64 family exists at 175M and
600M and nowhere else; **the seed axis is already discontinuous across the
ladder**, and no single replicate seed spans it.

## What they already buy

- **The noise denominator.** `effect_vs_noise.py` reads every intervention's
  |Δ| against the seed std of the baseline cell. Without replicates there is no
  denominator, and "A beats B by 0.01" means nothing.
- **The seed-holdout generalization test.** `compare_seed_splits.py` fits the
  variant ranking on seeds 64/313 and tests it on 1904 — currently 175M/600M
  only (`pools.predictivity_seeds_train` / `_test`).
- **They are deliberately kept out of the headline pool.** `pools.predictivity`
  is seed 1904 only: "seeds are noise, not signal".

## What the numbers say

Median over benchmark tasks:

| size | seed noise | n | ckpt noise (detrended) | seed / ckpt |
| --- | ---: | ---: | ---: | ---: |
| 175M | 0.0066 | 106 | 0.0019 | 2.65 |
| 350M | — | 0 | 0.0023 | — |
| 600M | 0.0055 | 131 | 0.0032 | 2.12 |
| 1B | 0.0054 | 109 | 0.0038 | 1.54 |
| **1.7B** | **—** | **0** | 0.0034 | **—** |

Two readings, and they pull in opposite directions:

1. **Seed noise is nearly size-invariant** — 0.0066 → 0.0055 → 0.0054 across a
   6× span of parameters, still falling but clearly flattening. Extrapolating
   ~0.005 to 1.7B is about as safe as extrapolation gets here. Checkpoint noise
   does the opposite (it grows with size), which is why the seed/checkpoint
   ratio falls from 2.65 to 1.54.

2. **Most effects at the reference are inside that noise.** Taking the
   extrapolated 0.0053 at 1.7B, the share of per-task intervention effects
   within 2σ (and within 1σ):

   | intervention at 1.7B | n | median \|Δ\| | <2σ | <1σ |
   | --- | ---: | ---: | ---: | ---: |
   | depth | 379 | 0.0073 | 0.64 | 0.39 |
   | language lists (A vs B) | 152 | 0.0078 | 0.61 | 0.39 |
   | temperature | 136 | 0.0098 | 0.54 | 0.24 |

   At 1B, using the **measured** 0.0054, the same shares are 0.67 / 0.68 / 0.58.
   The prediction and the measurement agree — which is itself the evidence that
   the extrapolation works.

So the reference ranking is, task by task, substantially a coin flip. That caps
decision accuracy, and it is why the ceiling below matters more than another
noise estimate.

## Ideas for the models we already have — no new compute

1. **The seed-noise ceiling (null DA).** DA between two seeds of the *same*
   design variant is the empirical chance level for DA. The right baseline on
   every rq02/rq05 figure is that band, not 0.5 — a measured DA of 0.60 against
   a null of 0.58 says the benchmark decides nothing. Already written down as a
   TODO at `analysis/paper_figures.md:289` and not implemented. 1B gives 3
   variants × 3 seeds (3 pairs per seed pair) now that L50 is out of the 1B
   triple, and 175M and 600M give 3 variants each. (`paper_figures.md:289`
   still says 4 variants / 6 pairs — fix it there when the figure is built.)
2. **Seed as a fourth line in the rq05 transformations figure.** The placebo:
   language count, temperature, depth, language lists — and seed, which by
   construction should sit at the null. On one gated item set it makes the whole
   figure interpretable, and it is a small change to `TRANSFORMATIONS`
   (`rq05/transformations.py:48-57`), whose cell namer currently pins
   `GRID_SEED`.
3. **Effect-size-resolved DA** (`paper_figures.md:288`): per pair, the
   reference's gap expressed in seed sds against DA at each proxy size. Turns
   "DA is 0.6" into "DA is 0.85 where the gap exceeds 2σ and 0.5 below it",
   which is the actually useful statement.
4. **Publish the noise floor per (size, L)** as a table in its own right. It is
   the denominator every reader needs and it currently lives only inside
   `effect_vs_noise.csv`.
5. ~~**Finish the 1B × L50 triple**~~ — 2 runs, ~502 node-h, and the one gap
   that would make the 1B column match the L ∈ {1, 2, 50} shape of 175M and
   600M. **Dropped 2026-09-21**: by this document's own argument the quantity
   is near size-invariant, so a fourth 1B setting buys no precision the other
   three do not already give, and the 502 node-h is better spent on the proxy
   rungs that are short of DA pairs. The 1B seed column now shares L ∈ {1, 2}
   with the other sizes and adds L30 of its own.

## Should we train 1.7B at seeds 313 and 64?

**For**

- It is the only *direct* measurement of the reference's own seed noise.
  Everything in the table above at 1.7B is extrapolated.
- It would extend the seed-holdout test to the reference rung, where the
  framework-generalization claim actually matters.
- The DA ceiling is what separates "small models are poor proxies" from "the
  1.7B ranking is itself noise", and more than half of the reference effects
  sit inside 2σ.

**Against**

- **Cost.** 605 node-h per run. Matching the 175M/600M shape (L ∈ {1, 2, 50} ×
  2 seeds) is 6 runs = **3,630 node-h** — more than three times what is left on
  the entire AT3 programme. Even one setting at two seeds is 1,210 node-h.
- **The quantity is nearly size-invariant,** and the 1B measurement already
  corroborates the extrapolation to the precision the argument needs.
- **Seeds 313 and 64 do not match the 1B triple** (28, 1797). A 1.7B-seed313
  cell forms families that pair with 175M and 600M but not with 1B, so the seed
  axis stays discontinuous. Two expensive runs would not fix the structural
  problem they appear to fix.
- **A 3-seed std has 2 degrees of freedom.** The interval on it is wide; we
  would be buying a noisy estimate of something we already predict well.
- **Nothing in the headline table changes** — `pools.predictivity` is seed 1904.

**Recommendation: no, not now.**

Spend a twelfth of it on the **350M triple** instead — L ∈ {1, 2, 50} × seeds
{64, 313}, 6 runs × 49 = **~294 node-h**. That closes the one real gap in the
size axis of the seed-noise curve and turns the extrapolation to 1.7B from
three points into four, using the same seeds as 175M and 600M so the families
line up.

Do ideas 1–3 first. They cost nothing, they are what makes the replicates we
already paid for pay, and the ceiling is the number that tells us whether 1.7B
seeds are worth buying at all. Revisit only if the measured DA at the reference
sits near the null: then two runs at **L50** (the richest gated task set — 108
seed-noise rows at 600M against 9 at L1) is the minimal informative buy at
1,210 node-h.

## Provenance note

These tables were written by the analysis run of 2026-09-20/21, which collided
mid-run with the `analysis/RULES.md` refactor (`trained_only` / `parents_only`
became loader defaults, `last_n` → `noise_window`). The medians and shares above
are directionally reliable but should be re-read after the next clean
`FORCE=1 bash scripts/refresh_analysis.sh`.

## What changed on 2026-09-21

The grid dropped **24 planned-but-never-launched replicate cells**, 3,782
node-h (`launch_trainings.seeds_for`, `SEED_TRIPLES`):

- **every shallow replicate** (22 cells). The seed axis is read as the noise
  denominator of the interventions, and every one of those is measured against
  a *deep* baseline — `INTERVENTIONS["arch"]` holds `("scheme", "A")` with
  `deep` as the baseline level, and the seed-holdout pools are declared "Deep
  scheme-A cells only". No analysis reads a shallow seed std. Every replicate
  ever trained is in fact deep, so this makes the grid say what the sweep has
  always done.
- **the 1B × L50 deep pair** (2 cells), idea 5 above.

Nothing on disk was touched: none of the 24 had a single checkpoint. The two
empty `lm-1B-L30-schemeB-shallow-seed{28,1797}` directories stay where they
are; they simply fall out of the grid. Their `configs/models.json` stubs (no
results attached) become prunable by `sync_models_json.py --prune`, which is a
manual step and was not run.

This does not touch the recommendation above: **still no 1.7B seed replicates**,
and the 350M triple (~294 node-h) remains the cheap way to turn the
extrapolation to 1.7B from three points into four.
