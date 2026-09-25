# RQ10 — Does a ranking that holds at 1.7B still hold one rung above it, at 3B?

## Research question

Every other question stops at the reference: a proxy is judged by whether it
makes the decisions the 1.7B model makes (rule 10, `analysis/RULES.md`). The
3B rung exists to ask the one question that cannot be asked inside that
frame — whether the reference itself is a proxy for the next rung. Four cells
were trained at 3B for it (deep, L ∈ {8, 15}, schemes A and B;
`plan/3b_models.md`), so this RQ is the only reader of
`build_snr_pool(above_reference=True)` and the only folder the rule-10 checker
exempts (`check_rules.EXEMPT`).

Snapshot: the ladder report of **2026-09-23 06:16** holds no 3B evaluation
yet; the tables below are written with their headers and the figure says so.
The driver (`run_all_predictivity.sh`, last RQ block) reruns the step every
pass, so the block fills in when the evaluations land.

## Setup

- **Population.** Every family with a final at the reference rung; pairs at
  the grid seed, multi-axis and mono-axis (rule 15). With the four 3B
  families that is six multi-axis and four mono-axis pairs — above rule 5's
  minimum, but a per-task DA sits on a coarse lattice (k/6, k/4), so the
  pooled ratio over tasks is what the lines draw.
- **Gate.** `predictivity`'s mask at the proxy; at the reference rung the same
  one-sided 95 % Wilson rule computed on the reference's own runs
  (`above_random.scores_and_mask`), since the committed mask stops at 1.7B.
- **Comparison.** Panels (a) and (c) read the same families to 1.7B, so the
  two references differ in nothing but the reference. Today's preview
  (`--reference 1.7B --design 3B`) is exactly that comparison line on its own.
- **Known-answer check.** `--reference 1.7B --check` reproduces rq02's
  `decision_acc_size_<proxy>` per task for both pair sets
  (`predictivity_schemes/da_per_task.csv`, the pool the rq02 decision figures
  pair over) — exact on all 3,312 cells on 2026-09-23. The script and rq02
  share the pair sets and the kernel; the `predictivity` folder's table is the
  A/B-only pool and differs by construction.

## Figures, in storyline order

<!-- BEGIN auto:above-reference-3B (above_reference.py --pool predictivity --reference 3B) -->
## The 3B rung as the reference

**DA-size and DA-goal · reference 3B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 3B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 3B`.

**Status: waiting for the 3B evaluations.** The ladder report holds 0 3B runs with scores, so every table is written with its headers only and the figure says so; the driver reruns this step every pass and the block fills in by itself.

![Size generalisation to 3B](pretraining/predictivity/above_reference_3B.png)

Files: [`above_reference_3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.png), [`above_reference_3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B.csv), [`above_reference_3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_3B_per_task.csv).
<!-- END auto:above-reference-3B -->

Key findings: none until the 3B evaluations land.

Follow-ups:

- When the 3B rows arrive, read panel (c) first: the per-task DA-size
  1.7B → 3B against 1B → 1.7B on the same four families says whether the
  reliable-task list of rq02 (`da_reliable_tasks.csv`) is reliable one rung
  further, which is the claim the paper's rq2 figure implicitly makes.
- Add the reference's earlier checkpoints as proxies (DA-ckpt at 3B) once
  its k/10 grid is evaluated; `per_task()` already takes any `frac`.
- The rung has four families, all deep: no depth pair, so the
  by-transformation reading is limited to the language count and the list.

### The preview: the four 3B-design families read to 1.7B

The comparison line of panel (a) on its own, so the 3B figure has something to
be read against the day it fills in.

<!-- BEGIN auto:above-reference-1.7B-design3B (above_reference.py --pool predictivity --reference 1.7B --design 3B) -->
## The 1.7B rung as the reference — preview on the `3B` design set

**DA-size and DA-goal · reference 1.7B · multi-axis and mono-axis pairs at the grid seed · gate `predictivity` at the proxy, the Wilson rule on the 1.7B runs at the reference · no filter.** Regenerate with `python analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 1.7B --design 3B`.

**Population.** 4 families with a final at 1.7B (lm-L15-deep-seed1904, lm-L15-schemeB-deep-seed1904, lm-L8-deep-seed1904, lm-L8-schemeB-deep-seed1904); DA-size pooled over the gated benchmark tasks with ≥ 3 pairs.

![Size generalisation to 1.7B](pretraining/predictivity/above_reference_1.7B_design3B.png)

| axes | proxy | DA-size → 1.7B | tasks | DA-size → 1.7B (same families) |
|---|---|---|---|---|
| mono-axis | 175M | 0.49 | 42 | 0.49 |
| mono-axis | 350M | 0.48 | 64 | 0.48 |
| mono-axis | 600M | 0.44 | 71 | 0.44 |
| mono-axis | 1B | 0.49 | 72 | 0.49 |
| multi-axis | 175M | 0.48 | 44 | 0.48 |
| multi-axis | 350M | 0.48 | 66 | 0.48 |
| multi-axis | 600M | 0.43 | 74 | 0.43 |
| multi-axis | 1B | 0.49 | 76 | 0.49 |

Files: [`above_reference_1.7B_design3B.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.png), [`above_reference_1.7B_design3B.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B.csv), [`above_reference_1.7B_design3B_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq10_size_generalisation/pretraining/predictivity/above_reference_1.7B_design3B_per_task.csv).
<!-- END auto:above-reference-1.7B-design3B -->

Key findings (preview, 06:16 snapshot):

- On these four families alone, DA-size to 1.7B sits at a coin flip from every
  proxy size (multi-axis 0.48 / 0.48 / 0.43 / 0.49 at 175M / 350M / 600M / 1B
  over 44–76 gated tasks; mono-axis 0.49 / 0.48 / 0.44 / 0.49), against
  0.53 → 0.56 on the full 23-family population (rq02, figure 1). Four
  families give six pairs, three of them between the two language lists at
  one L — decisions rq02 already reads at chance — so the 3B question starts
  from a population whose 1.7B ranking the ladder does not resolve.
- DA-goal along the run (panel b) never leaves 0.40–0.60 for any proxy size.
- The gated task count is 44–76 because the four families train 8 or 15
  languages (rule 2): the 3B answer will be about those languages' tasks.

Follow-ups: see above; and consider whether the rung's design set should be
widened (a shallow cell, or L30) before it is read as a generalisation test.

## Extensions from other sweeps

None: the 36-model sweep has no rung above its 1B reference, and the public
models' size steps (rq02, "Extensions") are between-lab decisions, not this
ladder's.

## Files

- `above_reference.py` — the one script; `--reference` picks the rung,
  `--design 3B` restricts to the rung's design set, `--check` compares with
  rq02, `--out-dir` writes elsewhere (the check).
- `pretraining/predictivity/above_reference_<ref>[_design<d>].{png,csv}`,
  `..._per_task.csv`.
