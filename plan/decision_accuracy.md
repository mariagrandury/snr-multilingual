# Decision accuracy: definitions, pairs, pools, and the RQ0–RQ2 figures

*2026-09-22. Status: **§7 steps 1, 2 and the rq02 half of 5 are implemented**
(see §9 for what landed and what the new tables say); steps 3 and 4 are not.
Everything below §1 was written against the tables of 2026-09-21 and its
§5–§6 numbers are superseded — see §9. The two exploratory figures it
introduces are written by `analysis/rq02_decision_accuracy/pair_axes.py` and
`analysis/rq01_scaling_predictability/regimes_survivorship.py` and replace
nothing.*

## 0. Recommendations, in one screen

| # | Decision | Recommendation | Why (one line) |
|---|---|---|---|
| 1 | Pair definition | Carry **both** `multi-axis` and `mono-axis` in every DA table (an `axes` column); make **mono-axis the headline for decisions**, keep multi-axis for continuity and power | Upstream's "all pairs" *is* mono-axis: DataDecide's recipes differ in one axis. Our multi-axis mixes six. Measured: it lifts DA-size/DA-goal by 0.04–0.05 without changing the trend, and leaves DA-ckpt untouched (§2.3) |
| 2 | Reference definition | Keep the three names, compute **two grids**: DA-goal (proxy@ckpt vs reference final) and DA-ckpt (proxy@ckpt vs own final). DA-size is DA-goal's final-checkpoint row, not a third computation | Removes one code path and one way for the three to disagree (§2.1) |
| 3 | Seed pairs | Exclude from every decision set; report DA over seed pairs as **DA-null**, the chance floor | Two draws of one design decide nothing; their DA is what "no signal" looks like (§2.4) |
| 4 | Pools | Collapse to **two**: `grid` (seed 1904, every scheme) and `all` (every seed). Everything else becomes a filter the script declares | A pool today is already a filter plus an output folder plus a gate identity; the folder is what stops figures being compared and the A/B filter is what leaked into DA (§3) |
| 5 | Gate | Keep rule 1 as is; compute the mask once on `grid`; **rerun rq01 and rq02** — both chains predate the twins entering the pool | Only rq00 carries the twins; rq01's and rq02's tables have zero of them (§4) |
| 6 | RQ01 paper figure | Add the gate's survivorship (new panel or labels), drop the saturated Spearman panel, tighten the axes | 110 of 379 tasks survive; five families lose every task and the figure never says so (§5) |
| 7 | RQ02 paper figure | Draw the three panels on **mono-axis** pairs, black line = pooled mono-axis, coloured lines = per axis; give the per-panel decision counts | The per-axis lines are already mono-axis; the black line beside them is not (§6) |
| 8 | `rq2.*` filename | Decide which figure the paper's `fig:rq2` shows; `make_rq_figures.py:48` and `04_analysis.tex:65` still point at the ten-checkpoint figure | A rerun of `make_rq_figures.py` replaces the paper's figure with one its caption does not describe |

Open questions the user has to answer are collected in §8.

---

## 1. What AllenAI's decision accuracy is, exactly

**Kernel.** `snr/metrics.py::decision_acc_fast_upstream` — every unordered pair
of a score vector, `a > b` on the proxy side against `a > b` on the target
side, the share that agree. Our `decision_acc_fast` keeps the same pairs and
compares the sign of the difference instead, so a tie is not decided by the
listing order (2.8 % of ladder pairs are exact ties; documented in the kernel's
docstring and rq02's README).

**Population.** `snr/snr_simple.py::compute_decision_accuracy` groups the
frame by **`mix`** and takes each mix's last checkpoint: the score vector has
one entry per data recipe. In DataDecide the recipes are the *only* thing that
varies at a size — same architecture, same seed, same token budget — so:

| | DataDecide (upstream) | Predictivity ladder (ours) |
|---|---|---|
| what a "model variant" is | one data recipe | one cell: (L, arch, list, T, lang2, seed) |
| axes a pair can differ on | **1** (the recipe) | up to **6** |
| pairs at a size | C(25, 2) = 300, every one a single-axis decision | 231 at the grid seed, 174 of which move ≥ 2 axes |
| reference | 1B final | 1.7B final |
| checkpoint axis | none in DA (only in the noise estimate) | DA-ckpt, DA-goal (ours) |

**Consequence.** "All pairs" upstream and "mono-axis" here are the same
object. Our multi-axis convention is the departure: it admits comparisons such
as `L8-A-deep vs L50-B-shallow`, which no practitioner makes and which never
existed in the paper we extend. The comparability argument runs the other way
from how it was made in this session's discussion: mono-axis is the faithful
generalisation.

---

## 2. The 3 × 2 taxonomy

### 2.1 The three references

| name | proxy read at | reference | question | where today |
|---|---|---|---|---|
| **DA-size** | final checkpoint of size N | final of the largest size | how small may a *fully trained* model be | `compute_da.compute_size_decision_accuracy`, `scale_convergence` |
| **DA-ckpt** | checkpoint f of size N | final of **size N** | how early may I stop a run | `compute_da.compute_ckpt_decision_accuracy`, `by_L` (`da_own`) |
| **DA-goal** | checkpoint f of size N | final of the largest size | how early *and* how small | `compute_early_small_decision_accuracy`, `by_L` (`da_ref`) |

Two identities the code does not yet exploit:

- **DA-size = DA-goal at f = 1.0.** It is the last row of the DA-goal grid,
  computed today by a separate function with its own gating call.
- **DA-ckpt at N = reference = DA-goal at N = reference.** The reference's
  own trajectory is the same line in both readings.

*Recommendation.* Compute two grids — `goal[size, frac]` and `ckpt[size, frac]`
— and name DA-size as a slice. One kernel (`scale_convergence.reliability`,
which already takes an explicit pair list, a fraction list and a reference
size, and reproduces `decision_acc_fast` exactly) serves all three; the
`_flops` figures are already built this way. This is what makes the pair-set
column (§2.3) a one-place change rather than a three-place one.

### 2.2 The two pair sets

| | multi-axis | mono-axis |
|---|---|---|
| definition | every unordered pair of design variants at the grid seed | pairs differing on **exactly one** of L, arch, list, T, lang2 (seed held) |
| `scheme` handled as | one token (A, B, AT3, BT3, ZH, ES) | three axes: list ∈ {A, B}, T ∈ {1, 3}, lang2 ∈ {ru, zh, es}; AT3 = A at T3, ZH = A's list with Chinese second |
| pairs, grid seed, every scheme | **231** (190 at the reference) | **57** (54 at the reference) |
| pairs, grid seed, A/B only (`predictivity`) | 153 | 51 |
| what upstream would call it | — | "all pairs" |

Composition of the 153 `predictivity` pairs, the population behind every DA
number in the repo today: L only 36, arch only 9, list only 6 → 51 mono-axis
(33 %); L+arch 36, L+list 30, L+arch+list 30 → 102 multi-axis (67 %).

The 57 mono-axis pairs by axis: language count 48 → 36 at the reference;
depth 10 → 10; language list 8 → 6; temperature 2 → 2 (below `MIN_PAIRS`;
becomes 3 when BT3 lands); second language 3 → 0 (ZH's 1.7B eval is not in the
report, ES is capped at 1B).

### 2.3 Measured: what the pair set changes

`pair_axes.py`, the 23 `above_66_both` cells, pooled over decisions, both pair
sets on the same gate and `MIN_PAIRS`
(`analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_both_axes.{png,csv}`):

| definition | reading | multi-axis | mono-axis | Δ | decisions @175M (multi / mono) |
|---|---|---|---|---|---|
| DA-size | 175M final | 0.647 | 0.603 | −0.044 | 921 / 343 |
| DA-size | 1B final | 0.786 | 0.746 | −0.040 | |
| DA-ckpt | 175M @0.5C | 0.590 | 0.594 | +0.004 | 1024 / 350 |
| DA-ckpt | 175M @4.5C | 0.879 | 0.857 | −0.022 | |
| DA-ckpt | 1B @4.5C | 0.882 | 0.855 | −0.027 | |
| DA-goal | 175M @0.5C | 0.562 | 0.510 | −0.052 | 921 / 343 |
| DA-goal | 175M @5C | 0.647 | 0.603 | −0.044 | |
| DA-goal | 1B @5C | 0.786 | 0.746 | −0.040 | |

Three findings:

- **DA-ckpt is indifferent to the pair set** (within ±0.03 everywhere).
  Whether a run's early ranking matches its own final ranking depends on the
  run's noise, not on how far apart the two designs are.
- **DA-size and DA-goal drop uniformly by 0.04–0.05 under mono-axis, with the
  same slope** (DA-size rises +0.139 multi, +0.143 mono). Multi-axis pairs are
  *easier* — a pair that moves three axes has a larger score gap, so even a
  175M model orders it correctly — and they are two thirds of the set. The
  headline levels are overstated by about 0.05; the headline trends are not.
- **Mono-axis has a third of the decisions.** The 600M → 1B dip in mono-axis
  DA-size (0.757 → 0.746) is the price: one third the decisions, per-size
  lines that wobble. Per-cell `MIN_PAIRS` losses are not yet counted (§8).

### 2.4 Pros and cons

**Multi-axis (today's convention)**

- pro: three times the decisions per cell; rule 5 rarely bites
- pro: continuity — every rq02/rq03/rq04 number on disk is multi-axis
- pro: for the *reliability* question ("does this benchmark order models
  consistently?") every pair is arguably the right population
- con: two thirds of the decisions are comparisons nobody makes
- con: inflates the level by ~0.05 and flattens the scale trend (§2.3)
- con: not what upstream measured, despite the repo saying it is

**Mono-axis**

- pro: every decision is attributable to one design choice — the
  `--by transformation` lines are exactly its per-axis decomposition, so the
  pooled black line would finally be the pooled version of the coloured ones
- pro: the faithful generalisation of DataDecide's pair set (§1)
- pro: the multi-axis "easy pair" inflation disappears
- con: a third of the decisions; temperature (2 pairs) and second language
  (0 at the reference) draw no line until BT3 and ZH's 1.7B land
- con: `decision_acc_fast` cannot express it — every consumer of
  `da_per_task.csv` needs the explicit-pair kernel or the new column

**The seed axis.** A seed pair is two draws of one design: there is no right
ordering, so agreement between a small and a large model on it measures only
whether noise is *consistent* across scale. That is a null, and a useful one:
DA over seed pairs is what a benchmark with no signal reads. Today seed pairs
exist only at 175M/600M/1B (never at the reference), so DA-null is defined for
DA-ckpt at those sizes and for DA-size/DA-goal only with a per-size reference.
*Recommendation:* exclude seed pairs from both decision sets; report DA-null
where it exists; draw it as the floor on the DA-ckpt panel.

**Is 3 × 2 the right shape?** Yes as a *table schema* — `panel ∈ {size, ckpt,
goal}` × `axes ∈ {multi, mono}` — with two additions that are group-bys, not
new columns: the per-axis breakdown of the mono-axis set (`--by
transformation` already does it) and DA-null. The one thing I would *not* add
is a third pair set between the two ("pairs moving ≤ 2 axes"): it has no
interpretation.

### 2.5 Implementation sketch (from the compaction discussion, confirmed)

- one `da_per_task.csv` with an `axes` column; every existing consumer
  filters `axes == "multi-axis"` by default, so no number moves silently
- the axis decomposition (`list`, `T`, `lang2` from `DATA_SCHEMES`) moves from
  `scale_convergence.py` into `analysis/utils.py` (CLAUDE.md: a rule lives in
  the shared layer)
- `compute_da.py` gains the explicit-pair kernel for mono-axis; DA-goal
  columns join the table (`decision_acc_goal_f<NN>_<size>`)
- rq2 figures come in `_multi_axis` / `_one_axis` pairs in one directory (§3)

---

## 3. Pools: what they are, and two instead of six

### 3.1 What a pool is today

`build_snr_pool` loads the whole ladder frame and applies the member spec
through `_LADDER_FILTERS = {seeds, sizes, L, arch, scheme}`. A pool name adds
exactly three things a filter would not: the **output directory**
(`analysis/<rq>/<stage>/<pool>/`), the **gate's identity** (`above_random.py
--only predictivity`), and the **README block's identity**.

| pool | filter | families | used by | what it is for |
|---|---|---|---|---|
| `predictivity` | seed 1904, scheme ∈ {A, B} | 18 | rq00 gate, rq02 DA tables, rq03, rq04, rq09, the paper | the headline SNR pool — schemes that would widen the *signal* are kept out |
| `predictivity_schemes` | seed 1904 | 22 | nothing in the driver | "analyse the scheme axis itself" |
| `predictivity_all` | none | 36 | rq01, rq05, rq06, `by_L` and `scale_convergence` *data* | every cell |
| `predictivity_seeds` | scheme ∈ {A, B} | — | rq03 seed noise | replicates as separate models |
| `predictivity_seeds_train` / `_test` | seeds 64/313 vs 1904 at six cells | — | rq03 holdout | |

### 3.2 What went wrong with this in practice (this week)

- the A/B filter of `predictivity` was **inherited** into `by_L`'s pooled
  panel and `scale_convergence`'s black line for a reason (signal dispersion)
  that does not apply to a rank agreement; found and removed 2026-09-22
- `reliable_tasks` still judges a cell "reliable" from `predictivity`'s DA
  table (A/B only, multi-axis) and that verdict filters `predictivity_all`
  decisions — a population mismatch that no checker sees
- `by_L` and `scale_convergence` read `predictivity_all` but write under
  `predictivity`, because the *gate* lives there; rq01's `regimes.py` reads
  `predictivity_all` and filters to deep/A/1904 **inside the script** — the
  filter-not-pool pattern already exists in the repo
- comparing the same figure across pools means opening two directories

### 3.3 Two pools, filters for the rest

| | keep six pools | **two pools + declared filters** |
|---|---|---|
| directories per RQ | up to 4 | 1 (or 2) |
| where a population is stated | `models.json` | in the script, and therefore in its caption (rule 13 for free) |
| gate | one per pool that runs `above_random` | computed once on `grid`; `all` inherits it |
| SNR signal (rq03/rq04) | protected by the pool | the script must declare `scheme ∈ {A, B}` explicitly — a one-line filter, but it must be *remembered* |
| migration | none | paths in every README, `make_rq_figures.py`, `paper_figures.md`, `check_rules.py`; one gate recompute; a full pipeline run |
| risk | the leak in §3.2 recurs | a script forgets its filter and its signal widens silently |

*Recommendation.* Two pools, named for what they are: **`grid`** (seed 1904,
every scheme — today's `predictivity_schemes`) and **`all`** (every seed —
today's `predictivity_all`). The A/B restriction becomes a named filter
constant in `analysis/utils.py` (`HEADLINE_SCHEMES = ("A", "B")`) that rq03/
rq04/rq09 apply and print. The two holdout pools become a pair of seed
filters inside rq03. `check_rules` gains a rule: a script that pools an SNR
signal must name its scheme filter in its output's caption.

The migration is mechanical but touches every README; do it in one commit,
after the `axes` column, so the pipeline is rerun once.

---

## 4. The above-random gate (RQ00) and how it reaches DA

**Definition** (`above_random.py`): a run is above chance when the one-sided
95 % Wilson lower bound of its accuracy over the task's items clears
`1/n_options` (`ALPHA = 0.10` two-sided); a (task, size) cell passes when at
least `MIN_SHARE = 0.5` of the size's runs do. No fixed margin. BPB and the
loss have no chance level and pass unconditionally. Computed only for the
`predictivity` pool.

**Reach into DA** (rule 1): DA-size and DA-goal need the task above chance at
the proxy *and* at the reference; DA-ckpt at the proxy only, since its
reference is the proxy. `early_small.py` and `scale_convergence.py` do this;
`reliable_tasks.py` did not until 2026-09-22 (three cells at chance at 1.7B
had counted as reliable).

**Survivorship, in numbers.** rq00 highlight: 647 benchmark tasks, 331 clear
chance at ≥ 1 size, 307 at 1.7B, 316 nowhere. rq01: 379 tasks have a size fit,
110 survive to a point (29 %). The families that vanish are the four-option
MCQ families — `global_piqa_parallel_cloze` 0/63, `belebele` 0/59,
`global_mmlu_full` 0/29, `arc` 2/28, `include_base_44` 1/36 — which the rq00
README already attributes to option count and format. The reformulated
twins (`rf_*`, `rfgm_*`) exist to rescue exactly these: RULES.md's new section
reports the gate keeping 0 of 37 Global-MMLU tasks against 35 of their `rf_`
twins.

**Where the twins are.** `first_size_above_random.png` (rq00) has
`rf_belebele`, `rf_global_mmlu_full`, `rf_include_base_44` rows with many blue
cells: **rq00 is current**. `rq1_fits.csv` and `scaling_regimes.csv` contain
**zero** twin tasks against 124 in today's `predictivity_all` frame: **the
whole rq01 chain is stale** (rule 14). rq02's `da_per_task.csv` (382 rows,
written 2026-09-21 21:35) contains **zero** twin tasks as well, so every DA
table, the reliable populations and both rq2 figures are stale in the same
way. Of the three RQs only rq00 carries the twins.

**RQ00 figure comments.**

- `first_size_above_random.png` is the right object (benchmark × language,
  cell = smallest size above chance) but at ~80 languages it renders 4845 px
  wide; unreadable at paper width. For the paper: per family, the *share of
  languages* above chance at each size — a 17 × 5 heatmap — with the twins
  beside their originals (`grids.display` already names them `belebele-rf`).
- The figure header says "Wilson 90 % lower bound", the README says
  "one-sided 95 %"; both are `ALPHA = 0.10`, say it one way.
- rq00 README line 36 still describes the retired fixed-margin rule
  (`mean score > 1/n_options + 0.05`); line 103 has the current one.

---

## 5. RQ01 paper figure: `scaling_regimes_outliers_paper.png`

**What it shows today.** One point per surviving task (110), medians over the
deep / scheme-A / seed-1904 cells: (a) median R² of the log-N fit against
median Spearman ρ; (b) that R² against the median R² of the trajectory fit,
quadrants at 0.5. Family labels carry the task count; six outliers named
(`multiblimp:bn/da`, `xcopa:th/tr`, `xwinograd:fr/ru`).

**Findings from the current version, from `scaling_regimes.csv`.**

| regime | tasks |
|---|---|
| predictable across both | 102 |
| predictable across size only | 6 |
| weak / unpredictable in both | 2 |
| declines with size | 0 |

Medians: R² size 0.951 (IQR 0.892–0.965), R² trajectory 0.887 (0.801–0.922).
The reading "scaling is predictable" is true of the 29 % of tasks that clear
the gate; the figure does not say which 71 % it is silent about.

**Problems.**

- *Survivorship is invisible.* 224 tasks removed; five families lose every
  task (151 + 4 tasks); the labels say "hellaswag (18)" where "18/26" is the
  fact. A reader cannot tell that the four-option families are absent
  because they are *at chance*, not because they are unpredictable.
- *Panel (a) is saturated.* ρ is over ≤ 5 rungs (of 944 fits, 261 have 5
  rungs, 127 have 4, 431 have 0); ρ = 1 means "monotone over five points",
  which nearly everything is. The y axis runs −1 to 1.5 while the data sits
  in 0.2–1.0 and no task declines: 60 % of the panel is empty by design.
- *Stale.* No twins (above). Rerunning `analyze.py` → `regimes.py` will
  change the population materially — the twins are the families the gate
  removed.
- *It already filters inside a pool* (deep/A/1904 from `predictivity_all`):
  evidence for §3.

**New figure: `scaling_regimes_survivorship.png`** (same table, same
medians; `regimes_survivorship.py`). Panel (a) is one bar per family, kept
against removed; panel (b) is the current right panel with kept/total in
every label and the five no-survivor families listed inside it.

Findings it adds: 110 of 379; `bpb` 30/50 — twenty per-language BPB tasks are
gated by rule 2 (untrained languages), not by chance; `multiblimp` 22/34 and
`hellaswag` 18/26 are the only benchmark families that survive at scale;
everything the paper can say about MCQ scaling today rests on `arc` (2) and
`include` (1). After the twins the picture should invert for belebele and
Global-MMLU — that is the test of the reformulation, and this figure is how
to show it.

**Recommendations, ordered.** (1) Rerun the rq01 chain on today's pool before
anything is cited. (2) In the paper figure, replace panel (a) with the
survivorship bars or fold kept/total into the labels and drop (a). (3) If (a)
stays, use the fit's *slope per decade* (median 0.129 for benchmarks, in
`rq1_fits.csv`) instead of ρ: effect size beside predictability. (4) Tighten
the axes to the data; note "0 tasks decline" in the caption instead of
reserving half the panel for it. (5) Caption: "of N tasks with a size fit, n
are above chance somewhere and drawn".

---

## 6. RQ02 paper figure: `rq2_above_66_both_transformation.png`

**What it shows today.** Three panels on the 23 `above_66_both` cells (14
languages, 5 families): DA-size by design axis with a pooled black line;
DA-ckpt and DA-goal per proxy size on the checkpoint axis.

**Findings from the current version** (`rq2_above_66_both_transformation.csv`).

| panel | line | 175M | 350M | 600M | 1B | reading |
|---|---|---|---|---|---|---|
| DA-size | all pairs | 0.647 | 0.751 | 0.771 | 0.786 | rises, never reaches τ = 0.9 |
| DA-size | language count | 0.580 | 0.734 | 0.837 | 0.774 | recovers, dips at 1B |
| DA-size | depth | 0.793 | 0.647 | 0.520 | 0.785 | non-monotonic |
| DA-size | language list (A vs B) | 0.556 | 0.667 | 0.417 | 0.479 | below a coin flip at 600M and 1B |
| DA-ckpt | 175M → 1.7B at 4.5C | 0.850 | 0.890 | 0.895 | 0.888 (1B) | every size 0.85–0.90 |
| DA-goal | 175M / 1B at 4.5C | 0.619 | | | 0.795 | separated by size, stays separated |

The three claims the README draws — reading early is cheap, reading small is
not, the pooled DA-size hides which decisions recover — all hold.

**Problems.**

- *The black line and the coloured lines are different populations.* The
  coloured lines are mono-axis by construction (a pair moving two axes is
  dropped); the black line is multi-axis. The figure invites the reader to
  see the black line as their pool, and it is not.
- *Multi-axis inflation.* On this population the pooled DA-size reads 0.647
  → 0.786 multi-axis and 0.603 → 0.746 mono-axis (§2.3). The paper would
  overstate the level by ~0.04.
- *The list line rests on 18–48 decisions* and swings 0.42–0.67; it is the
  least stable line in the figure and the one that carries the sharpest
  claim.
- *The population was judged on a different pair set and pool* than it
  filters (§3.2).
- *Three panels, two x-semantics* (size; tokens) with one shared y — by
  design, and worth keeping, but the caption must say the middle and right
  panels' x is the *proxy's* run.
- *The filename* (`rq2.*`) collides with the paper's current `fig:rq2`
  (§0, row 8).

**New figure: `rq2_above_66_both_axes.png`** (`pair_axes.py`): the same three
definitions, rows = pair set, same 23 cells, same gate. Findings are §2.3's
table: DA-ckpt unchanged, DA-size/DA-goal −0.04 to −0.05 with the same shape,
a third of the decisions. Conclusions are robust to the pair set; levels are
not.

**Recommended paper version.** Left: mono-axis, black = pooled mono-axis,
coloured = per axis (they are then parts of the black line); a faint
multi-axis black line for the reader who wants the old convention. Middle and
right: mono-axis, with DA-null (seed pairs) as the floor on the DA-ckpt panel
once computed. Every panel states its decision count at the smallest and
largest size. Read against the *reference's own trajectory* (the 1.7B line)
rather than a fixed dotted line, since 0.75 is a convention and the
reference's own DA-ckpt at 4.5C (0.871) is a measured ceiling.

---

## 7. Implementation order

1. `analysis/utils.py`: the design-axis decomposition (`list`, `T`, `lang2`,
   `design_axes(df)`), the `HEADLINE_SCHEMES` constant, a `pairs(attrs,
   axes=)` helper returning multi-axis / mono-axis / seed-only pair lists.
2. `compute_da.py`: the explicit-pair kernel (lift
   `scale_convergence.reliability`), the `axes` column, DA-goal columns,
   DA-null. Consumers filter `multi-axis` by default.
3. Two pools; `above_random` on `grid`; paths migrated; `check_rules` rule for
   declared filters. One full pipeline run.
4. rq01 chain rerun (it is stale regardless of 1–3); `regimes.py` gains
   survivorship labels; the paper figure is re-cut.
5. rq02: `reliable_tasks` on `grid`'s table; rq2 figures in `_multi_axis` /
   `_one_axis` pairs; the mono-axis version becomes the paper's; the `rq2.*`
   name resolved with `make_rq_figures.py` and the tex caption in one change.

Steps 1–2 are the big change discussed on 2026-09-22 and are unstarted; the
review of this session's rq02 work is committed (`2f6c05c`).

## 8. Open questions

1. **Headline pair set for the paper: mono-axis?** The levels drop ~0.05; the
   story does not change; the definition becomes defensible against upstream.
2. **Does `MIN_PAIRS = 3` survive mono-axis per cell?** Pooled decisions drop
   to a third; the count of (task, size) cells that fall below 3 pairs is not
   yet measured. If many do, a lower floor for mono-axis or a per-axis pooling
   is needed — decide after measuring.
3. **Two pools, or three** (`grid`, `all`, and a `headline` = A/B for the SNR
   RQs)? Two is cleaner; three needs no filter discipline in rq03/rq04.
4. **Which figure is the paper's `fig:rq2`** — the ten-checkpoint one the
   tex describes, or the three-panel one? This blocks the next
   `make_rq_figures.py` run.
5. ~~Does rq02's `da_per_task.csv` contain the twins?~~ Resolved: it does
   not (0 of 382 rows). rq01 **and** rq02 are stale; only rq00 carries the
   twins. Both chains rerun before anything is cited — which argues for
   doing steps 1–3 of §7 first and running the pipeline once.
6. **DA-null:** report it in the paper as the floor, or keep it as a
   diagnostic?
7. **The 600M dip** (two of four DA-size axes; language count peaks there and
   falls at 1B): a 1B-rung effect or a population effect? The balanced-panel
   check on the pooled line did not remove it.


---

## 9. Implementation log — 2026-09-22

**Landed.** `analysis/utils.py` now owns the decomposition and the pair sets
(`DESIGN_AXES`, `SECOND_LANG`, `AXES_SUFFIX`, `design_axes`, `pair_sets`,
`pair_agreement`, `one_axes`; the `HEADLINE_SCHEMES` constant §3.3 proposed
was written and then removed in review — nothing read it, the A/B filter
stays in `models.json`'s pool definitions until §7.3 happens); `compute_da.py`
computes every table once per pair set and tags it `axes`, and adds the DA-goal
wide columns; `reliable_tasks.py` gates the three kinds separately, carries
both pair sets and reads its DA from `--da-pool predictivity_schemes` while the
gate and the outputs stay with `predictivity` (option C, so #9 is fixed without
renaming a directory); `by_L`, `scale_convergence` and `paper_rq2` take
`--axes` and write `_one_axis` twins beside the existing names; the driver runs
both passes. Every other consumer calls `one_axes()` and so still reads the
multi-axis table it always read.

**Two checks worth keeping.** `pair_agreement` reproduces
`snr.metrics.decision_acc_fast` exactly on a full pair set, ties included, so
the mono-axis reading is the same kernel on fewer pairs and not a second
implementation. And `compute_da` asserts DA-size == DA-goal@f100 on every run:
two independent code paths, max |diff| = 0.00e+00.

**Not landed.** §7.3 (two pools) renames output directories, which is a file
relocation and needs the owner's go-ahead; §7.4 (rq01 rerun, survivorship in
the paper figure); the `rq2.*` filename (#4).

### What the recomputed tables say

The ladder report refreshed at 06:13 on 2026-09-22: the reformulated twins
entered the pool (**541 parent tasks against 382**) and AT3 gained L15-deep and
L30-deep (38 families, the temperature axis 2 -> 4 mono-axis pairs). So the
§5-§6 numbers were stale twice over.

| population (`above_66_both`) | cells | languages | benchmark families |
|---|---|---|---|
| 2026-09-21 (stale) | 23 | 14 | 5 |
| multi-axis | **59** | 31 | 15 |
| mono-axis | 43 | 31 | 9 |

**The reformulation works.** `rf_belebele`, `rf_global_mmlu_full`,
`rf_include_base_44` and `rfgm_include_base_44` are all in the reliable set,
and `belebele`, `arc`, `include_base_44`, `xcopa` and
`global_piqa_nonparallel_cloze` reappear beside their twins. These are exactly
the four-option families §4 found the gate removing wholesale.

**Open question #2 is answered: mono-axis survives `MIN_PAIRS`.** 268
reliable-eligible tasks against 289 multi-axis (93 %), and 43 against 59 in
`above_66_both` (73 %). The pair minimum is not the binding constraint, so the
mono-axis reading is viable as a headline.

**DA-size on the paper population** (`rq2_above_66_both_transformation`):

| line | | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| all pairs | multi | 0.645 | 0.762 | 0.752 | 0.770 |
| | mono | 0.600 | 0.742 | 0.785 | 0.766 |
| language count | multi | 0.590 | 0.764 | 0.802 | 0.747 |
| | mono | 0.579 | 0.787 | **0.855** | 0.788 |
| language list (A vs B) | multi | 0.619 | 0.556 | 0.467 | 0.517 |
| | mono | 0.583 | 0.600 | 0.533 | 0.600 |
| depth | multi | 0.771 | 0.643 | 0.489 | 0.721 |
| | mono | 0.783 | 0.655 | 0.509 | 0.796 |

Three things changed against §6, and two of them change a claim:

- **The language-list decision is no longer below chance.** On the fresh
  tables it reads 0.583-0.600 under mono-axis and 0.467-0.619 under
  multi-axis, against 0.417-0.479 before. §6's sharpest sentence — "a fully
  trained proxy smaller than the reference is uninformative about which list
  to train on" — does not survive; the honest reading is now "weakly
  informative, and the weakest of the four axes".
- **Mono-axis is no longer uniformly lower.** §2.3 measured it 0.04-0.05 below
  multi-axis everywhere. On the fresh population it is *higher* at 600M
  (0.785 vs 0.752) and on three of the four axis lines. The "multi-axis pairs
  are easier" effect was partly an artefact of the old, thinner population.
- **The 600M dip survives** on depth and the language list, and language count
  still peaks at 600M and falls at 1B. Still unexplained (§8.7).

### What to do next, in order

1. Rerun rq01 (§7.4) — it is the only chain still on pre-twin tables, and §5's
   survivorship finding is the one the twins were built to change.
2. Rewrite §5 and §6 from the regenerated tables; the three claims above are
   the ones that moved.
3. Decide #1 (mono-axis as the headline), #3 (pools) and #4 (`rq2.*`).
