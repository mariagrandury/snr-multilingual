# L100: build the mixture and train the ladder? (2026-09-20)

*Decision (2026-09-20): the ladder ends at L50. L100 is dropped from the grid;
the temperature intervention is replicated at L30 and L15 instead — see
"The plan" and the two paper paragraphs below.*

## Paper paragraphs (draft)

**Research question.** The predictivity sweep varies four things a
practitioner would settle at small scale: the number of training languages
$L$, the sampling temperature $T$ over those languages, the model depth at
fixed parameter count, and the language list itself. We ask whether a proxy
size ranks these decisions the way the 1.7B reference does, and whether the
answer differs by decision kind. Specifically, for every pair of cells that
differ in one factor, we compute decision accuracy, the share of evaluation
items on which the proxy's ranking of the two cells agrees with the
reference's at its final checkpoint.

**Setup.** The language-count decision is read on consecutive settings of the
resource-ranked lists ($L = 1, 2, 8, 15, 30, 50$) at $T = 1$; the temperature
decision on the same lists at $T = 1$ versus $T = 3$ at $L = 50$, and, in the
extension, at $L = 30$ and $L = 15$, where flattening starves no language
(the smallest language keeps 343M and 1.2B tokens at $T = 1$); the depth
decision on deep versus shallow models at every $L$; the list decision on
resource-ranked versus diversity-first lists at $L \in \{8, 15, 30\}$. All
pairs share seed 1904 and the per-size budget $D = 100N$. Because the four
decisions are evaluated on different task sets, we compare them on one item
set: the items every decision kind decides somewhere, gated to the tasks
above chance at both the proxy and the reference (Table T1;
`analysis/rq05_design_decisions/transformations.py`).

**Expected finding.** We hypothesize that the two data decisions, language
count and temperature, are predictable from 350M upwards at a similar level,
because both act through the per-language token allocation, whose sign a
small model reproduces, while the depth decision stays at chance because its
effect at the reference is inside seed noise. Table T1 supports both parts on
the evidence available: on benchmarks the language count reaches 0.74 and the
temperature 0.77, and depth never leaves 0.47–0.63. It also shows that the
same decisions are ranked far more reliably on bits per byte than on
benchmarks, 0.75–0.92 against 0.52–0.77, which we attribute to the gate
leaving few benchmark items per pair. The temperature extension tests
this directly: if predictability tracks the size of the token reallocation,
the $L = 30$ and $L = 15$ pairs, whose reallocation is smaller, should be
less predictable than the $L = 50$ pair; if they are predicted equally well,
predictability is a property of the decision kind rather than of its effect
size.

**Why the ladder stops at $L = 50$.** The grid was designed with a seventh
setting of 99 languages, to be trained at $T = 3$ only. We did not train it.
At $T = 1$, 51 of the 99 languages receive under 100M tokens and the smallest
3.5M, so half the list is trained on data too small to move a benchmark.
Flattening the allocation to $T = 3$ raises the floor to 14.5M tokens and the
median to 373M, but the tail is data-limited rather than allocation-limited:
the build then realizes 75.4B tokens, short of the 83.6B the 1.7B rung draws,
so the reference would repeat its mixture 1.11 times, the only repetition in
the grid. The $L = 50$ pair, which trains the same 49 languages at both
temperatures, shows what such a reallocation buys: tripling a tail language's
tokens raises the share of its tasks above chance from 27% to 31% at 1.7B and
not at all at 1B. Consequently, an $L = 100$ ladder would add 50 languages
whose benchmarks stay at chance at either temperature, at the cost of a
repeated reference. We therefore stop the language axis at 50 languages,
where every language receives at least 83M tokens, and spend the compute on
replicating the temperature decision at $L = 30$ and $L = 15$.

### Table T1 — the four transformations on one item set

Mean decision accuracy over each transformation's pairs, at the proxy's final
checkpoint against the 1.7B reference, on the items every transformation
decides somewhere. Benchmarks are gated by the above-random mask at both the
proxy and the reference (92–94 tasks); bits per byte (BPB) is read on all 100
validation languages, the only language set every pair shares. Source:
`analysis/rq05_design_decisions/transformations.py`, pool `predictivity_all`,
ladder report of 2026-09-19.

| transformation | pairs (with data) | 175M | 350M | 600M | 1B |
| --- | --- | ---: | ---: | ---: | ---: |
| **benchmarks** | | | | | |
| language count (L vs next L) | 5 (5) | 0.71 | 0.74 | 0.64 | 0.58 |
| temperature (T = 1 vs T = 3) | 1 (1) | 0.51 | 0.70 | 0.77 | 0.67 |
| language lists (A vs B) | 3 (3) | 0.58 | 0.60 | 0.60 | 0.52 |
| depth (deep vs shallow) | 6 (4) | 0.58 | 0.63 | 0.47 | 0.58 |
| **bits per byte** | | | | | |
| language count | 5 (2) | 0.82 | 0.87 | 0.92 | 0.85 |
| language lists | 3 (3) | 0.57 | 0.76 | 0.82 | 0.75 |
| depth | 6 (1) | 0.82 | 0.53 | 0.67 | — |
| temperature | 1 (0) | — | — | — | — |

The temperature row rests on the single L50 pair, and its BPB row is empty
because the AT3 cells have no BPB scored yet. On benchmarks the two data
decisions are predicted at a similar level, 0.58–0.74 for the language count
and 0.51–0.77 for the temperature, while depth stays between 0.47 and 0.63.
On BPB the language decisions are predicted far better than on benchmarks,
0.75–0.92 from 350M up.


## The question

- The grid plans L=100 only as AT3 (T=3), deep and shallow, 175M–1.7B (+90M).
  Nothing at L100 exists: no AT3 build, no model, no eval.
- Is it worth ~1,000+ node-hours per ladder to add it, and if so which
  variant(s): the grid's AT3, a T=1 scheme-A twin, deep only, seeds?

## Where L100 stands

| | state |
| --- | --- |
| AT3 L100 build | never submitted (`launch_builds.sh` would; `build-at3-L100` is in its dry-run) |
| T=1 L100 build on capstor | 92B, built 2026-08-21 in 9.6 h — **pre-swap list** (`ltz` in, `kin jav xho hat fao zul ibo sot` missing); stage copy stalled at 22 GB (`data/fineweb_L100.bin.tmp`, 2026-08-25). Unusable as is |
| Current `FW_L100` | 99 languages, swap-ins present, L50 ⊂ L100; 50 languages only L100 trains |
| Models | 0 of 12 planned AT3 cells; `configs/models.json` already carries them (and 20 stale T=1 entries) |
| Paper text | abstract and intro say "seven language settings from 1 to 100", 162 runs — with L100 empty the grid delivers six settings |
| Eval load | 446 tasks per L100 cell vs 329 at L50 (+117, all in the tail 50 languages) |

Tokens per language per setting (92B target, builder's own per-language
estimates, no repetition; the simulation reproduces the built L50 plans exactly):

| setting | T | realizes | median | smallest | exhausted | under 100M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L15 (14 languages) | 1 | 92.0B | 4.0B | 1.2B | 0 | 0 |
| L15 | 3 | 91.6B | 6.1B | 3.8B | 2 | 0 |
| L30 (29) | 1 | 92.0B | 1.46B | 343M | 0 | 0 |
| L30 | 3 | 91.4B | 2.76B | 1.32B | 2 | 0 |
| L50 (49) | 1 | 92.0B | 935M | 83M | 0 | 4 |
| L50 | 3 | 87.1B | 1.83B | 336M | 13 | 0 |
| L100 (99) | 1 | 92.0B | 90M | 3.5M | 0 | 51 |
| L100 | 3 | **75.4B** (1.7B draws 83.6B: **1.11 epochs**) | 373M | 14.5M | 60 | ~16 |

Below L50 nothing is starved at either temperature; the flattening only
matters at L50 (four languages) and L100 (half the list).

**Does more tail data buy benchmark signal?** The L50 pair answers this
directly: the same 49 languages, T=3 giving the tail 3–5× the tokens. Tasks
above chance (Wilson lower bound over the random baseline, final checkpoint,
316 shared non-English tasks):

| tokens seen at T=1 | languages | tasks | above chance, T=1 | T=3 | T=3 tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.7B rung, all | 49 | 316 | 112 (35%) | 127 (40%) | |
| < 150M | 9 | 49 | 13 (27%) | 15 (31%) | 305–514M |
| 150–500M | 12 | 60 | 21 (35%) | 24 (40%) | 0.7–1.4B |
| 0.5–1.5B | 16 | 101 | 31 (31%) | 39 (39%) | 1.5–2.0B |
| > 1.5B | 12 | 106 | 47 (44%) | 49 (46%) | 2.0–4.4B |
| 1B rung, all | 49 | 316 | 98 (31%) | 103 (33%) | |
| 1B rung, < 150M | 12 | 64 | 17 (27%) | 17 (27%) | 172–489M |

Tripling a tail language's tokens lifts its above-chance share by ~4 points
at 1.7B and 0 at 1B. Between 100M and 1.5B tokens the share is flat (27–35%);
it is the model size, not the tokens, that caps it. Caveat: the L100 T=1 tail
(3.5–50M tokens for 37 languages) is below anything measured here; assume it
is at chance on every task, i.e. those languages contribute BPB only.

## What the DA results say

**Per L, benchmarks, mean DA at the proxy's final checkpoint (`early_small_by_L.csv`):**

| pairs | 175M | 350M | 600M | 1B | 1.7B@0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all (schemes A + B, pooled) | 0.56 | 0.64 | 0.68 | 0.71 | 0.81 |
| within L30 (3 pairs) | 0.60 | 0.59 | 0.57 | — | 0.75 |
| within L50 (3 of 6 pairs have data) | 0.54 | 0.65 | 0.64 | 0.63 | 0.81 |
| within L100 (1 planned pair, AT3 deep vs shallow) | — | — | — | — | — |

**Per pair, deep seed 1904 (1.7B effect = mean Δ over ungated tasks; families beyond the ±0.007 seed noise):**

| pair | DA 175M | 350M | 600M | 1B | 1.7B mean Δ | families beyond noise / inside |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A-L8 vs A-L15 | 0.52 | 0.56 | 0.56 | 0.52 | −0.012 | 13 / 5 |
| A-L15 vs A-L30 | 0.53 | 0.65 | 0.59 | 0.63 | −0.006 | 11 / 7 |
| A-L30 vs A-L50 | 0.53 | 0.56 | 0.60 | 0.52 | −0.006 | 7 / 11 |
| A-L50 vs AT3-L50 (T=1 vs T=3) | 0.47 | **0.71** | **0.74** | **0.70** | −0.007 | 7 / 8 |
| A-L50 deep vs shallow | 0.64 | 0.57 | 0.49 | 0.52 | +0.002 | 4 / 11 |
| A-L30 vs B-L30 | 0.55 | 0.52 | 0.53 | 0.54 | −0.002 | 2 / 12 |

Reading:

- **Language-count pairs are the weakly predicted kind.** Each step up the L
  axis moves the 1.7B by ~0.01 on average, mostly inside seed noise, and the
  proxies rank it at 0.52–0.65. An L100 point adds one more pair of exactly
  this kind (A-L50 vs L100, or AT3-L50 vs AT3-L100); expect the same numbers.
- **The temperature pair is predicted like the language-count pairs, not
  better.** Its pooled 0.70–0.74 came from a task mix skewed to the
  predictable families (its 1.7B cell has no reformulated or reading
  comprehension evals yet). On one shared item set (Table T1) the language
  count reaches 0.71–0.74 at 175M–350M where the temperature is at 0.51–0.70,
  and the temperature leads at 600M–1B; depth stays at 0.47–0.63 throughout.
- **The architecture pair is at chance** (0.49–0.64, 1.7B effect +0.002): the
  planned L100 pair (AT3 deep vs shallow) would be the least informative one
  the grid can add. A shallow L100 twin buys almost nothing for DA.
- **The tail is mostly invisible to benchmarks at either temperature.** The
  117 extra L100 tasks sit in languages that get 3.5–90M tokens at T=1 and
  14.5–373M at T=3. From the L50 evidence above, T=3 turns roughly one task
  in twenty-five from chance to signal in those languages; at T=1 the 37
  languages under 50M contribute BPB only. At L15 and L30 the temperature
  changes no language's signal — every language is already above 343M.
- **What L100 does buy:** the "1 to 100" framing the abstract and intro
  already make, the end point of the L axis (does the L30→L50 flattening hold
  at 100?), and per-language BPB curves for 50 more languages.

## Options

Cost basis (deep, `ITER_MS`): 175M 14 · 350M 49 · 600M 111 · 1B 251 · 1.7B 605 = **1,029 node-h per ladder**; evals at 446 tasks (10/10/10/20/30 due checkpoints, 72–98 min each) ≈ 116 node-h + BPB ≈ 55 → **~1,200 node-h per 5-cell ladder**. One build ≈ 10 h on one CPU job. The 1.7B alone is 29 h wall on 21 nodes plus queueing: **no L100 reference lands before 2026-09-25**; 175M–1B can.

| option | builds | models | node-h | buys | costs / caveats |
| --- | --- | ---: | ---: | --- | --- |
| **0. Nothing** | 0 | 0 | 0 | — | paper text becomes six settings, L1–L50, 150 runs; the L axis ends at 50 |
| **1. AT3-L100, deep 1904** (the grid's own cells) | AT3 L100 | 5 | ~1,200 | "1 to 100" as planned; AT3-L50 vs L100 pair; tail BPB (floor 14.5M) | zero wiring (`launch_builds.sh` + `--scheme AT3` already do it); **1.7B repeats the mixture 1.11×** (75.4B < 83.6B) — the only repeated cell in the grid; L100 confounded with T (mitigated by the L50 pair) |
| **2. A-L100 (T=1), deep 1904** | T=1 L100, current list | 5 | ~1,200 | a clean T=1 axis L1…L100; 0.91 epochs at 1.7B; A-L50 vs A-L100 pair | 51 of 99 languages under 100M tokens (their BPB measures an unseen language); ~1 h wiring: `DATA_SCHEMES["A"].langs += 100`, and the build must go to a rebuild root (`rebuild-92B/fineweb_L100`) because the capstor root holds the pre-swap build behind the idempotency guard; the stale 22 GB `.tmp` on the stage needs a decision |
| **3. Options 1 + 2** | both | 10 | ~2,400 | everything above plus the temperature pair at L100 — the only pair kind that is well predicted | 2× cost; the L100 T pair would be the second copy of a result L50 already gives |
| **4. Option 1 or 2 + shallow twin** | 1 | 10 | ~2,400 | the grid's planned L100 pair (deep vs shallow) | that pair is at chance at L50 (DA 0.49–0.64, effect +0.002) |
| **5. Option 2 + seeds 313/64 at 175M and 600M** (the seed-triple policy) | 1 | 9 | ~1,500 | the noise floor at L100, where tail tasks are noisiest | seeds are the colleague's columns at other sizes |
| **6. AT3-L30, deep 1904** (no L100) | AT3 L30 | 5 | ~1,100 | a second temperature pair, where no language is starved (floor 1.3B); evals at 238 tasks; 91.4B realized, no repetition | the L axis still ends at 50 and the paper text changes; `DATA_SCHEMES["AT3"].langs` gains 30 (one line, additive) |
| **7. AT3-L15 + AT3-L30, deep 1904** | 2 | 10 | ~2,200 | temperature pairs at three L (15/30/50): is the T effect and its predictability independent of L? | 2× option 6; L15's pair will sit inside noise if L30's does (floor 3.8B at T=3) |

## Recommendation

Two different goals, two different buys:

- **If the paper keeps "1 to 100 languages":** option 1, the grid's own
  AT3-L100 deep ladder (~1,200 node-h, one build, no code). On benchmarks
  T=3 buys only a few points of tail signal over T=1, but it is the
  registered cell, it keeps 50 more languages' BPB meaningful (floor 14.5M
  vs 3.5M), and the temperature confound is calibrated by the L50 pair.
  State the 1.11 epochs wherever the 1.7B-L100 cell appears. Do not add the
  T=1 twin (option 2 or 3): the L50 evidence says the two would rank the
  same on almost every task.
- **If the paper can say "1 to 50":** option 6, AT3-L30 (~1,100 node-h, one
  build). It replicates the temperature decision where every language has
  signal, with cheaper evals and no repetition, and tests whether its
  predictability follows the size of the token reallocation (see the paper
  paragraphs). Option 7 (L15 too) only if the L30 pair turns out to be
  outside noise at 1.7B.
- **Skip in every case:** the shallow L100 twin (option 4; the arch pair is
  at chance wherever measured) and seeds at L100 (option 5).
- **Timing:** any 1.7B is ~29 h wall on 21 nodes plus queueing and 30 eval
  jobs — none lands before 2026-09-25; 175M–1B can, and they complete the
  BPB curves but give no DA row without the reference.

**Wiring:** option 1 needs nothing (`launch_builds.sh` already submits
`build-at3-L100`; `--scheme AT3 --langs 100` launches). Option 6 is one line
in `DATA_SCHEMES` plus the same build/launch path. Option 2 needs a 92B
rebuild root for the T=1 L100 (the capstor root holds the pre-swap build
behind the idempotency guard) and a decision on the stale 22 GB stage `.tmp`.

## The plan (decided 2026-09-20)

1. **Build AT3 L30 and AT3 L15** (92B each, T=3; realize 91.4B / 91.6B).
   Submitted: `build-at3-L15` job 3449776, `build-at3-L30` job 3449777
   (singleton chains, capstor `AT3/`, staged to `data/AT3/`).
2. **Train AT3-L30 at 1B and 1.7B** (deep, seed 1904) as soon as the L30
   build is staged:
   `python3.11 src/pretrain/launch_trainings.py cscs --scheme AT3 --langs 30 --size 1B --dry-run`
   then `--size 1.7B`, then without `--dry-run` after approval.
3. **Gate:** if the 1.7B (or 1B) AT3-L30 cell is above random on the L30
   tasks where A-L30 is, launch the smaller L30 sizes (175M–600M) and the
   L15 ladder (175M–1.7B). If it is not, stop: the temperature effect at L30
   is inside the gate's resolution and L15 would be too.
4. Wired: `DATA_SCHEMES["AT3"]` has `langs={15, 30, 50}`, `max_size` 1.7B at
   L15/L30 (no 3B) and `arches_by_L` deep-only at L15/L30; `arches_for()`
   takes the cell's L; `LANG_SETTINGS` no longer lists 100;
   `launch_builds.sh` derives the two builds from the registry;
   `configs/models.json` carries the 12 new deep cells; the 32 L100 entries
   (20 T=1, 12 AT3) are stale and listed by `sync_models_json.py --prune --dry-run`.
