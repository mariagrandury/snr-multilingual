# The 600M depth pair is not size-matched: shapes, retrain cost, options

*2026-10-09. Decision doc for audit finding F5
([`signal-audit-2026-10-08.md`](signal-audit-2026-10-08.md)). Numbers:
`src/pretrain/hyperparams/hyperparams_{deep,shallow}.json`, recomputed with
`calculate_params_lr_bs.parameter_count` (the same function the generator uses);
node-hours from `sacct` (elapsed × nodes, 4 GPUs/node, both users) and the
`pretrain-600M-*.out` logs under
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training/`.
No job was submitted, no code or data was changed.*

## The problem

At 600M the shallow model (14 × 2048, heads 32/8) is not size-matched to the
deep one (24 × 1536, heads 24/6):

- non-embedding N 616.6M against 594.5M (**+3.7 %**);
- width 2048 against 1536, so the tied embedding is 33 % larger;
- with D = 100 N and C = 6 · (N + d·V) · D (V = 131,072, tied), it spends
  **+15.3 % compute** (3.27e20 against 2.84e20);
- 14 layers, the same depth as shallow 350M.

Shallow beats deep on `bpb_dclm` in 9 of 10 settings at 600M and loses at every
other size. On per-language BPB the depth-pair DA-size is 0.95 / 0.72 / **0.04** /
0.89 at 175M / 350M / 600M / 1B. This is the "600M dip"
([`decision_accuracy.md`](decision_accuracy.md) §8.7). Today the pairs are kept
and flagged (`utils.UNMATCHED_PAIRS`, `analysis/RULES.md` "Size matching"), and
the paper says so (`03_methodology.tex` l. 58, `04_analysis.tex` l. 139–140).

## Why it happened

Commit `2f69d742` (2026-08-21, "Deconfound the depth intervention",
[`team-discussion-2026-08-21.md`](team-discussion-2026-08-21.md) §5) pinned
FFN = 4d and q/kv = 4 so that depth is the only thing that moves. The new rule in
`find_hyperparams_shallow.py` is: d a multiple of 256, width/depth in [96, 160],
**N within ±4 % of the deep target, then the ratio closest to 128**. It matches N
only. Compute was never checked, and a shallow model is wider, so its embedding
term d·V is larger at equal N. At 600M four candidates pass the ±4 % N window and
14 × 2048 has the ratio closest to 128 (146), so it won with +15 % compute. The
team-discussion table reported "N vs deep target" (+3.7 %, inside ±5 %) and no
compute row, so nobody saw it.

The same effect is milder at every other size: the shallow ladder spends
+2.6 % to +8.9 % more compute everywhere (table below). Matching on
non-embedding N is the ladder's convention (OLMo / Signal-and-Noise), so those
are accepted. 600M is the outlier at about twice the others.

## The ladder: ratios and contrast

Contrast = shallow width/depth ÷ deep width/depth.

| size | deep (L × d) | deep ratio | shallow (L × d) | shallow ratio | contrast | ΔN | ΔC |
|---|---|---:|---|---:|---:|---:|---:|
| 90M | 15 × 768 | 51.2 | 8 × 1024 | 128.0 | 2.50× | −5.2 % | +8.9 % |
| 175M | 16 × 1024 | 64.0 | 10 × 1280 | 128.0 | 2.00× | −2.3 % | +6.9 % |
| 350M | 20 × 1280 | 64.0 | 14 × 1536 | 109.7 | 1.71× | +0.8 % | +8.0 % |
| **600M** | 24 × 1536 | 64.0 | **14 × 2048** | **146.3** | **2.29×** | **+3.7 %** | **+15.3 %** |
| 1B | 28 × 1792 | 64.0 | 17 × 2304 | 135.5 | 2.12× | +0.4 % | +6.4 % |
| 1.7B | 30 × 2304 | 76.8 | 20 × 2816 | 140.8 | 1.83× | −0.4 % | +2.6 % |

The contrast spans **1.71×–2.50×** (`models.md`). Interpolating 350M and 1B,
600M "should" sit near 1.9×.

### 600M candidates

Every shape inside the script's constraints with N within ±4 %, plus two for
reference. All have head_dim 64 (fixed: heads = d / 64) and FFN = 4d.

| candidate (L × d) | ratio | contrast | heads q/kv | N | ΔN | ΔC | iters (D = 100 N) |
|---|---:|---:|---|---:|---:|---:|---:|
| 14 × 2048 (trained) | 146.3 | 2.29× | 32/8 | 616.6M | +3.7 % | +15.3 % | 29,860 |
| 13 × 2048 | 157.5 | 2.46× | 32/8 | 572.5M | −3.7 % | +1.8 % | 27,740 |
| **17 × 1792** | **105.4** | **1.65×** | **28/7** | 573.2M | −3.6 % | −2.1 % | 27,760 |
| 18 × 1792 | 99.6 | 1.56× | 28/7 | 606.9M | +2.1 % | +8.0 % | 29,400 |
| 15 × 1920 (pre-Aug-21) | 128.0 | 2.00× | 30/7.5 ✗ (ran 30/10) | 589.8M | −0.8 % | +4.9 % | — |

- **15 × 1920** is the shape from before 2026-08-21. It is the best match on
  every count, but 30 heads give 7.5 KV groups at q/kv = 4, so it ran at
  q/kv = 3 (10 KV heads), and d = 1920 is not a multiple of 256. Bringing it
  back reopens the KV-head confound that `2f69d742` removed, so it is out.
- **Is 157.5 a weird ratio?** Not by the rules: it is inside [96, 160], and its
  contrast (2.46×) is inside the ladder's own range (90M is 2.50×). It is odd in
  two other ways. It sits at the edge of the window and jumps away from both
  neighbours (350M 1.71×, 1B 2.12×). And it makes shallow 600M (13 layers)
  shallower than shallow 350M (14), so depth stops increasing along the
  shallow ladder. It fixes compute by removing a layer, not by moving the shape
  toward 128.
- **Why not 17 × 1792?** There is no rule against it. **The "1.7× minimum" is not
  a constraint.** Neither `find_hyperparams_shallow.py` nor
  `hyperparams_shallow_explanation.txt` mentions one. The phrase "contrast ≥ 1.7×
  at every size" appears only in the pros column of option A in
  `team-discussion-2026-08-21.md` §5, as a description of the shapes the rule
  produced (the minimum happened to be 350M's 1.71×). At 1.65× the 600M contrast
  would be the ladder's lowest, 0.06 below 350M's.

### Recommendation: 17 × 1792

| criterion | 13 × 2048 | 17 × 1792 |
|---|---|---|
| N match | −3.7 % | −3.6 % (tie) |
| compute match | +1.8 % | −2.1 % (tie) |
| heads (head_dim 64, q/kv 4) | 32/8, integer | 28/7, integer (the deep 1B's layer) |
| contrast vs neighbours 1.71× / 2.12× (≈ 1.9× interpolated) | 2.46×, 0.56 off | 1.65×, 0.25 off |
| shallow-ladder depth 8, 10, 14, ?, 17, 20 | 13: decreases after 350M | 17: never decreases (ties 1B) |
| shallow-ladder width 1024 … 2816 | 2048: increases | 1792: increases |
| the script's own tie-break (ratio closest to 128) | 29.5 away | **22.6 away: wins** |

On size and compute the two candidates are equivalent. **17 × 1792** wins on
everything that is about the ladder's shape: it keeps the depth and contrast
lines smooth, and it is what the existing tie-break picks once the trained
shape is excluded. Its only cost is that 1.65× is the weakest depth contrast in
the ladder, 3 % below 350M. That should be stated, not avoided.

It also falls out of a principled rule rather than a hand override. Adding
**"and compute within ±10 % of deep"** to the ±4 % N window reproduces every
other rung unchanged and selects 17 × 1792 at 600M. Any window from ±8 % to
±15 % gives the same ladder. ±4–5 % would also move 1B to 21 × 2048, so don't
use it. (Checked by re-running the selection loop over all six sizes.)

New config: 17 layers, hidden 1792, FFN 7168, 28 heads / 7 query groups,
N = 573.2M, `train_iters` 27,760, warmup 1,100, WSD decay 5,600, lr 9.06e-4
(6ND law). The micro-batch is hand-tuned in the JSON and needs a value; the
deep 1B uses the same layer at MBS 6.

## What a retrain costs

### Measured: the cells as trained

| | cells | jobs | node-h | per cell | ms/iter (median) | wall/job |
|---|---:|---:|---:|---:|---:|---:|
| shallow 600M (14 × 2048), 21 nodes | 10 | 10 | **1,005** | 92.8–111.3 (mean 100.5) | 524–553 | 4:25–5:18 |
| deep 600M (24 × 1536), 21 nodes | 22 | 36 | 2,294 | 104 incl. preemptions; 93–108 clean | 545–549 | 4:25–5:07 |
| swiglu + muon 600M (deep shape) | 6 | 30 | 697 | ~116 incl. preemptions | — | — |

Each shallow cell finished in a single job. mariagrandury's six ran at 93–99
node-h each, and aromanou's four (L8/L15/L30/L50, 2026-09-01, run in parallel)
at 107–111. This agrees with [`compute-budget.md`](compute-budget.md): 539 ms/iter,
~4.4 h/run, 21 nodes ≈ 92 node-h, and with
[`compute-costs.md`](compute-costs.md): 600M pretrain kept 3,991 node-h across
all 38 cells.

Eval cost per cell, measured on the same cells: benchmark evals 12 node-h
(shallow, kept) to 22 (deep, more high-L cells and seeds), plus BPB on all 20
checkpoints at ~1,176 s each (6.5 node-h), plus ~0.5 for conversions. That is
**~20–30 node-h per cell.** The analysis reads 12 of the 20 checkpoints
(`due_iters`); BPB scores all 20.

### Estimate for the new shape

Scale the measured 100.5 node-h/cell by compute (0.849 for 17 × 1792, 0.882 for
13 × 2048) or by steps at a deep-like ms/iter (27,760 / 28,800 × 92.8). Either
way the result is **85–90 node-h per cell** for both candidates. At 600M on
84 GPUs the runs reach only 12–20 TFLOP/s/GPU and do not scale with FLOPs: the
+11 % FLOPs/iter shallow cell runs *faster* per iteration than deep. So the step
scaling is the more realistic number.

| scope | cells | training | evals + BPB + convert | total |
|---|---:|---:|---:|---:|
| **(a) shallow 600M only** | 10 | ~850–900 | ~200–300 | **~1.1–1.2k node-h** |
| (a′) pilot: L8 + L50 shallow first | 2 | ~180 | ~50 | ~0.23k node-h |
| (b) all 600M, deep + shallow | 32 | ~2,900–3,200 | ~650–950 | **~3.6–4.1k node-h** |
| (b′) (b) plus swiglu + muon | 38 | ~3,600–3,900 | ~800–1,100 | ~4.4–5.0k node-h |

For scale: the sweep has charged 49.8k node-h to date (`compute-costs.md`), so
(a) is ~2.4 % of it and ~25 % of what the 600M rung has kept.

The 10 shallow cells, by name (`launch_trainings.predictivity_cells` with
`ladders_for`; shallow is seed 1904 only, `seeds_for`):

```
lm-600M-L1-shallow-seed1904          lm-600M-L50-AT3-shallow-seed1904
lm-600M-L2-shallow-seed1904          lm-600M-L8-schemeB-shallow-seed1904
lm-600M-L8-shallow-seed1904          lm-600M-L15-schemeB-shallow-seed1904
lm-600M-L15-shallow-seed1904         lm-600M-L30-schemeB-shallow-seed1904
lm-600M-L30-shallow-seed1904
lm-600M-L50-shallow-seed1904
```

The 22 deep cells are scheme A at L1–L50 (seeds 64/313/1904 at L1, L2 and L50),
AT3 at L15/L30/L50, schemeB at L8/L15/L30, ZH and ES at L2, and dclmP and fweb
at L1. Retraining them (scope b) **buys nothing on this problem**: deep is the
size-matched side. It would also replace the 600M seed triples, the noise
denominator of rq03. Scope (b) only makes sense if something else at 600M is
being changed too.

### Wall clock

Each cell is one 21-node job of ~4.5–5.3 h, which fits a single 12 h segment.
The precedents are four 600M jobs at once on 2026-09-01 (no queue wait) and four
on 2026-09-22 (1.5–4 h queue wait).

- **(a)** at 4 concurrent: 3 waves, ~16 h plus queueing ≈ **1 day of
  training**. At 10 concurrent (210 nodes) it is ~6–9 h. The auto-eval watcher
  trails by ~0.5–1 day (1-node eval jobs, and ~6.5 h of BPB per cell), so about
  **1.5–2 days end to end**.
- **(a′)** about 6–8 h of training, ~1 day end to end.
- **(b)** at 4–6 concurrent: 6–8 waves, ~1.5–2 days of training, **~2.5–3 days**
  end to end.

## What a retrain implies

**Old and new cells must never be mixed.** They are different models under the
same design levels. Cell names stay as trained (checkpoint dirs, W&B ids,
models.json keys and HF repos all exist under the old names), so the new shape
needs its own name. The checkpoint-dir hazard is the same one the 90M/175M
retrain met ([`90M-175M-batch-retrain.md`](90M-175M-batch-retrain.md)): under
the old name the launcher would *resume* the 14 × 2048 checkpoint into a
17 × 1792 model.

Naming, two options:

1. **A shape tag in the `-b<N>` slot (proposed):**
   `lm-600M-L8-17x1792-shallow-seed1904`. This is the 2026-09-23 precedent.
   `mix_label` adds the tag only when given a size, so the analysis **family**
   (`snr/download/ladder.py`: `mix_label(L, ladder, data)` without size) stays
   `lm-L8-shallow-seed1904`, and the new 600M point joins the shallow family at
   every other size. Needed: an optional `(?:-(?P<shape>\d+x\d+))?` group in
   `pretrain_progress.NAME_RE` and `ladder_report` CELL_RE / LOG_RE, and an
   `on_grid` check on it so the old cells drop out the way the batch-504 runs
   did. The old shape must also stay readable: models.json already carries
   `hyperparams_key`, so the old entries can point at the preserved
   14 × 2048 config (e.g. key `600M-14x2048`). `ladder.cell_params(size,
   ladder)` must then read that key, not the size.
2. **A new ladder token** (e.g. `shallow2`, a `HYPERPARAMS` / `LADDERS` entry).
   `NAME_RE` matches it automatically, but it breaks the cross-size family: the
   600M cell would be `lm-L8-shallow2-…`, which pairs with no other size, so
   every 600M depth pair would silently fall out of DA-size. It also makes
   `ladder_of()` ambiguous, because two ladders share the levels
   `arch=shallow, xielu, ademamix`. Not recommended.

The old cells stay on disk and in models.json (never deleted), out of the
analyses by the filter. Make the edits in a worktree, because a live edit to the
launcher can end running chains (2026-10-05).

**What changes in the analysis and the paper** once the new cells are evaluated:

- `utils.UNMATCHED_PAIRS` loses its `("arch", "600M")` entry, so `size_matched`
  flags and `size_match_note` captions disappear. `analysis/RULES.md` "Size
  matching" is rewritten to describe the retrain.
- **rq02 DA-size**: the 600M per-language BPB point (0.74 in
  `04_analysis.tex` l. 139, vs 0.92–0.96 at the other proxies) and the
  `bpb_dclm` depth line (0.20).
- **rq05 depth**: the per-language BPB depth line (600M = 0.00 at L8–L50) and
  the benchmark depth line (600M mean 0.47). The sentence "the depth comparison
  shows no similarly consistent transfer" (`04_analysis_gen.tex` l. 22) needs
  re-checking.
- **rq03 flip attribution** (`tab:app_rq03_flip_attribution`, its caption, and
  `04_analysis.tex` l. 140 and 231).
- `decision_accuracy.md` §8.7, and the 600M row of the shallow table in
  `app_01_model_ladder.tex` (l. 100) and `models.md`.
- The flag sentences in `03_methodology.tex` l. 58 and
  `07_responsible_nlp_research.tex` l. 10, plus the contrast range everywhere
  it is quoted. It becomes **1.65×–2.50×** with 17 × 1792; it stays
  1.71×–2.50× with 13 × 2048.
- Every pool that reads the 600M shallow cells as models (predictivity pools,
  rq04 surrogates, rq01 scaling fits of the shallow ladder). Regenerate with
  `run_all_predictivity.sh`.

**A retrain may not remove the dip.** The paper's own attribution
(`04_analysis.tex` l. 140) finds most 600M depth differences below one seed SD,
and the deep 600M seed-1904 run is the least favourable of its three seeds:
seeds 64 and 313 flip 15 and 43 of 50 languages, against all 50. At the 1.7B
reference the depth effect is only 1.3 seed SD on BPB and 1.1 on benchmarks
(rq05). So the confound is real (+15 % compute), but size may be only part of
the 600M result. The pilot (a′) tests exactly that. L50 has the deep seed
triple to read the new shallow cell against. If `bpb_dclm` flips back to
"deep wins", as at every other size, the full retrain is justified. If not, the
dip is mostly noise, and option 2 or 1 is the honest reading.

## Options

| option | cost | pros | cons |
|---|---|---|---|
| **1. Flag only** (status quo) | 0 | Already implemented (`UNMATCHED_PAIRS`, RULES.md, paper sentences). No regeneration. | The 600M point of every depth line is "depth plus a little size" and cannot be read. Keeps drawing attention and reviewer questions. |
| **2. Exclude the 600M depth pairs from DA** | ~1 h code + a regeneration pass | Removes the confound from every DA number without compute. A small change: `size_matched` already identifies the pairs, so it is a filter, not a new mechanism. | Loses the 600M rung on the depth axis (proxies 90M–350M and 1B remain), including multi-axis pairs that move depth at 600M. Changes a RULES.md rule. Can read as hiding an inconvenient point. |
| **3. Retrain the 10 shallow 600M cells at 17 × 1792** | ~1.1–1.2k node-h, ~1.5–2 days | Removes the confound at its source. The ladder becomes compute-matched at 600M (−2.1 %, the best of any rung). The depth line is complete, and the result is clean whatever it shows. | Compute, plus a full analysis regeneration and paper-number changes. Naming and filter code (above). 1.65× becomes the ladder's weakest contrast. |
| 3′. Pilot first: L8 + L50 shallow | ~0.23k node-h, ~1 day | Tells whether size explains the flip before spending the rest. | Two cells cannot carry a DA line on their own. The other 8 follow if it pays. |
| 4. Retrain all 600M (deep too) | ~3.6–4.1k node-h (~5k with swiglu/muon), ~3 days | — | Deep is the matched side, so this fixes nothing extra and replaces the 600M seed triples. Not recommended unless 600M changes for another reason. |

Recommended path: **3′ then 3** at 17 × 1792, with option 2 as the fallback if
the pilot shows the flip is noise. Whatever happens, the generator rule in
`find_hyperparams_shallow.py` should gain the ±10 % compute window, so that
`hyperparams_shallow_explanation.txt` documents why 600M changed.

## Decision still open

- Option: 1 / 2 / 3′→3 / 4.
- Shape if retraining: 17 × 1792 (recommended) or 13 × 2048.
- Naming: shape tag `-17x1792` in the `-b<N>` slot (recommended) or a new ladder
  token.
- Who launches the 10 cells (4 of the originals were aromanou's), and when,
  relative to the paper revision.
- Per standing practice, nothing is submitted without an explicit go: dry-run
  the launcher, show the job list, then wait for a yes.
