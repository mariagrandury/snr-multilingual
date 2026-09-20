# Paper figures

The figures proposed for the paper, one block per research question: the
question, the figure, how each cell is computed, the key finding, the other
findings, and what is still open. Paths are relative to this folder; every
PNG has its table next to it under the same name. Numbers are from the
`predictivity` pool (seed 1904, schemes A and B, reference 1.7B) unless the
block says `predictivity_all` (every seed and scheme). Written 2026-09-18;
the L15 and L100 cells at 1.7B and the BPB of some 1.7B cells are not final
yet, which blanks every per-L figure at those L.

## rq00 — Which benchmarks carry any signal, and from which size?

![rq00 highlights](rq00_gate_and_curves/pretraining/predictivity/highlights.png)

![Smallest size above chance per language and benchmark](rq00_gate_and_curves/pretraining/predictivity/first_size_above_random.png)

How it is computed
- A run is *confidently above chance* when the **one-sided 95 %** Wilson lower bound of its final accuracy over the task's items (`n_items`, tasks.json) clears chance (`1/n_options`) — `proportion_confint(alpha=0.10)`, whose alpha is the two-sided level; a (benchmark, language, size) cell is above random when at least half of the size's runs are (`above_random.py`, 2026-09-19; before that: the mean beat chance by a fixed 0.05).
- Left: share of a benchmark's languages above the gate per size (number = language tasks). Middle: median margin above chance. Right and second figure: the smallest size from which the gate holds at every larger size.
- The gate NaN-s every at-chance SNR cell downstream, so it propagates to every RQ.

Key finding
- Of 462 benchmark tasks, 180 clear chance at some size and 155 at 1.7B; 282 are random at every size (one-sided 95 % Wilson gate, 2026-09-19; the fixed +0.05 margin gave 128 / 119 / 334, the two-sided-95 % bound 165 / 146 / 297). The test admits the cells the margin refused — mostly HellaSwag, XNLI, XStoryCloze and PAWS, 1–4 points above chance on 500–10 000 items. The gate is still the first result: most multilingual accuracy benchmarks carry no information below 2B.

Other findings
- MultiBLiMP is above chance in every language from 175M (margin +0.29 to +0.41); HellaSwag in 22–24 of 31 languages from 175M; XNLI 13–14 of 18 and XStoryCloze 8–9 of 13 from 350M; XWinograd in every language from 1B; XCOPA in 7–8 of 11 from 600M.
- Global-MMLU, INCLUDE, TruthfulQA and both PIQA clozes stay at chance in (almost) every language; belebele clears it in 5 of 105 languages at 1.7B only. Statistically above chance is not usefully above chance: a HellaSwag cell passes at +0.01 over 10 042 items — an effect floor (e.g. margin ≥ 0.02) on top of the test is the paper's call.
- The reformulation (rq00_task_reformulation, same models for both task sets, 2026-09-19 report): at 1.7B in the trained languages belebele goes from +0.011 to +0.109 over chance (56 of 59 tasks significant), Global-MMLU from −0.010 to +0.048 (26/29), INCLUDE from +0.005 to +0.052 (18/36); the gain grows with size and 38/59 belebele tasks are already significant at 175M. Significance is tested on one metric — the original's `acc` against the rf run's own `acc` — because the rf twins are *scored* with `acc_norm`, whose offset is family-shaped (belebele −0.016, Global-MMLU +0.016, INCLUDE +0.018; median +0.005 overall) and exceeds half the plotted gain in 36 % of the (task, model) pairs.

Issues / open
- rq00 and the reformulation are regenerated under the new gate; the rq01–rq04 numbers below still come from the old fixed-margin one, so regenerate (`run_all_predictivity.sh`, FORCE=1) before quoting them.
- The per-language map is only readable at appendix size.

Measurement issues
- *Fixed — chance is per task, the margin was not.* The old +0.05 was a larger step on a 2-option task than on a 4-option one and ignored the test-set size; the gate is now the one-sided 95 % Wilson lower bound per run against 1/n, so a 500-item 2-option task and a 100-item 4-option task are held to the same evidence. Still open: the runs of a size are not independent draws (they share the test set), so "at least half of the runs" is a majority rule, not a pooled test; `_APPROX` families (TruthfulQA, AGIEval) have a variable option count per item, so for them `random_baseline` is the family table's approximation, not a measured null; and the test has no effect floor — a HellaSwag cell passes at +0.01 over 10 042 items, and 141 of the positives would not survive a Bonferroni bound over the 2 310 cells, so `above_random_thresholds.png` shows what each rule keeps and an effect floor (the gate plus 0.02) remains the paper's call.
- *The cell's population can change along the size axis.* Where a size has cells that trained the benchmark's language the gate reads only those, and where it has none it falls back to every cell of the size; `above_random_share.csv` now carries a `population` column per bucket (1 545 cells `trained`, 765 `all`) so the switch is visible.
- *Formulation decides the gate.* The letter-format families are at chance because of the prompt format, not the knowledge (the rf twins clear the gate at 1.7B). Fix: report the gate for both formulations (the rf evals are running) and make "formulation" a row of the rq00 figure, see "New analyses".

## rq01 — What scales predictably with model size?

![Median R² per benchmark and L](rq01_scaling_predictability/pretraining/predictivity_all/fit_r2_median.png)

![Scaling regimes per benchmark-language pair](rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes.png)

![Appendix: scaling regimes per family, tasks named by language](rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_by_family_paper.png)

How it is computed
- Per (task, L): final score = a + b·log₁₀N over the rungs trained at that L (deep, scheme A, seed 1904); the cell is the median R² over the benchmark's tasks (number = fits).
- Scaling regimes (`regimes.py`): one point per task, medians over the deep scheme-A seed-1904 cells that train its language. (a) R² and Spearman ρ of the log-N fits (one per L; ρ of BPB and the loss negated so improving is positive). (b) the same R² against the R² of score ~ log tokens over each run's checkpoints (one fit per (L, size), ≥ 5 points); quadrants split at R² = 0.5, a heuristic.

Key finding
- Per-language BPB (0.94–0.96), LAMBADA (0.96–0.99), XStoryCloze (0.90–0.96), XWinograd (0.89–0.95) and HellaSwag (0.86–0.94) follow a log-linear law at every L; knowledge and reading benchmarks do not (Belebele 0.13–0.62, Global-MMLU 0.10–0.63, INCLUDE 0.05–0.64).

Other findings
- The unpredictable families are the gated ones: R² collapses once languages are added (best at L1), so it measures noise around chance, not scaling.
- The answer count splits the families: median R² 0.81 (3 options), 0.64 (2), 0.39 (4). Recompute after the PIQA fix: its 91 parallel tasks were counted as 2-option.
- Regimes (264 tasks in trained languages): 113 predictable across both size and training (BPB, HellaSwag, LAMBADA, XStoryCloze, MultiBLiMP, ARC), 52 across size only (PAWS, XWinograd, XCOPA, XNLI: the run-level fit is weaker, median trajectory R² 0.3–0.7), 99 weak in both (Belebele, Global-MMLU, INCLUDE, PIQA), none during training only. TruthfulQA is the one family that scales monotonically the wrong way (ρ −0.82 with R² 0.73).

Issues / open
- 90M is excluded (diverged runs); PIQA non-parallel and TruthfulQA rest on 3–4 fits.

Measurement issues
- *R² over five points does not measure predictability.* Any monotone curve through 175M–1.7B gets a high R², and an at-chance family gets the R² of its noise. Fix: report the leave-top-out extrapolation error — fit 175M–1B, predict 1.7B, |relative error| — which `scaling_law_error.py` already computes for BPB (`scaling_law_error.png`); extend it to the gated benchmark tasks and make it the paper figure, with R² in the appendix.
- *The gate is not applied.* The low-R² families are the at-chance ones, so the figure partly restates rq00. Fix: grey the cells whose task is at chance at 1.7B (`grids.mark_gated`), as every other RQ does.

## rq02 — How early and how small can the 1.7B ranking be read?

![rq02 highlights](rq02_decision_accuracy/pretraining/predictivity/highlights.png)

![Smallest safe proxy size per language and benchmark](rq02_decision_accuracy/pretraining/predictivity/safe_size.png)

How it is computed
- Decision accuracy (DA) = share of pairs of design variants (L × arch × scheme, seed 1904) that a proxy orders like the 1.7B final checkpoint. DA-size reads the proxy's final checkpoint; the early-and-small grid reads it at 1C–5C of its own run (C = Chinchilla-optimal 20 tokens/parameter; every run trains 5C).
- Left: mean DA over the gated tasks, solid BPB, dashed benchmarks. Middle: DA-size per benchmark family. Right two: per (benchmark, language), the smallest size / fewest FLOPs at which DA ≥ 0.75 over ≥ 3 pairs and stays so at every larger level ("safe").

Key finding
- Per-language BPB reads the 1.7B ranking at DA ≥ 0.75 from 175M at 2C (0.76) and from 350M at 1C (0.77); benchmarks need 1B at 2C (0.77). BPB is the cheap signal; accuracy benchmarks are not.

Other findings
- BPB DA-size: 0.80 / 0.81 / 0.85 / 0.85 from 175M / 350M / 600M / 1B. Benchmarks: 0.63 / 0.73 / 0.72 / 0.75.
- `bpb_macro` (mean over the 100 languages) is the best single proxy: 0.86 / 0.90 / 1.00 / 0.93; the training loss 0.84 / 0.91 / 0.85 / 0.87.
- HellaSwag (0.90 at 350M), XStoryCloze (0.79–0.89) and ARC (0.81 at 175M) are the benchmark exceptions; MultiBLiMP, the only ungated benchmark everywhere, reads the ranking at 0.70–0.73 only.
- Safe size per (benchmark, language): never 75, 175M 55, 1B 29, 600M 27, 350M 22 of 208 cells; for BPB most languages are safe from 175M–350M.

Issues / open
- 90M never reaches 0.75 and is diverged in several cells (dropped at load from the next run); 1.7B's own curve is its DA-ckpt (early checkpoints vs its final).
- The pair set is 28 design variants at most; "safe" with 3 pairs is a weak guarantee.

Measurement issues
- *DA is not conditioned on the size of the effect.* Most pairs of design variants differ in L or in list membership, so on a language's BPB one variant saw the language and the other did not: those pairs are ordered correctly by any proxy, and they dominate the BPB DA of 0.80–0.85 (cross-task shows it: 0.81–0.87 on a language's own BPB, 0.57–0.62 across languages). Fix: per pair, the reference's gap in seed sds (rq05 `seed_sd`); report DA on the pairs above 2 sds, and DA against the gap as a line per proxy size. rq05 does this now (`da_lines_decided`, below), rq02 does not yet.
- *No noise ceiling and no intervals.* With 11 variants a DA moves in steps of 1/55, and 0.75 against 0.80 is within resampling noise. Fix: bootstrap over design variants (not tasks, which share the variants) for a 90 % interval on every line and cell; and a ceiling — the DA between two seeds of the same variants at 1B (the 1B ×3 cells) — drawn as a band, since no proxy can beat the reference's own seed-to-seed agreement.
- *Early checkpoints are un-annealed.* Runs use WSD with a 1-sqrt cooldown over the last ~20 %, so every checkpoint up to 4C is read at the peak learning rate while every final is annealed; DA-ckpt compares the two regimes and understates what an early *annealed* proxy would read. Fix: mark the annealed points (≥ 4C and finals) in the line and FLOPs figures; to measure the cost of the missing anneal, branch short cooldowns (the 20 % decay) from the 1C and 2C checkpoints of a few 350M/600M cells and re-read DA there.

### rq02, per language count

![Early and small per L](rq02_decision_accuracy/pretraining/predictivity/early_small_by_L.png)

How it is computed
- Same DA, but the pairs are restricted to the design variants that share the L (`predictivity_all`, seed 1904, every scheme), on the ten evaluated checkpoints (0.5C–5C). A panel needs ≥ 3 pairs; the first panel pools every pair of schemes A and B.
- `pairs_by_L.csv`: planned pairs against 1.7B per L and size, vs pairs with data today (BPB / benchmarks / loss).

Key finding
- Only L30 has ≥ 3 usable pairs today (3 of 6 planned), and there the benchmarks stay at 0.4–0.6 while 1.7B's own early BPB checkpoints climb to 0.77 at 4.5C. L1, L2 and L100 will never have more than one pair against 1.7B (ZH/ES stop at 1B); L8, L15, L30 reach 6 pairs when scheme-B shallow finishes at 1.7B, L50 with AT3.

Issues / open
- Not a paper figure until the 6-pair L's are complete; keep the pairs table (appendix) to justify why per-L DA is coarse.

### rq02, cross-task predictability

![Cross-task DA-size by benchmark](rq02_decision_accuracy/pretraining/predictivity/cross_task_size_by_family.png)
![Cross-task DA-ckpt by benchmark](rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt_by_family.png)

Full task × task maps: `cross_task_size.png`, `cross_task_ckpt.png` (569 parent tasks; x = proxy task, y = target task).

![Cross-task DA-size by language](rq02_decision_accuracy/pretraining/predictivity/cross_task_size_by_language.png)
![Cross-task DA-ckpt by language](rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt_by_language.png)

How it is computed
- Every parent task x as the proxy for every other task y: DA of x's ranking of the design variants against y's final ranking (`predictivity`, seed 1904, schemes A/B). The diagonal is rq02's own-task DA (checked, identical).
- DA-size: x at each proxy size (final checkpoint) vs y at 1.7B final, 11 variants = 55 pairs; cell = smallest size that clears 0.75 and keeps clearing it at every larger size.
- DA-ckpt: x at each of the ten checkpoints vs y's final at the same size, within-size pairs pooled over 175M–1.7B (406 pairs); cell = earliest checkpoint (Chinchilla multiple) from which it holds.
- Gate on both sides (x at chance at its size, y at chance at the size it is ranked at) and ≥ 3 pairs; the `_by_family` maps take the median level over the task pairs of two benchmarks (rounded up) and the CSV adds the share that never reach it.
- `_by_language`: language × language over the pairs of tasks of the same benchmark (bpb_x → bpb_y, arc_x → arc_y, …), axes in the resource order of the scheme-A lists (English, then L100; the lines mark where the L2 … L100 lists end).

Key finding
- Cross-task predictability is low: 2.4 % of task pairs reach a safe size (1.1 % a safe checkpoint) against 68 % of the non-gated diagonal. Between two BPB languages the median DA is 0.57–0.62 at every proxy size (own language 0.81–0.87): the design variants are language mixes, so the ranking on one language is not the ranking on another by construction.
- Where cross-task reading works it is late and large: median 1B / 4C for benchmark → benchmark cells; the only sub-1B block is BPB → BPB (median 350M, 1.5C) and hellaswag/xnli → hellaswag/xnli (600M).
- High-resource languages do not predict low-resource ones: a higher-resource proxy reaches a safe size for 7.2 % of the language pairs, a lower-resource proxy for 6.9 % (DA-ckpt 4.1 % vs 4.2 %). English and Russian, trained by every variant, predict almost nothing (their ranking is about capacity, every other language's about inclusion).
- What does transfer is membership in the same L-list: languages that enter the mixture at the same L (the blocks between the lines) predict each other (DA-size 13–18 % of pairs reached inside the L8/L15/L30 blocks vs 0–8 % across; DA-ckpt 16–23 % vs ≤ 11 %), because the same design variants train both. Within the L50/L100 tail nothing predicts anything (≤ 9 %).

Issues / open
- 84 % of the cells are gated (belebele, mmlu, piqa, include, truthfulqa at chance), so the figure mostly says which tasks have a ranking at all; the task-level maps are appendix material, the family maps could carry the "language mixes do not transfer across languages" point in the text.
- With 55 pairs a cell's DA moves in 1/55 steps, but 11 variants of 5 language counts make "never" the honest answer for most cross-language pairs, not a resolution artefact.
- Belebele, MMLU, PIQA, INCLUDE and TruthfulQA are at chance at 1.7B for every language (rq00 mask), so their final ranking is noise and no proxy can predict it: their rows are grey by construction, not because the proxies are small.

Measurement issues
- *The family maps show the median of the pairs that succeed.* A cell reads "1B" when half of the few task pairs that ever reach a safe level do so at 1B, while most pairs never do; the share that never does is only in the CSV. Fix: draw the share of task pairs that reach a safe size (or the median DA at a fixed proxy size, e.g. 600M) as the cell, and keep the median level as the small number.
- *Same-L blocks share the design variants.* Two languages that enter at the same L are trained by the same variants, so their rankings agree for the same reason a language agrees with itself; the "same L-list transfers" finding is partly by construction. Fix: compare against a permutation baseline — the cross-task DA of two languages whose L-lists are shuffled — before calling it transfer.

## rq03 — Where is the signal above the noise?

![rq03 highlights](rq03_noise_and_snr/pretraining/predictivity/highlights.png)

How it is computed
- SNR (`rel_std`) per (task, size) = relative spread of the final scores across the design variants / relative std over the last 5 checkpoints of each variant (the checkpoint noise). Cells the gate filters out are left out.
- Left: median log₁₀ SNR per family × size (number = tasks). Middle/right: benchmarks and languages ranked by median log₁₀ SNR at 1.7B.

Key finding
- BPB has the most signal per unit of noise at every size (log₁₀ SNR 1.6 at 175M, 0.5 at 1.7B); HellaSwag (0.81), XStoryCloze (0.69), ARC and LAMBADA (0.55) lead the benchmarks at 1.7B; MultiBLiMP (0.25) and PAWS (−0.03) are at or below noise.

Other findings
- SNR of BPB *falls* with size (1.6 → 0.5) while benchmark SNR rises (HellaSwag 0.7 → 0.8–1.0): the design variants converge in BPB as models grow.
- Languages: Finnish, Czech, Hungarian, Polish above 1.0; Amharic, Lithuanian, Gaelic, Kazakh, Kurmanji below 0: their benchmarks are noise.
- Seed noise vs detrended checkpoint noise: median ratio 1.88 over 7870 cells; the depth effect is 1.37 seed sds at the median, 33 % of cells above 2×.

Issues / open
- The checkpoint-noise window (last 5 of 20/40/60 saves) is a convention; the seed replicates (rq03 `effect_vs_noise`) are the cleaner noise but exist only for the ×3 cells.

Measurement issues
- *The noise window sits on the cooldown.* On the shared k/10 benchmark grid the last five evaluated checkpoints are 3C–5C, which span the whole WSD cooldown (the last ~20 %), so "checkpoint noise" is mostly the annealing trend, and more so for BPB, which moves smoothly with the learning rate. Whether that trend, rather than the variants converging, is what makes BPB's SNR fall from 1.6 to 0.5 with size is untested. Fix: use the detrended checkpoint noise (residual sd around a per-run linear or LR-aware fit) for the headline, or the seed sd where the ×3 cells exist; show the raw window as a sensitivity row.
- *Signal mixes interventions.* The spread across design variants in the `predictivity` pool mixes L, depth and the scheme, so a task can have high "signal" only because it separates L1 from L100. Fix: an SNR per intervention (the spread between the two levels of one decision over its seed sd — rq05's `effect_at_reference`), which is the SNR that should predict that decision's DA.

## rq04 — Which cheap statistic predicts decision accuracy?

![rq04 highlights](rq04_surrogates/pretraining/predictivity/highlights.png)

![SNR definitions vs DA per proxy size and population](rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_b.png)

How it is computed
- 22 SNR definitions (upstream's variants) per (task, size). Highlights: Pearson r of log₁₀ SNR with DA over tasks, all sizes pooled; a language counts with ≥ 5 tasks.
- Second figure: Spearman ρ per proxy size, benchmarks and per-language BPB apart; DA-size = proxy final vs 1.7B final, DA-ckpt = the size's own early checkpoints (mean over the nine before the last) vs its final.

Key finding
- SNR predicts *within-run stability* (DA-ckpt) at every size (ρ 0.7–0.8 for the relative-dispersion family) but *cross-size agreement* (DA-size) only from 600M (ρ −0.02 at 175M, 0.26 at 350M, 0.45 at 600M for `rel_std` on benchmarks).

Other findings
- The definitions are a family: rel_std, rel_mpd, rel_dispersion, iqr, aad within 0.05 of each other; tukey, projection, discrepancy ≈ 0 on benchmarks. Recommend the family, not an exact variant (the seed holdout ranking has ρ 0.07 on DA-ckpt).
- On BPB the picture is the same with `iqr` on top (0.49 DA-size at 600M, 0.78–0.80 DA-ckpt).
- Best single surrogate of DA-size is not an SNR at all: early-checkpoint agreement (20 % of the proxy's own run) has ρ 0.36 on benchmarks and 0.51 on BPB; the margin above chance predicts nothing (−0.09).

Issues / open
- The seed holdout shows DA-ckpt variant rankings do not transfer (ρ 0.07), so per-variant differences are noise.
- Per-L SNR correlations (`snr_variant_min_size_by_L.png`) are all "never" for DA-size: per-L DA is binary with ≤ 3 pairs. Drop from the paper; the L-dependence of SNR needs SNR computed over the L's own variants, degenerate with fewer than 3.

Measurement issues
- *The highlights pool populations and sizes.* Pearson r over every (task, size) mixes BPB with benchmarks and small with large proxies, so it partly measures which population and which size a definition favours (Simpson's paradox). Fix: the per-population, per-size Spearman ρ of the `_b` figure is the paper figure; drop the pooled bars or restrict them to one population.
- *The definitions are not separable.* The top six are within 0.03 of each other and the seed-holdout ranking has ρ 0.07. Fix: bootstrap the ρ over tasks (a 90 % interval per definition) and claim the family ("relative dispersion over checkpoint noise"), not a winner; drop the "languages in which a definition is best" panel, which counts noise.
- *The compute axis joins un-annealed and annealed points.* In `min_level_by_L_flops` a bigger model's first (peak-LR) checkpoint is placed next to a smaller model's annealed final, which explains the zigzags near 10⁻². Fix: one line per proxy size with the annealed points marked, plus the compute frontier (the best DA reachable at or below each compute), which is the practical answer; see "New analyses".

### rq04, how much compute reads the decision

![Earliest checkpoint per measurement and size](rq04_surrogates/pretraining/predictivity/min_level_by_L_b.png)

![DA against training compute](rq04_surrogates/pretraining/predictivity/min_level_by_L_flops.png)

How it is computed
- Every pair of the pool, ten checkpoints per run. First figure, left: per measurement (training loss, `bpb_macro`, per-language BPB mean, benchmark mean, each family) and proxy size, the earliest checkpoint at which DA vs the 1.7B final ranking is ≥ 0.75 and stays so; right: the DA at 1C. Second figure: the same DA with every (size, checkpoint) cell at the compute it has spent, as a share of the 1.7B run.

Key finding
- Loss, `bpb_macro` and per-language BPB read the 1.7B ranking at DA ≥ 0.75 from 0.5C (10 % of the run) at 350M, 600M and 1B; at 175M they need 2C (`bpb_macro`) or 4C. In compute: `bpb_macro` clears 0.75 at ≈ 0.6 % of the 1.7B run and stays there; the benchmark mean approaches 0.75 only at ≥ 30 %.

Other findings
- DA at 1C: `bpb_macro` 0.81 / 0.95 / 0.86 / 1.00 (175M → 1B), per-language BPB 0.70–0.84, benchmark mean 0.63–0.72.
- HellaSwag is the one benchmark that behaves like BPB (0.5C from 350M); XStoryCloze needs 4C at 350M/600M; ARC, XCOPA, XNLI, XWinograd are non-monotone or never.

Issues / open
- The FLOPs line joins cells of different sizes: the zigzags near 10⁻² are a bigger model's first checkpoint scoring below a smaller model's last. One line per size, or per-size markers, for the paper version.
- Benchmark families panel is unreadable; keep the aggregates panel only.

## rq05 — Would a small proxy make the same design decision as the reference? (`predictivity_all`)

![Decisions by proxy size and by checkpoint](rq05_design_decisions/pretraining/predictivity_all/da_lines.png)

![The same, on the items the reference decides outside seed noise](rq05_design_decisions/pretraining/predictivity_all/da_lines_decided.png)

![Which depth wins, in seed sds](rq05_design_decisions/pretraining/predictivity_all/depth_crossover.png)

How it is computed
- For each intervention with two levels (depth, language lists A/B, temperature T1/T3, second language ru/zh and ru/es) and L, DA = share of items (per-language BPB of the languages both levels train, or the benchmark tasks) on which the proxy prefers the level the reference prefers at its final checkpoint. The reference is the largest size trained at both levels at that L — 600M for depth, temperature and zh/es, 1.7B for the lists — and each line keeps only the L's sharing one reference (listed in the note) and averages over them. Ten checkpoints per run.
- Left: the proxy's final checkpoint per size. Right: the reference's own checkpoints vs its final decision.
- `da_lines_decided`: the same DA on the items whose reference |Δ| between the two levels is ≥ 2 seed sds (seed sd per task = median over the replicated baseline cells); a cell needs 3 such items.
- `depth_crossover`: deep − shallow final BPB per size × L, on the languages the scheme-A list trains, in seed sds (median over languages; < 0 = deep wins).

Key finding
- Most decisions on the shared languages are not decisions at the reference: its |Δ| clears 2 sds of the two-run difference (√2 × the seed sd) on **0 of 206** trained-language items for depth and **0 of 142** for the lists. Only temperature has a real effect there (82 of 100 items), and every proxy reads it (DA 0.92, and 1.00 on the decided items).
- Benchmarks read no decision at any size or checkpoint, and restricting to the items the reference decides does not help (0.33–0.50 over the five interventions, on 7 521 decided items): the benchmarks' failure is not noise at the reference.

Other findings
- Depth is a vanishing advantage, not a crossover: deep beats shallow by 3.9–10.1 difference sds at 175M, by |z| ≤ 0.4 at 350M, and at the 600M reference shallow is ahead by ≤ 1.1 sds — inside noise. The DA of 0.0 at 175M and 0.03–1.0 at 350M read a reference that has no real preference.
- On every language's BPB (`bpb_all`, which includes the languages only one level trains) the decided items are read well: lists 0.96, ru-vs-zh 0.95, ru-vs-es 0.98, temperature 0.91 (mean over proxy sizes) against 0.72–0.85 on all items — but that population is dominated by "the model that saw the language wins" (rq06).
- Effect sizes at the reference (`rq4_effect_vs_seed.csv`): temperature 3.9 and second language 3.4 seed sds on BPB, depth 1.5, lists 1.6; benchmarks 1.1–1.7. Where the effect is large the decision is read early.

Issues / open
- The training loss is one item per L, so its dotted line is a 0/0.5/1 step function; drop it from the paper version (already dropped from `da_lines_decided`).
- The 1B point of the list decision (0.65) rests on L8 and L30 with one pair each; the depth 600M point on L2 alone.
- The ru-vs-zh and ru-vs-es decisions have no shared-language population (the two levels share only English), so they are read on BPB of every language or the macro BPB only.
- `da_lines_flops.png` (same data at compute) is unreadable with 5 × 3 lines of 60 cells; not for the paper.

Measurement issues
- *Items are not independent.* A language's BPB items move together (a better model is better on every language), so 37 languages agreeing is closer to one decision measured 37 times than to 37 decisions. Fix: report the number of *decisions* (intervention × L) that agree, with the item share as a secondary number, and bootstrap over L, not items.
- *The reference changes between lines.* Depth, temperature and zh/es are read against 600M, the lists against 1.7B, and a 1.7B depth or temperature decision may differ. Fix: once the 1.7B shallow and AT3 cells finish, re-read every line against 1.7B; until then name the reference in the legend, not only in the note.
- *The seed sd comes from the baseline cells.* It is the median over the deep scheme-A cells with replicates (175M, 600M, 1B), applied to every size and scheme, and each cell's own sd rests on 3 seeds (the median-of-sd is biased low by ~17 %). The |Δ| of two runs is compared against √2 × that sd. Fix: where the ×3 cells exist at the reference's size, use that size's sd, and widen DECIDED to cover the sd's sampling error.

## rq06 — Does the decision transfer to languages the proxy did not train? (`predictivity_all`)

![The list decision by language group](rq06_language_transfer/pretraining/predictivity_all/transfer_da_lines.png)

![The decisions by language count](rq06_language_transfer/pretraining/predictivity_all/transfer_da_by_L.png)

How it is computed
- rq05's per-item agreement on the per-language BPB of all 100 evaluation languages, grouped per (intervention, L) by what the two levels' lists do with the language: both train it, only one does, neither does but one trains its script, or neither trains even the script. The last two groups are the transfer test; "only one" is the decision the language's own inclusion makes. First figure: the list decision (A vs B) at L8 and L30 (reference 1.7B), by proxy size and by the reference's checkpoint. Second: every intervention, x = L, DA-size averaged over the proxy sizes.

Key finding
- The decision does not transfer to untrained languages at a proxy's scale: on the list decision, languages whose script a list trains are read at 0.61 / 0.72 / 0.74 / 0.77 (175M / 350M / 600M / 1B) and unseen scripts at 0.47 / 0.33 / 0.72 / 0.73 — neither clears 0.75 below 1B. The reference's own checkpoints hold 0.75 on them only from 4C (script trained) and 4.5C (unseen script).
- The earlier "trained languages read it at 1.0" was the languages only one list trains, where the model that saw the language wins at every size (1.0 throughout). The languages both lists train are read at 0.52 / 1.0 / 1.0 / 0.65.

Other findings
- Script helps a little, and only small: script-trained is ahead of unseen-script by 0.14 at 175M and 0.39 at 350M, level from 600M on.
- Scaling-law transfer (rq06 README): one 175M rung plus the exponent pooled over the other languages predicts the 1.7B BPB of a never-trained language within 6.7 % (median), 3.4 % for trained ones.
- Agreement falls with L for the list decision (both-trained 1.0 → 0.67 → 0.59 from L8 to L30, mean over proxy sizes). The depth decision on both-trained languages at L30/L50 (0.02, 0.13) is rq05's depth result: the 600M reference prefers shallow within seed noise (+0.9 and +1.4 sds) while 175M prefers deep by 14 and 5.5 sds, so the proxies disagree with a reference that has no real preference.

Issues / open
- Temperature, zh and es have one L each (single points); fold into a table.
- At L1 only English is trained, so "script trained" = every Latin-script language.
- The "mean over proxy sizes" of `transfer_da_by_L` averages a proxy that flips with one that agrees into 0.5; the per-size values are in `transfer_da_lines.csv`. Use one line per proxy size, or the largest proxy below the reference, in the paper version.

Measurement issues
- *Script is a coarse proxy for relatedness.* Latin covers Welsh and Vietnamese alike. Fix: group the untrained languages by the trained tokens of their language family / genus (from the data manifest), or by tokenizer overlap with the trained languages, and plot DA against that exposure instead of three bins.
- *Same caveats as rq05:* items are correlated and the reference is 1.7B only for the lists; bootstrap over L and read the other interventions against 1.7B when their cells exist.

## Appendix candidates
- rq00 `first_size_above_random.png` and the rq02 `safe_size.png` maps (language × benchmark).
- rq02 `pairs_by_L.csv` (why per-L DA is coarse) and `early_small_by_L.png` once the 6-pair L's are complete.
- The per-benchmark / per-language long grids (`*_by_benchmark.png`, `*_by_language.png`) of rq02 and rq03.
- rq04 `snr_variant_min_size_by_L_flops.png`: ρ of each definition with the cell's DA against compute (rises from ≈ 0.2 to ≈ 0.5; every size restarts the curve because SNR is a property of the size).
- rq05 `depth_crossover.png` (which depth wins per size × L, in seed sds) and `da_lines_decided.png`, if the main text keeps only `da_lines`.

## New analyses

Implemented 2026-09-18 (rq05/rq06 regenerated; rq00 reformulation regenerated from the 2026-09-18 evening report):
- **Effect-size-resolved decisions** — rq05 `decision_acc_decided` / `n_decided` in `intervention_da.csv` and `da_lines_decided.png`: DA on the items the reference decides outside 2 seed sds. Result in rq05 above.
- **Depth crossover map** — rq05 `depth_crossover.png`: deep − shallow BPB per size × L in seed sds.
- **Clean language transfer** — rq06 groups split into trained by both levels / by one level / script only / neither (`analyze.language_group`).
- **Fair reformulation comparison** — `rq00_task_reformulation/compare.py` reads both task sets on the same (model, task) pairs.

To do (not quick; each needs new computation or new runs):
- **rq02 effect-size-resolved DA** — the rq05 treatment for pairs of design variants: per pair the reference's gap over the pooled seed sd, DA against the gap per proxy size. Needs per-pair seed sds from the ×3 cells.
- **Seed-noise ceiling** — DA between two seeds of the same design variants at 1B (seeds 1904/28/1797), drawn as a band on the rq02 and rq05 lines. `SEED_TRIPLES` gives 4 variants (deep scheme-A L1/L2/L30/L50) × 3 seeds at 1B, so 6 pairs per seed pair; 3 variants at 175M and 600M.
- **Bootstrap intervals** — over design variants for rq02, over L for rq05/rq06, over tasks for the rq04 ρ. A shared helper in `analysis/utils.py`, then one call per figure.
- **Formulation as a variable** — for belebele, Global-MMLU and INCLUDE, the first size above chance, DA-size and SNR per formulation (letters vs rf cloze), once the rf evals cover the ladder (today: deep scheme-A seed 1904, no 1B yet). Answers "which formulation makes a benchmark usable at 1B".
- **Compute frontier** — in `min_level_by_L_flops` (rq04) and `da_lines_flops` (rq05), one line per proxy size with annealed points marked, plus the running best DA at or below each compute. A drawing change in two `panels.py` files.
- **Cooldown branches** — short WSD cooldowns branched from the 1C/2C checkpoints of a few 350M/600M cells, to separate "early" from "un-annealed" in every DA-ckpt read. Costs training compute; a planning decision.
- **Cross-task maps as shares** — draw the share of task pairs reaching a safe size (the CSV already has it) instead of the median level, plus a shuffled-L-list permutation baseline for the same-L-block finding.
- **rq01 extrapolation error** — extend `scaling_law_error.py` from BPB to the gated benchmark tasks and make it rq01's paper figure.
- **Relatedness instead of script** — rq06 transfer against the untrained language's exposure (trained tokens of its family/genus, or tokenizer overlap) rather than three bins.
