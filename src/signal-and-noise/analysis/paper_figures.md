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
- A (benchmark, language, size) cell is *above random* when the mean final score of the size's models beats chance (`1/n_options`) by 0.05.
- Left: share of a benchmark's languages above the gate per size (number = language tasks). Middle: median margin above chance. Right and second figure: the smallest size from which the gate holds at every larger size.
- The gate NaN-s every at-chance SNR cell downstream, so it propagates to every RQ.

Key finding
- Of 462 benchmark tasks, 128 clear chance at some size and 119 at 1.7B; 334 are random at every size. The gate is the first result: most multilingual accuracy benchmarks carry no information below 2B.

Other findings
- MultiBLiMP is above chance in every language from 90M (margin +0.20 to +0.41); XWinograd from 350M; XNLI, XStoryCloze, XCOPA and HellaSwag from 600M in about half their languages.
- Belebele, Global-MMLU, INCLUDE, TruthfulQA and both PIQA clozes never leave chance in any language. Global-PIQA parallel sits *below* chance (−0.3): a format artifact.

Issues / open
- The margin (0.05) is a choice; the answer-count effect (2-option tasks clear it, 4-option tasks do not) should be stated.
- The per-language map is only readable at appendix size.

## rq01 — What scales predictably with model size?

![Median R² per benchmark and L](rq01_scaling_predictability/pretraining/predictivity_all/fit_r2_median.png)

How it is computed
- Per (task, L): final score = a + b·log₁₀N over the rungs trained at that L (deep, scheme A, seed 1904); the cell is the median R² over the benchmark's tasks (number = fits).

Key finding
- Per-language BPB (0.94–0.96), LAMBADA (0.96–0.99), XStoryCloze (0.90–0.96), XWinograd (0.89–0.95) and HellaSwag (0.86–0.94) follow a log-linear law at every L; knowledge and reading benchmarks do not (Belebele 0.13–0.62, Global-MMLU 0.10–0.63, INCLUDE 0.05–0.64).

Other findings
- The unpredictable families are the gated ones: R² collapses once languages are added (best at L1), so it measures noise around chance, not scaling.
- The answer count splits the families: median R² 0.81 (3 options), 0.64 (2), 0.39 (4).

Issues / open
- 90M is excluded (diverged runs); PIQA non-parallel and TruthfulQA rest on 3–4 fits.

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
- 90M never reaches 0.75 and is diverged in several cells; 1.7B's own curve is its DA-ckpt (early checkpoints vs its final).
- The pair set is 28 design variants at most; "safe" with 3 pairs is a weak guarantee.

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

How it is computed
- For each intervention with two levels (depth, language lists A/B, temperature T1/T3, second language ru/zh and ru/es) and L, DA = share of items (per-language BPB of the trained languages, or the benchmark tasks) on which the proxy prefers the level the reference prefers at its final checkpoint. The reference is the largest size trained at both levels at that L; each line keeps only the L's sharing one reference (listed in the note) and averages over them. Ten checkpoints per run.
- Left: the proxy's final checkpoint per size. Right: the reference's own checkpoints vs its final decision.

Key finding
- On BPB every decision is read at 350M: temperature 0.96, both second-language choices 1.0, language lists 1.0 (0.95 at 600M), and the reference's own checkpoints read them from 0.5C. On benchmarks no decision is read at any size or checkpoint (0.45–0.6, a coin flip).

Other findings
- Depth flips: 175M prefers the opposite architecture to the 1.7B/600M reference on every trained-language BPB (DA 0.0), 350M agrees (1.0). Shallow wins small, deep wins large: the one decision a small proxy gets wrong.
- Effect sizes at the reference (`rq4_effect_vs_seed.csv`): temperature 3.9 and second language 3.4 seed sds on BPB, depth 1.5, lists 1.6; benchmarks 1.1–1.7. Where the effect is large the decision is read early.

Issues / open
- The training loss is one item per L, so its dotted line is a 0/0.5/1 step function; drop it from the paper version.
- The 1B point of the list decision (0.65) rests on L8 and L30 with one pair each; the depth 600M point on L2 alone.
- `da_lines_flops.png` (same data at compute) is unreadable with 5 × 3 lines of 60 cells; not for the paper.

## rq06 — Does the decision transfer to languages the proxy did not train? (`predictivity_all`)

![The list decision by language group](rq06_language_transfer/pretraining/predictivity_all/transfer_da_lines.png)

![The decisions by language count](rq06_language_transfer/pretraining/predictivity_all/transfer_da_by_L.png)

How it is computed
- rq05's per-item agreement on the per-language BPB of all 100 evaluation languages, grouped per (intervention, L) by the cell's own lists: the language is trained, only its script is, or neither. First figure: the list decision (A vs B) at L8 and L30 (reference 1.7B), by proxy size and by the reference's checkpoint. Second: every intervention, x = L, DA-size averaged over the proxy sizes.

Key finding
- Trained languages read the list decision at 1.0 from 350M; languages whose script is trained at 0.72–0.77; unseen scripts need 600M (0.72) and the reference's own checkpoints read them at 0.4–0.8 only. Transfer follows the script, not the language.

Other findings
- Scaling-law transfer (rq06 README): one 175M rung plus the exponent pooled over the other languages predicts the 1.7B BPB of a never-trained language within 6.7 % (median), 3.4 % for trained ones.
- Every group's agreement falls with L for the list decision (trained 1.0 → 0.68 from L8 to L30) and the depth decision on trained languages collapses at L30/L50 (0.01, 0.13): the depth crossover of rq05 again.

Issues / open
- Temperature, zh and es have one L each (single points); fold into a table.
- At L1 only English is trained, so "script trained" = every Latin-script language.
- The depth collapse at L30/L50 needs an explanation before publication: a real crossover, or an artifact of the 1.7B L30/L50 cells' completeness.

## Appendix candidates
- rq00 `first_size_above_random.png` and the rq02 `safe_size.png` maps (language × benchmark).
- rq02 `pairs_by_L.csv` (why per-L DA is coarse) and `early_small_by_L.png` once the 6-pair L's are complete.
- The per-benchmark / per-language long grids (`*_by_benchmark.png`, `*_by_language.png`) of rq02 and rq03.
- rq04 `snr_variant_min_size_by_L_flops.png`: ρ of each definition with the cell's DA against compute (rises from ≈ 0.2 to ≈ 0.5; every size restarts the curve because SNR is a property of the size).
