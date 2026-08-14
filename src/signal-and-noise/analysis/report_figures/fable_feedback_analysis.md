# Analysis

Here is the full Analysis section. Numbers come from the repo (pool `custom_swissai_hf` for headlines, `seeds_28_1797_1904` where noted). Figure files are referenced as `figures/fig*.pdf` to match the generation prompts below. Two caption values are marked `XX` because they must come out of the regenerated figures, each has a `% TODO` comment.

```latex
\section{Analysis}
\label{sec:results}

We organize the analysis around four questions, namely which benchmarks are
measurable at all on small multilingual models
(\S\ref{subsec:exp-da}), which SNR definition best predicts decision accuracy
(\S\ref{subsec:exp-snr-da}), whether the resulting ranking generalizes across
seeds and model suites (\S\ref{subsec:exp-generalization}), and whether
benchmark subsets and design choices can improve reliability
(\S\ref{subsec:exp-benchmark-design}).

\subsection{Calculation of Decision Accuracy}
\label{subsec:exp-da}

\textbf{Checkpoint selection.} For each custom and open-source model, we
evaluate 10 evenly spaced checkpoints across training and 5 additional
checkpoints in the final 10\% of training FLOPS. This schedule captures both
the long-range trajectory and the late-training variation. DA by checkpoint
uses early checkpoints at 12\%, 36\%, and 56\% of each model's training,
expressed as relative fractions so that external models with different
absolute schedules can participate.

\textbf{The above-random gate.} Before computing SNR, we filter
(benchmark, size) cells whose mean score does not exceed the random baseline
by at least 5 points. Figure~\ref{fig:gate} illustrates the two regimes. Of
118 benchmarks, only 44 clear chance at one or more custom model sizes, and
74 are at chance everywhere. The failures are almost entirely an answer-count
effect, since four-option translated knowledge tasks sit at chance across all
custom sizes while two-option tasks clear the gate. Notably, the three
benchmarks that separate the data mixtures most are exactly the ones the gate
removes. Their apparent signal is variation around chance, not capability.
This shows that raw score separation cannot guide benchmark selection on its
own.

\textbf{Extension with external models.} Many tasks are at chance for the
custom models, which are trained on 200 languages with only 100B tokens and
at most 1B parameters. We therefore extend the analysis with 68 external
models from 270M to 70B parameters (Appendix Table~\ref{tab:external}).
External sizes pool into buckets so that each bucket holds at least two
models, and buckets contribute to signal and decision accuracy while
singleton sizes only appear on the accuracy curves. With capable models in
the pool, additional benchmarks clear the gate and enter the analysis.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_gate.pdf}
\caption{The above-random gate. Accuracy against training compute for the
three data mixtures (colored curves) with final checkpoints marked. Left, a
benchmark that clears the gate (\texttt{multiblimp\_rus}) shows scores well
above the random baseline (dashed line) and consistent mixture separation.
Right, a gated benchmark (\texttt{belebele}) fluctuates around its baseline,
so any apparent separation between mixtures is noise. A (benchmark, size)
cell is kept iff its mean score exceeds the baseline by 5 points.}
\label{fig:gate}
\end{figure}

\subsection{Definition of SNR}
\label{subsec:exp-snr-da}

\textbf{Which SNR definition predicts decision accuracy.}
Table~\ref{tab:snr-variants} reports the mean Pearson correlation between
log SNR and decision accuracy across languages for representative variants
of the 22 we compare, grouped into five families. The dispersion and
relative-spread families lead under DA by checkpoint, with correlations up
to 0.51, while no variant exceeds 0.32 under DA by size. The depth and
projection families never help, with correlations at or below zero.
Figure~\ref{fig:snr-da} shows the relation for the best variant. This
confirms the central premise of the framework in the multilingual setting,
since benchmarks with higher SNR make more reliable development decisions,
and it identifies which definitions carry that relation.

\begin{table}[t]
\centering
\small
\caption{Mean Pearson correlation between $\log_{10}$ SNR and decision
accuracy across languages, for representative SNR variants on the
\texttt{custom\_swissai\_hf} pool (3 seeds plus external pretraining models).
DA-size compares small to large models, DA-ckpt compares early to final
checkpoints. Bold marks the best value per column.}
\label{tab:snr-variants}
\begin{tabular}{llccc}
\toprule
Variant & Family & DA-size $r$ & DA-ckpt $r$ & Overall \\
\midrule
\texttt{dist\_std}                  & dispersion       & \textbf{0.32} & 0.43 & \textbf{0.38} \\
\texttt{rel\_mpd}                   & relative spread  & 0.11 & \textbf{0.51} & 0.31 \\
\texttt{rel\_std}                   & relative spread  & 0.11 & 0.50 & 0.31 \\
\texttt{star\_discrepancy\_shifted} & discrepancy      & 0.14 & 0.15 & 0.15 \\
\texttt{gini}                       & dispersion       & 0.13 & 0.12 & 0.13 \\
\texttt{tukey}                      & depth            & 0.05 & 0.22 & 0.14 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.6\textwidth]{figures/fig2_snr_vs_da.pdf}
\caption{SNR against decision accuracy by checkpoint at the 1B target, one
point per above-random benchmark, colored by language. Higher SNR predicts
higher decision accuracy
% TODO: fill r from regenerated figure
($r = $ XX). Extreme points are labeled.}
\label{fig:snr-da}
\end{figure}

\textbf{Per-language reliability.} The most reliable benchmark differs by
language. Table~\ref{tab:reliability-map} reports the highest-SNR
above-random benchmark per language with its decision accuracy, and
Figure~\ref{fig:reliability-map} shows the full benchmark-by-language map.
MultiBLiMP is the most reliable benchmark in six of eleven languages, with
decision accuracy between 0.79 and 0.87, which matches the expectation that
grammatical acceptability emerges early and therefore discriminates small
models. Swahili retains no above-random benchmark on the custom models, so
no reliable small-scale decision can be made for it at this scale. This
confirms that benchmark reliability is language dependent and motivates the
per-language reliability map as the practical output of the framework.

\begin{table}[t]
\centering
\small
\caption{Per-language reliability map. For each language, the
above-random benchmark with the highest SNR (\texttt{dist\_std} at 1B) and
its decision accuracy by checkpoint at the 1B target, on the
\texttt{custom\_swissai\_hf} pool. Swahili has no benchmark above random on
the custom models.}
\label{tab:reliability-map}
\begin{tabular}{llcc}
\toprule
Language & Top benchmark & SNR & DA-ckpt \\
\midrule
ar & \texttt{multiblimp\_arb} & 2.65 & 0.87 \\
en & \texttt{xwinograd\_en}   & 2.40 & 0.83 \\
es & \texttt{multiblimp\_spa} & 3.37 & 0.85 \\
eu & \texttt{multiblimp\_eus} & 1.28 & 0.64 \\
hi & \texttt{multiblimp\_hin} & 4.95 & 0.85 \\
ja & \texttt{xwinograd\_jp}   & 2.28 & 0.76 \\
ru & \texttt{multiblimp\_rus} & 7.08 & 0.86 \\
th & \texttt{xnli\_th}        & 1.28 & 0.75 \\
tr & \texttt{multiblimp\_tur} & 2.75 & 0.79 \\
vi & \texttt{xcopa\_vi}       & 1.61 & 0.76 \\
zh & \texttt{xcopa\_zh}       & 1.57 & 0.61 \\
sw & none above random        & --   & --   \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_reliability_map.pdf}
\caption{Benchmark reliability map. Decision accuracy by checkpoint at the
1B target for every benchmark family and language. Grey cells fail the
above-random gate. Practitioners can read off which benchmarks support
small-scale decisions for their target languages.}
\label{fig:reliability-map}
\end{figure}

\subsection{Framework Generalization}
\label{subsec:exp-generalization}

\textbf{Generalization across seeds.} We test whether the variant ranking
survives a change of training seed by fitting on two seeds and holding out
the third. The variant ranking transfers under DA by checkpoint, with a
holdout Spearman correlation of 0.81, and does not transfer under DA by
size, with a Spearman correlation of -0.07. The DA-size correlations are
small and clustered within roughly 0.1 of each other, so their ranking is
noise dominated. The exact best variant per language never transfers either.
We therefore recommend a metric family rather than a single variant, and we
treat DA-by-size variant rankings as unreliable at this scale.

\textbf{Agreement with the English SNR framework.} We compare our SNR values
against those of \citet{heineman_signal_2025} on the DataDecide corpus.
Seven English benchmarks are shared between the two model suites, of which
four survive the above-random gate. Over these four tasks, SNR values
computed on our custom models and on DataDecide agree closely, with a
Pearson correlation of 0.98 on log SNR and a Spearman rank correlation of
1.00 under the \texttt{dispersion\_shifted} variant. With so few shared
tasks this is indicative rather than conclusive, but it suggests the
framework measures a property of the benchmark rather than of the model
suite. A caveat applies to MMLU, since our suite runs the Cohere translation
of the English split while DataDecide runs the original, so this row is not
strictly like for like. We plan to rerun the original MMLU task on our
checkpoints to remove the alias.

\subsection{Improvement of Benchmark Quality}
\label{subsec:exp-benchmark-design}

\textbf{Subtask subsets raise SNR.} Per benchmark, we rank subtasks
(languages within a family, or subjects within MMLU) by standalone SNR and
sweep cumulative subsets. Figure~\ref{fig:subsets} shows the sweep for MMLU
subjects. A subset of one to two subjects matches or beats the full set of
roughly 48 subjects across model sizes, for example raising SNR from 2.12 to
3.65 at 175M with a single subject, and the same subjects recur across sizes
and languages. The largest gain raises Vietnamese MMLU from 2.05 to 4.01
with three subjects. This gives practitioners drop-in replacements for full
benchmark runs during ablations, at a fraction of the evaluation cost.

\textbf{Per-item selection overfits.} The same procedure at the level of
individual items produces even larger apparent gains, but the selected items
do not generalize. The best item subsets barely overlap across model sizes,
with a Jaccard overlap of 0.03 and an SNR rank correlation of 0.05. We
conclude that subset selection is reliable at the subtask level and not at
the item level, and we report item-level subsets only as an upper bound.

\textbf{What makes a benchmark reliable.} Among the benchmarks that survive
the gate, no single design feature predicts SNR. Family-level
Kruskal-Wallis tests on curation method ($p = 0.78$), task format
($p = 1.00$), and answer-option count ($p = 0.18$) are all far from
significance, in part because the gate leaves too little variation, since
most survivors are two-option tasks. The durable pattern instead sits
upstream of SNR, in the answer space. Every at-chance benchmark is a
four-option translated knowledge task, and a benchmark is sharper when the
model compares fewer and longer scored completions, since each extra option
adds another noisy likelihood estimate to rank. This suggests that benchmark
builders targeting small-scale evaluation should reduce the answer space
before refining curation.

\begin{figure}[t]
\centering
\includegraphics[width=0.7\textwidth]{figures/fig4_subset_sweep.pdf}
\caption{Cumulative subset sweep for Global MMLU subjects. Combined SNR as
subjects are added in order of standalone SNR, one line per model size, with
the full-set SNR as horizontal dashed lines. Small subsets match or beat the
full set
% TODO: confirm best_n per size from regenerated figure
(best subset size XX). Per-item selection (Appendix) yields larger gains
that do not transfer across sizes.}
\label{fig:subsets}
\end{figure}
```

Two notes on what I changed relative to your current draft. The new-languages extrapolation paragraph is gone from 4.3 because the de/fr rows are empty in the repo, it stays in Future Work where you already have it. And the seed holdout moved from 4.2 into 4.3 because it is a generalization result, which also gives 4.3 substance beyond the thin AllenAI comparison.

---

## Prompts for the Claude Code session

Paste the context block first, then one figure prompt at a time. Each prompt is self-contained enough to survive a fresh session.

**Context block (paste once at session start):**

```
Repo: mariagrandury/snr-multilingual, branch refactor/shared_config.
Run `git lfs install && git lfs pull` after cloning, all CSVs are LFS objects.
Data lives under src/signal-and-noise/analysis/.

Task: create a script src/signal-and-noise/analysis/report_figures/make_figures.py
that generates 4 publication figures as PDF into
src/signal-and-noise/analysis/report_figures/figures/.

Global style for all figures:
- matplotlib only, no seaborn. Single shared style block.
- Font size 9 for labels, 8 for ticks and legends. Font: DejaVu Sans.
- No figure titles (captions live in LaTeX). No top/right spines.
- Colorblind-friendly palette (matplotlib tab10 or Okabe-Ito).
- Sizes: full-width figures 5.5 x 2.6 in, single-column 3.4 x 2.8 in.
- Save with bbox_inches="tight", dpi irrelevant (vector PDF).
Git workflow: start from dev, merge main into dev, create branch
dev_report_figures from dev, commit the script and figures, push, open a PR
to dev. Update README.md with one line describing the new script.
Reuse existing loading logic from the rq* modules where possible instead of
reimplementing parsers.
```

**Prompt for Figure 1 (gate illustration):**

```
Figure 1: fig1_gate.pdf, full width, 1x2 panels sharing the y-axis label
"Accuracy".

Data: the per-task accuracy trajectories used by
analysis/rq00_acc_vs_flops (pool seeds_28_1797_1904). Reuse its loading
code (run_apertus.py) to get accuracy vs FLOPs per data mixture for the
custom models, seed 1904 only.

Left panel: multiblimp_rus. Right panel: belebele (English aggregate,
belebele_eng_Latn). For each panel:
- x: training FLOPs, log scale. y: accuracy.
- One curve per data mixture (90/10, 60/40, 30/70), smoothed with a
  centered rolling mean over 3 checkpoints. Label mixtures in a legend in
  the left panel only.
- Mark final checkpoints of each model size with an "x" marker and annotate
  sizes 175M and 1B only.
- Horizontal dashed grey line at the task's random baseline. Read the
  baseline from rq00_acc_vs_flops/pretraining/seeds_28_1797_1904/
  above_random_scores.csv (columns n_options, random_baseline).
- Identical y-limits across panels so the gap above baseline is comparable.
Sanity check before saving: multiblimp_rus mean final score must be well
above its baseline, belebele must hug 0.25. If belebele_eng_Latn is not in
the seed-1904 trajectories, fall back to agieval_sat_en and tell me.
```

**Prompt for Figure 2 (SNR vs DA scatter):**

```
Figure 2: fig2_snr_vs_da.pdf, single column size.

Data: rq02_snr_definition/pretraining/custom_swissai_hf/
snr_variants_per_task.csv (single source of truth, per-task SNR for every
variant and size bucket plus DA columns).

Plot: x = log10 of SNR for variant rel_mpd at the 1B bucket, log-scaled
axis with plain SNR tick labels. y = decision accuracy by checkpoint at the
1B target, fraction 56 (column matching decision_acc_ckpt_f56_1B, check
exact name in the header first). One point per above-random benchmark task.
- Color points by language (parse the language suffix from the task name,
  aggregate parents without a language suffix as "en" only if that matches
  how rq02 treats them, otherwise drop them and tell me).
- OLS fit line with a 95% bootstrap confidence band in light grey.
- Compute Pearson r and its standard error, print them to stdout, and place
  "R = ..., R^2 = ..." in the top-left corner.
- Label only the 3 highest-SNR and 2 lowest-DA points with the task name at
  8pt.
Report the r value back to me, it goes into the LaTeX caption.
If rel_mpd at f56 has fewer than 25 points after the gate, also try dist_std
and rel_std and pick the variant with the most points, then tell me which.
```

**Prompt for Figure 3 (reliability heatmap):**

```
Figure 3: fig3_reliability_map.pdf, full width, height scaled to the number
of benchmark families (roughly 0.18 in per row plus margins).

Data: same snr_variants_per_task.csv as Figure 2 for the gate mask and
DA values, pool custom_swissai_hf.

Build a matrix: rows = benchmark families (arc, belebele, global_mmlu_full,
global_piqa_completions, hellaswag, multiblimp, paws, truthfulqa, xcopa,
xnli, xstorycloze, xwinograd, plus any others present), columns = the 12
languages en, es, ru, hi, zh, ja, ar, vi, tr, th, sw, eu. Cell value =
decision accuracy by checkpoint at the 1B target (f56) for that family and
language, NaN if the (family, language) task does not exist or fails the
above-random gate.

Render with imshow:
- Sequential colormap (viridis) from 0.5 to 1.0, since DA below 0.5 is
  worse than chance. Clip below 0.5.
- Gate-failed cells in light grey with a small "x", missing tasks left
  white. Add both to a small legend.
- Annotate each colored cell with its DA value at 7pt, white or black
  depending on background luminance.
- Sort rows by row-mean DA descending so the most reliable benchmarks sit
  on top.
- Colorbar on the right labeled "Decision accuracy (ckpt, 1B)".
Sanity checks: the sw column must be entirely grey or white for
custom-model-only families, and multiblimp rows should be among the top.
```

**Prompt for Figure 4 (subset sweep):**

```
Figure 4: fig4_subset_sweep.pdf, single column size, slightly wider
(4.2 x 2.8 in).

Data: rq04_smooth_subtasks outputs for case "global_mmlu_full_subjects"
(MMLU subject subsets, scores averaged over the 10 global_mmlu languages),
pool custom_swissai_hf. Reuse the sweep logic from smooth_subtasks.py
rather than re-deriving it, the cumulative subset SNR must match the
repo's summary CSVs exactly.

Plot: x = number of subjects included, in order of standalone SNR
(1..N, linear). y = combined SNR.
- One line per model size (175M, 350M, 600M, 1B), tab10 colors, legend
  inside the axes.
- For each size, a horizontal dashed line in the same color at the
  full-set SNR, thin (0.8pt) and at 50% alpha.
- Mark the maximum of each curve with a filled circle.
- Print to stdout, per size: best_n, best SNR, full-set SNR, gain. These
  must match rq04's summary.csv rows for global_mmlu_full_subjects. If they
  do not match, stop and show me the discrepancy instead of saving.
Report best_n per size back to me for the LaTeX caption.
```

After the session finishes, give me the printed r from Figure 2 and the best_n values from Figure 4 and I will fill the two `XX` placeholders in the captions.

# Analysis II

- DA: I would like to tell based on the external suite, which are the most reliable benchmarks per language (only based on DA, which is then what we want to find)
- SNR: the pool custom swissaihf doesnt have high correlation, do you think this one is better than just focusing on external? or maybe recalculating for all (since custom-swissai-hf only evals pretraining)
- generatliazation: perfect, ignore from now on
- improvement: A) Per subtask: No one is going to evaluate only on one language or a couple of subjects, these are not supposed to tell users to eval only on vietnamese, but the results show that full benchmarks are noisy and invites to dive deeper and think about how redundant the benchmarks are. B) Per sample:the commment is too negative, we do find nice subsets. C) there are not many benchmark that survive, focus here on the all/external pool

All three points are answerable from the repo. I computed the external DA table directly, here is everything.

## DA: most reliable benchmarks per language, external suite

I aggregated `rq01/all/external/da_per_task.csv` over the small-to-large size pairs (small in {270M, 600M, 1B, 1.7B}, large in {7-9B, 12-14B, 27-32B}, 9 pairs per task), since that is exactly the decision the framework targets. The winners:

| lang | top benchmark    | DA   | runner-up   | DA   |
| ---- | ---------------- | ---- | ----------- | ---- |
| en   | xwinograd        | 0.81 | xstorycloze | 0.76 |
| es   | hellaswag        | 0.97 | xstorycloze | 0.76 |
| ru   | xstorycloze      | 0.99 | hellaswag   | 0.95 |
| hi   | xstorycloze      | 0.78 | multiblimp  | 0.74 |
| zh   | xstorycloze      | 0.97 | xcopa       | 0.76 |
| ja   | paws             | 0.90 | xwinograd   | 0.83 |
| ar   | xstorycloze      | 0.98 | xnli        | 0.81 |
| vi   | xcopa            | 0.93 | hellaswag   | 0.73 |
| tr   | global_mmlu_full | 0.92 | xnli        | 0.66 |
| th   | belebele         | 0.90 | xnli        | 0.34 |
| sw   | belebele         | 0.57 | xcopa       | 0.56 |
| eu   | hellaswag        | 0.90 | truthfulqa  | 0.71 |

This is a much better story than the SNR-based table. XStoryCloze leads in four languages and is near the top in three more, and Swahili gets a value now but a damning one: even with capable external models, its best benchmark transfers rankings barely better than a coin flip. Replacement paragraph and table:

```latex
\textbf{Per-language reliability.} Table~\ref{tab:reliability-map} reports,
for each language, the benchmark whose small-model rankings best transfer to
large models on the external suite. For each benchmark and language we
average decision accuracy over all small-to-large size pairs, with small
models up to 1.7B and large models from 7B to 32B parameters.
Human-translated narrative and commonsense tasks dominate. XStoryCloze is
the most transferable benchmark in four languages and close behind in three
more, with decision accuracy up to 0.99, while machine-translated knowledge
tasks rarely lead. The most reliable benchmark differs by language, and for
Swahili no benchmark exceeds 0.57, barely above a random decision. This
confirms that benchmark reliability is language dependent and that a single
universal benchmark choice leaves some languages effectively unmeasured.
```

```latex
\begin{table}[t]
\centering
\small
\caption{Per-language reliability map on the external suite (68 open-source
models, 270M to 70B). For each language, the benchmark with the highest
decision accuracy averaged over the nine small-to-large size pairs (small
$\leq$ 1.7B, large 7B to 32B). DA is the fraction of model pairs whose
small-scale ranking holds at large scale.}
\label{tab:reliability-map}
\begin{tabular}{llclc}
\toprule
Lang & Top benchmark & DA & Runner-up & DA \\
\midrule
en & \texttt{xwinograd}         & 0.81 & \texttt{xstorycloze} & 0.76 \\
es & \texttt{hellaswag}         & 0.97 & \texttt{xstorycloze} & 0.76 \\
ru & \texttt{xstorycloze}       & 0.99 & \texttt{hellaswag}   & 0.95 \\
hi & \texttt{xstorycloze}       & 0.78 & \texttt{multiblimp}  & 0.74 \\
zh & \texttt{xstorycloze}       & 0.97 & \texttt{xcopa}       & 0.76 \\
ja & \texttt{paws}              & 0.90 & \texttt{xwinograd}   & 0.83 \\
ar & \texttt{xstorycloze}       & 0.98 & \texttt{xnli}        & 0.81 \\
vi & \texttt{xcopa}             & 0.93 & \texttt{hellaswag}   & 0.73 \\
tr & \texttt{global\_mmlu\_full}& 0.92 & \texttt{xnli}        & 0.66 \\
th & \texttt{belebele}          & 0.90 & \texttt{xnli}        & 0.34 \\
sw & \texttt{belebele}          & 0.57 & \texttt{xcopa}       & 0.56 \\
eu & \texttt{hellaswag}         & 0.90 & \texttt{truthfulqa}  & 0.71 \\
\bottomrule
\end{tabular}
\end{table}
```

One honesty requirement: this aggregation does not exist as a script in the repo yet, I computed it ad hoc. The last prompt below adds it so the table is reproducible.

## SNR: which pool to report

My recommendation is to report both pools side by side rather than recalculate a merged one, and here is the reasoning from the numbers. On the external pool, the DA-ckpt correlation holds at essentially the same level as on custom_swissai_hf: the dispersion and relative-spread families reach r of 0.43 to 0.45 (mpsd 0.45, dispersion/mpd/range 0.44, rel_std 0.43), versus 0.43 to 0.51 on the custom pool. That replication across two very different model populations is itself a result, and it is stronger evidence than either number alone. What collapses on the external pool is DA-size, with a maximum of 0.20, and the reason is structural rather than a defect: cross-bucket pairs there compare different model families (Qwen against Gemma against OLMo), which differ in data, tokenizer, and recipe, not only in size, so the ranking-transfer question is confounded. On the custom pool DA-size is confounded differently, by the undertrained 1B target, which is why the repo TODO already proposes Apertus-8B as a converged target.

Against merging into one "all" pool: signal means different things in the two populations. On the custom pool it is spread across controlled data mixtures, on the external pool it is spread across heterogeneous models in a size bucket. Pooling them would average two different quantities and blur both. So: external is the headline pool for the report, because it spans pretraining and post-training models and populates Swahili, and custom_swissai_hf stays as the controlled-mixture complement. Replacement paragraph for 4.2:

```latex
\textbf{Which SNR definition predicts decision accuracy.}
Table~\ref{tab:snr-variants} reports the mean Pearson correlation between
log SNR and decision accuracy across languages, on the external suite and on
the custom pool. The two populations differ in what signal measures, spread
across controlled data mixtures for the custom models and spread across
heterogeneous open-source models for the external suite, yet the result
replicates. Under DA by checkpoint, the dispersion and relative-spread
families reach correlations of 0.43 to 0.45 on the external suite and 0.43
to 0.51 on the custom pool, while depth and projection variants never help.
Under DA by size the correlation is weak in both pools, for structural
reasons. External size pairs compare different model families that differ in
data and tokenizer, not only in size, and the custom 1B target is not fully
converged. We therefore base benchmark selection on DA by checkpoint, and we
recommend a metric family rather than a single variant.
```

Update Table 2 accordingly: keep the same six variants but with two column groups (external, custom), values from `rq02/all/external/top_variants_overall.csv` and the existing custom numbers.

## Improvement: the three reframings

**A) Subtask subsets as a redundancy finding.** Agreed, the practitioner message is not "evaluate Vietnamese only", it is that full benchmarks carry redundant, noise-adding items. Rewritten:

```latex
\textbf{Full benchmarks are partially redundant.} Per benchmark, we rank
subtasks by standalone SNR and sweep cumulative subsets
(Figure~\ref{fig:subsets}). A subset of one to two MMLU subjects matches or
beats the full set of roughly 48 subjects across model sizes, raising SNR
from 2.12 to 3.65 at 175M, and the same subjects recur across sizes and
languages. We do not read this as a recommendation to evaluate on a handful
of subjects. We read it as evidence that a large share of benchmark content
adds variance without adding discriminative signal, so the size of a
benchmark is a poor proxy for its reliability. This invites a closer look at
which portions of standard suites are informative and which are redundant.
```

**B) Per-sample selection, positive framing.** Fair point, the gains are real and large, the open question is transfer. Rewritten:

```latex
\textbf{Item-level subsets.} The same procedure at the level of individual
items finds subsets with even larger SNR gains, showing that informative
items exist and can be identified within a single model size. The selected
items, however, differ across model sizes, with a Jaccard overlap of 0.03
between the best subsets at different scales. Item-level selection is
therefore promising as a per-scale tool, and making the selected subsets
transfer across scales is the main open problem, which we continue to
investigate.
```

**C) rq05 on the external pool.** The external pool is the right one here and the numbers are ready: 11 families survive the gate (versus 9 on custom), and crucially the survivors now include the four-option `belebele` and `global_mmlu_full`, so the option-count confound that crippled the custom-pool test is partially broken. The null still holds: curation H = 1.44, p = 0.49; format H = 0.00, p = 1.00; option count H = 0.05, p = 0.83; source origin p = 0.68. Rewritten:

```latex
\textbf{What makes a benchmark reliable.} We test whether benchmark design
features predict SNR on the external suite, where eleven benchmark families
clear the above-random gate, including four-option knowledge tasks that
capable models solve. No single feature is significant. Family-level
Kruskal-Wallis tests on curation method ($p = 0.49$), task format
($p = 1.00$), answer-option count ($p = 0.83$), and source origin
($p = 0.68$) are all far from significance. How a benchmark was built does
not predict its reliability once models clear chance on it. The pattern that
does hold sits in the answer space, since every benchmark that fails the
gate on small models is a four-option translated knowledge task, and
benchmarks are sharper when the model compares fewer and longer scored
completions. This suggests that for small-scale evaluation, reducing the
answer space matters more than curation quality.
```

## Updated Claude Code prompts

Figure 3 changes substantively, and one new prompt makes the DA table reproducible. Figures 1, 2, and 4 are unchanged.

**Replacement prompt for Figure 3:**

```
Figure 3: fig3_reliability_map.pdf, full width, DA-based on the external
suite (this replaces the SNR-based heatmap spec).

Data: analysis/rq01_decision_accuracy/all/external/da_per_task.csv.

Aggregation (must match make_da_reliability.py if it exists already, else
implement here and factor it out so both use it):
- Parse language from the task suffix (two-letter codes, plus belebele-style
  ISO3+script like eng_Latn and multiblimp ISO3 codes). Strip language and
  metric suffixes (_mc1, _mc2) to get the benchmark family.
- Small-to-large DA = mean over the 9 columns decision_acc_size_S_to_L with
  S in {270M, 600M, 1B, 1.7B} and L in {7-9B, 12-14B, 27-32B}. Average
  first within task, then across tasks of the same (family, language).
- Keep cells with at least 6 non-NaN pairs.

Render: rows = benchmark families sorted by row mean descending, columns =
en es ru hi zh ja ar vi tr th sw eu. imshow, viridis from 0.5 to 1.0
(clip below 0.5, since below 0.5 is worse than chance), missing cells white.
Annotate cells with DA at 7pt, white/black by background luminance.
Colorbar labeled "Decision accuracy (small to large, external suite)".
Sanity checks: xstorycloze_ru must be ~0.99, belebele_sw ~0.57,
hellaswag_es ~0.97. If any differs by more than 0.02, stop and show me the
intermediate dataframe instead of saving.
```

**New prompt, DA aggregation script (run before Figure 3):**

```
Create analysis/rq01_decision_accuracy/da_reliability_map.py.

Input: all/external/da_per_task.csv. Implement the aggregation exactly as
specified in the Figure 3 prompt (language parsing, family stripping,
small-to-large pair set, min support 6) and write two outputs next to the
input: da_reliability_map.csv (full family x language matrix of DA plus a
support column) and top_da_per_language.csv (per language: top benchmark,
DA, support, runner-up, runner-up DA). Print the top table to stdout.
Follow the repo's existing module conventions in rq01 (argparse --pool,
same logging style). Add one line to rq01's README under a "DA reliability
map" header with the regeneration command. Expected values to validate
against: ru/xstorycloze 0.99, es/hellaswag 0.97, sw/belebele 0.57,
tr/global_mmlu_full 0.92, ja/paws 0.90.
```

Also one small instruction to append to the Figure 2 prompt, given the pool decision: generate it twice, once from `pretraining/custom_swissai_hf/snr_variants_per_task.csv` and once from `all/external/snr_variants_per_task.csv`, saved as `fig2a` and `fig2b`, same axes and styling, so you can either show them side by side or keep external in the main text and custom in the appendix. Report both r values back.
