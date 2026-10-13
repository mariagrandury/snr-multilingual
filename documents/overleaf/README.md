# Overleaf exports

The paper is edited on Overleaf and generated in the repo (`documents/paper/`).
Each Overleaf download goes here as its own folder, e.g. `10-09-16h-full-overleaf/`
(or the `.zip` itself); with no argument the script reads the most recently
modified folder. The exports are not committed.

## Compare an export with the repo

```bash
python3 scripts/overleaf_diff.py                    # the newest export here
python3 scripts/overleaf_diff.py documents/overleaf/paper-overleaf-10-08-20h
python3 scripts/overleaf_diff.py EXPORT.zip         # unpacked next to the zip
```

It pairs files by name, so the two layouts can differ (Overleaf's `main.tex`
at the root, the repo's in `sections/`, the generated pages in
`sections/rqs/`, a renumbered `app_06_snr.tex` against `app_snr.tex`). It
prints one line per differing file and the files found on one side only. For
a text file it also says which side moved:

- `repo ahead (Overleaf = repo <commit>)`: the Overleaf copy is an older repo
  version. Copy the repo file to Overleaf.
- `diverged (X lines on Overleaf, Y in the repo, against repo <commit>)`: the
  Overleaf copy matches no repo version. X lines are Overleaf edits to port
  into the repo, Y are repo changes to bring to Overleaf, both counted against
  the closest repo version.

Figures and other binaries are compared by checksum. Then look at the text:

```bash
python3 scripts/overleaf_diff.py EXPORT --diff 04_analysis   # word diff in the terminal
python3 scripts/overleaf_diff.py EXPORT --diff               # every differing file
python3 scripts/overleaf_diff.py --code 04_analysis   # VS Code diff tabs (Overleaf left, repo right)
python3 scripts/overleaf_diff.py --html               # one report: <export>-diff.html here
```

Red is the Overleaf text, green the repo text.

The pages under `sections/rqs/` (and `tables/`, `figures/`) are regenerated
from the analysis, so an Overleaf edit to one of them is lost at the next
refresh. Port a layout change into its generator
(`documents/paper/figures/make_rq_appendix.py`) rather than into the page.




---

# Analysis

src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_outliers_paper.png -> why do we use all seeds for the rq01.png figure? how much do the results change if using only predictivity pool? do not update anyhting in the paper yet


\textbf{Monolingual comparison.} We repeat the analysis on the monolingual English cells ($K = 1$), under the same filter and fits. For each of the three English corpora (the DCLM-Edu English of every other cell, DCLM without the educational filter, and FineWeb), 48\% of the 33 English accuracy tasks are predictable along both size and training (Figure~\ref{fig:app_rq13_english_scaling_regimes_l1_accuracy}, Appendix~\ref{app:rq13_english_only}). These single-cell fits are not comparable in level with the medians over $K$ of Figure~\ref{fig:rq1}.
-> create a version of the figure with the benchmarks tagged and add in the main text a line specifying which are the benchmark that pass/not the filter, I'd be specially interested in knowding whether the "popular" benchmarks (mmlu, arc, hellaswag, openbookqa) are the ones that pass it or if some of the popular ones actually doesnt scale at all


At 600M the depth pairs are not size-matched: the shallow 600M model has 3.7\% more non-embedding parameters than the deep one and spends about 15\% more training compute (Table~\ref{tab:ladder}).
The 600M drop comes almost entirely from these pairs 
-> why do we have such a difference in the 600M deep/shallow? wasnt it possible to have them closer? do not update anything yet, just answer




% \CL{[NEW: decision accuracy of benchmark BPB on the cells where accuracy fails the filter of \S\ref{subsec:rq0}. If the values are near the per-language BPB levels of 0.8 to 0.9 rather than near 0.5, the recommendation follows: where accuracy is at chance, score the same items by log-likelihood.]}
bBPB on accuracy-gated cells (rq02 README, figure 11).
-> add it to a paper appendix and reference briefly in the main text


The discussion of the results of rq3.png change quite a lot, is this due to including the bBPB in the plot? Measure the impact differentiating a 3 subplot (benchmarks acc, benchmarks bbpb (new), per-language bpb).

write for the DA extension to 3B paragraph something different, not so focused on the 600M depth and 1B seeds, try to find a more optimistic approach and rewrite completely that paragraph









The lock is committed in the worktree as `fd38533d`, with no trailer. Your next step is to cherry-pick it into the live checkout:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual && git cherry-pick fd38533d
```

As soon as it's in, I'll do the rest in this order:
1. Land the auto edit (`land.sh`).
2. Submit the six 3B probe jobs. They wait until after landing so they don't repeat tasks that move into auto.
3. Start exactly one locked watcher with `nightly.sh evals`.

An agent is meanwhile extending the probe filter to 3B in the analysis. It only prepares the code and doesn't regenerate any paper files, so your Overleaf merge stays stable.

**What's new for Overleaf:**

New files to upload:
- **Appendix sections:**
  - `sections/rqs/app_05_majority_chance.tex`: the new majority-class chance appendix.
  - `sections/rqs/app_rq00_majority_chance.tex`: its figure page, inputted by the file above.
- **Table:** `tables/app_rq00_pass_counts_considered.tex`.
- **Figures in `figures/`:**
  - `app_rq00_chance_considered.png`: analysed vs screened benchmarks, on the rq00 page.
  - `app_rq00_majority_chance.png`: the majority-class appendix.
  - `app_rq02_bbpb_acc_gate.png`: bBPB on accuracy-gated tasks, on the rq02 page.
  - `app_rq13_english_scaling_regimes_l1_accuracy_tagged.png`: the tagged English figure, on the rq13 page.

Changed files to replace:
- **Text:** `sections/main.tex`, `sections/01_introduction.tex`, `sections/03_methodology.tex`, `sections/04_analysis.tex` and `sections/app_03_evaluation.tex`.
- **Appendix pages:** `sections/rqs/app_rq00_gate_and_curves.tex`, `sections/rqs/app_rq02_decision_accuracy.tex` and `sections/rqs/app_rq13_english_only.tex`.

`main.tex` carries two things: the new count macros, and the `\input{sections/rqs/app_05_majority_chance}` line after `app_04_chance_threshold`.





---

# Review 10-09-19h

## main.tex

\newcommand{\nbenchmarktasks}{1,699 }
\newcommand{\ntasksanalysed}{846 }
what is the first one? what is the difference between them?


% \rqgroup{title}{label} opens an appendix section of generated analysis pages: the next page
% (make_rq_appendix.py) typesets it with \rqparent inside its full-width block, then empties it.
\newcommand{\rqparent}{}
\newcommand{\rqgroup}[2]{\gdef\rqparent{\section{#1}\label{#2}\gdef\rqparent{}}}
-> why do i need this? is it totally necessary?


# Responsible NLP

- why did you update the question explanations? they were copied directly from the ACL guidelines, keep them exactly as original and instead of converting into comments with % wrap them with italics and itemize when necessary

## everywhere

- dashes are also okey for pairs, e.g. language--size
