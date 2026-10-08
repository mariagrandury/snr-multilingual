# agentE findings (appended as found)

## F1 (should-fix, CONFIRMED by code reading; repro pending) compute_da.py:287 crashes on every legacy (parquet) pool
40af906e. `psets = pair_sets(design_axes(df_pool))` — design_axes selects df[["family","L","arch","scheme","seed"]]; snr/download/apertus.py::_read_parquet keeps only model,size,mix,seed,step,task,primary_score,tokens,compute(+family). KeyError for seeds_1904 / seeds_28_1797 / custom_swissai_hf / external — run_all_pretraining.sh PASS A calls compute_da for those (cached by file existence, so only bites on regeneration). CLAUDE.md says the 36-sweep pools "remain runnable".
Also: the legacy tables were renamed *_both_axes.csv by d309cb72 but carry NO axes column (all/external, custom_swissai_hf, seeds_*), contradicting rule 16's "_both_axes when an axes column carries every pair set".

## F2 (nit, CONFIRMED) reliable_tasks.py:335-336 writes da_all_reliable_by_language_both_axes.csv with multi-axis rows only
committed file: axes value_counts {'multi-axis': 600}. Rule 16: name should be _multi_axes (or write every pair set).

## F3 (nit/latent) report_figures/make_figures.py:372-376 reads all/external/da_all_per_task_both_axes.csv without one_axes
Safe today (legacy table has no axes col); once F1 is fixed in a way that emits axes rows, fig3b averages stacked rows. Fix: `df = one_axes(pd.read_csv(path))`.

## F4 (should-fix, CONFIRMED) run_all_predictivity.sh:79 and :86 read tables that LATER steps write (CLAUDE.md failure mode #16)
- line 79 above_random_example.py reads rq02 `predictivity/da_all_per_task_both_axes.csv` (above_random_example.py:145) — written at line 112.
- line 86 reformulations_gate.py reads rq01 `predictivity_all/scaling_regimes.csv` (:113, written line 97), rq02 `da_all_per_task_both_axes.csv` (:107, line 112) and `da_all_reliable_tasks_both_axes.csv` (:110, line 130). Its own docstring: "this runs in seconds after them".
Effect: after a new report + FORCE=1, panel (c) of above_random_example and panels (b)-(d)/headline CSV of reformulations_gate are drawn from the PREVIOUS refresh's tables with a fresh timestamp. (reformulations_gate_paper = panel (a), mask only: unaffected.) Both added to the driver in a0a1e953.
Fix: move the two `run` lines to the rq02 block after reliable_tasks.py (line 130). Numbers: become current rather than one-refresh stale.

## F5 (nit/should-fix) run_all_predictivity.sh:120 compute_da --pool predictivity_schemes is not behind `fresh`
header lines 9-10 claim "per-task DA ... computed once per pool and reused"; this one recomputes every run. Consistency with cached per-pool tables only breaks when code changed without FORCE=1 (cached predictivity table = old kernel, schemes table = new).
Header nits: line 2-3 "L ∈ {1..100}", "five data schemes" (pre-existing, false: L ≤ 50, seven schemes); line 115 "AT3/BT3" (added on branch; BT3 retired by 49e24dad).

## item 1/3 verified clean (real pools, t1.out): no cell without -b at 90M/175M in any pool; tokens/step 344064 (90M), 688128 (175M), 2064384 else; predictivity 18 fam → 153 multi / 51 mono (arch 9, L 36, list 6); predictivity_schemes 25 fam → 300 / 63 (en 1, arch 10, L 39, T 4, list 6, lang2 3); seeds pool null 29; seeds_train (no 1904) 6/6/3.
## N1 (nit, design) L1 cells carry fictitious levels lang2="ru", list="A": L1 vs L2-ZH / L2-ES (and L1 vs L8/15/30-B) are multi-axis only, though at L=1 those axes are moot. Conservative (no wrong pair enters mono), 2 deep pairs missing from the mono "L" set.
## N2 (nit, rule 5) DA-goal cells below MIN_PAIRS have NaN in da_all_n_pairs (20886 cells in predictivity) — compute_da.py:236-238 `continue`s before the count is recorded; size/ckpt columns keep the count.
F1 repro: U.design_axes(parquet-schema frame) -> KeyError "['L', 'arch', 'scheme'] not in index". Fix in compute_da.run: fall back to {multi-axis: None (every pair)} when the pool has no design columns.
