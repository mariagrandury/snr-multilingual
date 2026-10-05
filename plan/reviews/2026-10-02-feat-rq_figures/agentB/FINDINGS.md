# FINDINGS agentB (rq04 catalogue.py/search.py @ d309cb72) — appended as found

NOTE: the df.pkl a previous run left in scratch is gone (not removed by this run).

## F1 (should-fix/blocker for the headline count, CONFIRMED) search.py:642-647 + filters_truth 314-342
Filter rows (basis=full) are built from the TOP_K surrogates selected on the same truth and data, then get p-values and enter the BH family and the README headline "N hold at BH q<0.05".
Null simulation (t5.py: y independent of 200 surrogates, n=240): share p<.05 = 0.055 in the screen, 0.56 for "[filter indicator]" rows, 0.16 for top-K-inside-filter rows.
Committed CSV: 7,560 indicator rows, 5,659 at q<.05. Fix: exclude subset_type=="filter" rows from p/q (set p NaN for basis full) or count significance on `corr` only; changes q of all rows + README counts -> search.py re-run.

## F2 (should-fix, CONFIRMED) search.py docstring 11-18, full_stats 136-157
"One point per cluster makes the points independent, so its p is a valid test" is false: 239 clusters come from 21 benchmarks x 34 languages, same models. ICC by benchmark: DA-size 0.24, n_items 0.77, kfold_abs 0.77, da_ckpt_mean 0.31. n_items rho=0.51 over 239 clusters p~1e-17, but it is effectively a benchmark-level covariate (21 units; within-benchmark rank corr 0.25).
Fix: reword docstring/README ("one point per cluster removes the proxy/checkpoint/twin replication; clusters of one benchmark or one language remain dependent, p is optimistic"), or permute within benchmark as the extra does.

## F3 (should-fix, CONFIRMED) catalogue.py:237 Kendall W has no tie correction
synthetic 8x5 rounded: code 0.8886 vs tie-corrected 0.9125; rows with identical columns incl. ties: 0.943 instead of 1; all-equal scores: 0.0.
Fix: W = 12 S / (m^2 (n^3-n) - m sum_j T_j), T_j = sum(t^3 - t) per column; NaN if denominator 0. Changes kendall_w_window values -> catalogue+search re-run.

## F4 (should-fix, CONFIRMED) catalogue.py:268,274,233-236,330-331,416-417 zero-variance guards are `> 0` on float sums
Window std of float-noise size (1e-16) passes `pooled > 0`: gain_over_noise = -3.6e12 (bbh_cloze_sports_understanding 600M), 4.8e14 (multiblimp_dan 350M); scale_gain_over_noise 8.1e13; cronbach_alpha 1.25/1.152 (20 rows > 1, impossible); icc -0.25 with all scores identical.
Fix: compare against a tolerance (e.g. `pooled > 1e-12`, `tot > 1e-12`, msb/ss_tot likewise), changes ~20-80 rows -> re-run.

## F5 (should-fix, CONFIRMED) catalogue.py:283 `c = 1 / n_opt` instead of `task_chance(task)` (rule 1: "read through task_chance")
truthfulqa_mc2: 0.143 vs 0.449; truthfulqa-multi_mc1: 0.2 vs 0.2253. Feeds chance, nonrandom, item_information, emerged_share, z_above_chance. truthfulqa_mc2 is also not a binary-item accuracy, so binomial_*/kfold are misapplied to it.
Fix: import task_chance; `p, c = xs.mean(), task_chance(task)`; registry "chance": "uniform-guessing chance level (rq00 task_chance)". Few rows change -> re-run.

## F6 (nit, CONFIRMED) duplicates in the registry
sign_persistence == da_ckpt_mean to 4e-16 on all 7,190 rows (same statistic, two names, two "sources"); signal__dispersion == signal__range (upstream). "~210 surrogates" counts them.

## F7 (nit, CONFIRMED) search.py:214-221 bh() returns all-NaN if any p is NaN (callers pre-filter, so latent).

## F1 quantified: of 75,442 "hold at q<0.05" (README:440), 39,576 are filter rows. Non-filter significant: 35,866 with filter rows in the BH family, 30,887 without (BH p cut-off 0.0157 -> 0.0090). Paper table bold cells 546 -> 528 of 626 (18 cells lose bold, |rho| 0.155-0.169).

## F8 (should-fix, CONFIRMED) DA-ckpt truth cells at 80 % and 90 % share their checkpoints with the window surrogates not in CIRCULAR (search.py:96)
Cluster-level rho with DA-ckpt by fraction of the truth: sign_consistency_window 0.31-0.62 at 10-70 %, 0.81 / 0.79 at 80 / 90 %; window_kendall 0.22-0.56 then 0.79 / 0.82; kendall_w_window 0.31-0.61 then 0.79/0.78; n_items flat 0.30-0.51. window_kendall IS the tau of the 80/85/90/95 % rankings with the final. README headline "window sign consistency for DA-ckpt (0.73) ... 87 % of the ceiling" is inflated by the two overlapping fractions.
Fix: either add window_kendall, sign_consistency_window, kendall_w_window, dior (and the window ANOVA family) to CIRCULAR for t_kind=="ckpt", or score DA-ckpt on frac < 1-NOISE_WINDOW only. Changes ckpt rows -> search.py re-run (catalogue not).

## F9 (should-fix, CONFIRMED) pseudo_ref_da is not labelled in the README, and the README says the opposite
search.py:611 writes "Surrogates are read on the proxy alone (rule 11)"; README:399 "211 proxy-only statistics". pseudo_ref_da (reads 1B) then appears unlabelled at README:486, 529, 544, 556-565 (16 of the 706 validated). Paper tex labels it (app_snr_new.tex:265, 296). Fix text only (+ suffix in the tables), no number changes.

## F10 (should-fix, CONFIRMED) catalogue.py:427-428, 431-432: language_consensus / item_total_corr peers include the task's own rf_/rfgm_ twin
`bench.get(u) != bench[t]` uses benchmark_family, which keeps the prefix (benchmark_family("rf_belebele_fra_Latn") = "rf_belebele"); 476 of 685 benchmark tasks at 1B have a twin in their cluster. tex:269 says "the task left out of its reference". Fix: compare twin-stripped families (G._TWIN). Changes two columns -> re-run.

## F11 (nit, CONFIRMED) docs drift: README:426 "(-0.20 to -0.35)" vs CSV -0.175..-0.317 on the three DA truths, and "every truth" false on the per-L truths (positive values exist); README:779 names `surrogate_by_language.csv` (no such file; it is surrogates_catalogue_by_language.csv) and omits surrogates_significant; registry "monotonicity_steps"/"size_monotonicity" say /#steps, code is /(#up+#down).

# --- continuation run (F8+) ---

## F8 (should-fix, CONFIRMED by code; numbers pending below) CIRCULAR is incomplete: search.py:96
CIRCULAR = {da_ckpt_mean, da_ckpt_half, sign_persistence, settling_time} but these also ARE early-vs-final agreements of the proxy on the same checkpoints the DA-ckpt truth reads:
window_kendall (tau of the 80/85/90/95 % rankings with the final = DA-ckpt tau_b at f=0.8, 0.9), sign_consistency_window (pairs ordered alike at 80..100 % => includes "80 % agrees with final" and "90 % agrees with final"), consecutive_kendall(_late), crossings, dior (tau of window draws with the final), kendall_w_window, split_half, prob_outperform. README headline "window sign consistency for DA-ckpt (0.73)" and the "Strongest surrogate per truth" rows for DA-ckpt are these.
Fix: add them to CIRCULAR (or to a second set left out for t_kind == "ckpt"), or restrict the DA-ckpt truth to fractions below the window (f <= 0.7) for window statistics. Changes the DA-ckpt rows, counts and README -> search.py re-run.

## F9 (should-fix, CONFIRMED) "read on the proxy alone (rule 11)" is stated without the pseudo_ref_da exception
catalogue.py:3-5 docstring, README auto block (search.py:611 "Surrogates are read on the proxy alone (rule 11)"), README:399 "211 proxy-only statistics". pseudo_ref_da (catalogue.py:404-405) reads the 1B finals for every proxy < 1B. The registry family says "pseudo-reference (reads 1B)", but README rows that show it (strongest DA-size L8, "language: en" 0.95, 4 of the 15 held-out rows) carry no label.
Also: for DA-goal/DA-ckpt truths at fraction f < 1 every surrogate is read on the proxy's FULL run (final checkpoint, 80-100 % window), i.e. on checkpoints later than the f the truth is "early" at. Rule-11-literal (proxy's data alone) holds, but the README never says the surrogate is not available at f.
Fix (text only, no numbers): search.py:611 -> "Surrogates are read on the proxy's whole run alone (rule 11) — so for DA-goal / DA-ckpt at an early fraction they use later checkpoints of the proxy — except `pseudo_ref_da`, which reads the 1B finals"; README:399 "211 statistics, 210 proxy-only".

## F10 (nit, CONFIRMED) registry formula vs code
- monotonicity_steps / size_monotonicity: registry "(#up - #down)/(#steps)"; snr/stats.py:25-29 divides by (#up + #down) (flat steps excluded, 0 if none). Fix the text.
- g_coefficient = (F-1)/F and icc_window = (F-1)/(F+m-1), F = MSB/MSW, m = 5 always (complete rows only): strictly monotone in each other => identical Spearman everywhere; one more duplicate for F6 (README "Particular cases" lists both at 0.99).
- _window_anova keeps only families with all five window points (catalogue.py:224): a run with three points (rule 4: "contributes three points") is dropped from icc/g/eta2/alpha/W but kept in gap_over_noise. Undocumented population difference.
- catalogue.py:268: a pair with s == 0 gets z = Z_CAP even when d == 0 (two tied, noiseless variants count as "perfectly separated").
- kfold closed form (catalogue.py:445-453): derivation checked: fold of m = n/k items drawn without replacement, Var(m_i) = p(1-p)/m * (n-m)/(n-1) = p(1-p)(k-1)/(n-1); exact for E[(1/k) sum (m_i-p)^2] under equal fold sizes, 0/1 items, ddof 0. Assumptions not stated: n divisible by k; sqrt(E var) >= E sd (Jensen; ~5-10 % at k=5, a constant factor); p = final accuracy. It is the binomial se times sqrt((k-1) n/(n-1)) => noise__kfold_abs is rank-identical to the mean binomial se; "k-fold beats checkpoint noise" = "binomial se / item count beats checkpoint noise".

## F4 addendum (CONFIRMED, t8.py): four identical noiseless variants -> gap_over_noise 50 (cap), resolved_pairs_noise 1.0, retest_agreement 1.0 (catalogue.py:268 gives Z_CAP to any s==0 pair, tied or not) while sign_consistency_window 0 and kendall_w 0. Float-noise window (pooled 1.4e-17) -> gain_over_noise 7.2e15.

## F12 (nit, CONFIRMED t4.py) per-L "best surrogate" rho is a max over 211: at L15 (55 clusters) observed max|rho| 0.27 vs permutation-null max median 0.30 (P=0.66); L8 0.38 vs 0.32 (P=0.24). README "Per language count" table and surrogates_by_L right panel show it without q.

## nits: bh() reimplements scipy.stats.false_discovery_control (identical output); catalogue.py:73/99 SURROGATES imported as a path then rebound to the registry dict; expected_sign never read by search.py or the table generator; search.py:382-385 four blank lines; fig_validated/fig_significant leave a stale PNG when empty while README embeds it; run_all runs search.py after a failed catalogue.py (stale inputs, fresh README block); "132" hard-coded in make_surrogate_tables.py:89 (correct today); no BLAS thread cap with 4 fork workers; numpy here links Accelerate (fork after BLAS use is unsupported by Apple, ran fine on 2026-10-01).

## checked clean: kfold closed form (sim 8.35e-4 vs 8.41e-4), ICC(1,1), ICC(1,k), eta2, alpha (=ICC(3,k)), Spearman-Brown, settling_time, strat_ranks, bh vs scipy, perm_p calibration (null 4.6 % at n=10), q reproduces from p in both CSVs, rq03 identity of snr__rel_std__ckpt_rel (max diff 0.0), no reference-size read outside pseudo_ref_da, rules 2/3/4/5/6/7/8/10/15, outputs all committed and none orphaned, 12 README numbers.

## F8 numbers (n2.py, recomputed from surrogate_values/targets via search.load; DA-ckpt da, multi-axis, benchmarks, 245 clusters)
cluster-mean Spearman, all fractions / truth at f<=0.7 only / truth at f>=0.8 only:
sign_consistency_window 0.728 / 0.603 / 0.885; window_kendall 0.640 / 0.498 / 0.877; kendall_w_window 0.705 / 0.582 / 0.859; icc_window 0.521 / 0.410 / 0.675;
non-window controls: autocorr 0.657 / 0.625 / 0.587; gain_over_noise 0.649 / 0.609 / 0.594; n_items 0.593 / 0.548 / 0.543.
=> the README's "window sign consistency for DA-ckpt (0.73)" is lifted by the two fractions (80 %, 90 %) that are inside the surrogate's own window; off the window it reads 0.60 and is no longer the best (consecutive_kendall_late 0.63, autocorr 0.63).

## F1 quantification (n1.py): of the 75,442 rows at q<0.05, 39,576 are subset_type=="filter" rows; 35,866 are not. 67,920 of the 409,273 configurations are filter rows.

## F11 (nit, CONFIRMED) the headline rho is not invariant to a monotone transform of the surrogate
full_stats (search.py:146) averages RAW surrogate values per cluster before ranking. icc_window and g_coefficient are monotone in each other: identical rho_w (0.150 / 0.126 / 0.330 on DA-size/goal/ckpt) but different rho (0.316 vs 0.335; 0.325 vs 0.351; 0.521 vs 0.482). Heavy-tailed ratios (F4's 1e12 values, the SNR grid) are dominated by one cell of the cluster.
Fix: cluster median, or the mean of within-stratum ranks (gives 0.289 for both). Changes every rho slightly -> re-run.

## CLEAN: composition confound of the cluster means
158 of 239 clusters have all five proxies; rho(mean proxy index, mean truth) = -0.13 / -0.17 / -0.24. Replacing raw cluster means by means of within-stratum ranks moves the headline rhos by <= 0.03 (da_ckpt_mean->DA-size 0.568 -> 0.578; n_items 0.510 -> 0.514; sign_consistency_window->DA-ckpt 0.728 -> 0.733). The rho (0.57) vs rho_w (0.29) gap is averaging, not composition.

## F12 (should-fix, CONFIRMED) the held-out "extra" shares every benchmark and every language between the halves
search.py:242-247 splits clusters WITHIN each benchmark (145 discovery / 134 validation clusters); a benchmark-level covariate (n_items, chance, kfold noise: F2) is the same on both halves, so "found on one half, confirmed on the other" confirms transfer to other languages of the same benchmarks, not to a new benchmark. The validation p itself is sound for that narrower claim: _blocks keys on benchmark, so truths are exchanged only among clusters of one benchmark with the same cell pattern. 706/1,849 at q<.05; the most-confirmed surrogates are the window statistics of F8 (kendall_w_window 67, sign_consistency_window 57, window_kendall 48), 134 of the 706 are "reliable (on the truth)" subsets.
Fix (text): README/docstring "split within each benchmark, so both halves hold the same benchmarks and languages; the test is of the association within a benchmark across languages". A leave-benchmarks-out split would be the stronger check (changes numbers).

## F13 (nit, CONFIRMED) docs vs CSV
- README:~419 "Range, standard deviation and pairwise distance across variants correlate negatively with every truth (−0.20 to −0.35)" MISMATCH. surrogate_correlations.csv, subset benchmarks: on the three multi-axis DA truths range −0.22/−0.22/−0.18, rel_std −0.24/−0.27/−0.32, rms_deviation −0.26/−0.28/−0.22, mpd −0.28/−0.31/−0.24; on tau_b/rho truths −0.01..−0.19; on per-L truths up to +0.19 (rel_std, DA-size L15), +0.09 (range). No dispersion signal reaches −0.35.
  Corrected: "…correlate negatively with DA-size, DA-goal and DA-ckpt over every pair (−0.18 to −0.32); the sign weakens on the τ-b / ρ truths (−0.01 to −0.19) and flips on the L8 / L15 per-L truths (up to +0.19)."
- README:399 "211 proxy-only statistics": 211 defined, 210 scored (signal__dispersion_shifted is constant, no ρ), 210 proxy-only (pseudo_ref_da reads 1B). README:13 "~210" fine.
- README Files (~l.779): `surrogate_by_language.csv` does not exist (the file is `surrogates_catalogue_by_language.csv`); `surrogates_significant.{csv,png}` and `surrogates_{snr_grid,catalogue,by_L}.csv` not listed.
- surrogate_definitions.csv "kendall_w_window" does not say "no tie correction" (the paper table l.215 does); "chance" says "1 / number of options" (F5).
- make_surrogate_tables.py: "of the 132 ratios" hard-coded; equals len(sigs)*len(noises) = 22*6 today. Fix: f"{len(sigs) * len(noises)}".
- app_snr_new.tex:299 "so the points are independent and the p-value is a valid test" = F2 (same fix).
MATCH (README vs CSV): 409,273 configurations / 240,212 with p / 75,442 at q<.05; 30 truths; 22x6=132 ratios; ceilings 0.66/0.81/0.84; 239–245 clusters; 0.73 / 0.59 / 0.57 and "73–87 % of the ceiling"; rel_std/ckpt_rel 0.25 (0.255) and rel_dispersion 0.23 on DA-size; best ratios 0.48/0.51/0.63; n_items 0.51; k-fold > ckpt for all 22 signals on all three truths; particular cases 0.94–1.00, 10–19 units; L8/L15/L30 mean DA-size 0.51/0.48/0.48; 706 of 1,849.
MATCH (tex prose < l.313): 22 signals, 22x6=132, sqrt(4/5) and 1.118, n_c=5, z cap 50, 100 DIoR draws, window {0.80..1}, ICC denominators (4 MSW), 5/4 alpha, split-half sets, ceiling 0.66, "ρ≈0.5–0.6" (top DA-size: 0.57, 0.57, 0.52, 0.51, 0.51, 0.50), SNR as released 0.23, item count 0.51 vs best ratio 0.48, k-fold raises every signal (22/22), dispersion negative on the three table truths, pseudo_ref_da labelled as the exception (l.265, l.295).

## CLEAN (this run)
- Rule 11: compute()/snr_grid() read only size == proxy (curves, window, finals), rungs <= proxy for the small-ladder family, same-size peers for the agreement family; TARGET_SIZE enters only as log N in projected_flip_rate; no 1.7B row is read. pseudo_ref_da absent at 1B (0 values), present at the four smaller proxies.
- Rules 1/5/7/2/3/4/10/15: truths gated at proxy and reference through long_da / G.mark_gated with GATE_REF (ckpt: proxy only); MIN_PAIRS in _curve_stats, _gate, pair_agreement (targets n_pairs min 4, values n_pairs min 4/6); languages_only on both tables (34 languages, no multi/??); ladder_frame -> build_snr_pool; on_shared_grid for the ten tenths, noise_checkpoints for the window; axes column on every truth, reliability filters computed per pair set. Retest DA is on the same pair count as its truth (0 of 30,192 rows off).
- Formulas: ICC(1,1), ICC(1,k), eta^2, Cronbach alpha, Kendall W (untied), Spearman-Brown, settling time, crossings, DIoR (tau_b, 100 draws, crc32 seed per cell), MDE, two-proportion z, k-fold closed form. BPB sign: score = -BPB in compute(), so bpb_rank_agreement / bpb_corr_training / gain / ladder statistics are oriented; truths are sign-invariant.
- Search: perm_p two-sided, (b+1)/(B+1), 20,000 shuffles, seed = n; null simulation n=15: 3.8 % < .05, 0.8 % < .01; perfect ρ at n=12 gives 5e-5 not 0. No p below 10 units. strat_ranks equals brute force; bh correct on finite input. Selection for the extra uses rho_disc only; filter thresholds/signs for basis "disc" come from half 0. Seeds fixed everywhere (load: 0; validate: truth index; DIoR: crc32), Pool.map keeps order.
- fork: this Mac has numpy on Accelerate and matplotlib backend macosx; a synthetic fork Pool(4) doing matmul/corrcoef after parent BLAS use and after a figure completed in all three orders.
- Outputs: all files the two scripts write are in HEAD (3 + 4 CSVs, six PNGs each with a CSV); surrogate_definitions.csv equals the in-code registry; both scripts are in run_all_predictivity.sh (l.202-203), make_surrogate_tables.py in refresh_analysis.sh (l.88). surrogates.csv / surrogates_by_{benchmark,language}.png belong to panels.py, not these scripts.
