# Make the nightly analysis refresh fast (2026-10-06)

> **DRAFT: interrupted 2026-10-06, investigation incomplete.** The timings, the loader measurements and the review of the decision-accuracy kernel below are measured or read from code. The profile, the micro-benchmarks, the consumer map and the keep/drop ranking were not done; see "Not done yet" at the end. Treat every expected saving marked *est.* as an estimate until the benchmark in §6 runs.

## 1. Goal and target runtime

Today's refresh (job 3591295, started 07:18, 6 h limit) will not finish. 213.5 min of steps were done when it reached the scheme-inclusive decision-accuracy pool, and that pool alone took 50.9 min on 2026-10-04, before the bBPB twins doubled the task count. Projected total: about 213 + 100 + 65 ≈ 380 min, more than the 360 min limit, so the job dies somewhere in the noise/SNR or surrogate block.

**Target: under 45 min wall for the nightly core.** Reasons:
- After the kernel rewrite in §3, decision accuracy is minutes of numpy instead of hours of pandas, so the floor is set by the ~80 script start-ups, each of which re-reads and re-filters the 77 MB report (§2).
- The tables are then ready about an hour after the 04:00 publish, whenever the job starts.
- The run stays far enough under the wall that adding tasks (twins, probes, the 3B rung) does not bring the timeout back.

## 2. Where the time goes

Sources: `/iopsstor/scratch/cscs/mariagrandury/logs/nightly/2026-10-04.refresh-3574595.log` (the full run, 77 steps), `.../2026-10-06.refresh-3591295.log` (today's, still running), and `sacct`.

| Step | 2026-10-04 | 2026-10-06 | Note |
|---|---:|---:|---|
| `compute_da.py --pool predictivity_seeds` | 60.6 min | 114.6 min | ×1.9 from the bBPB twins (`utils.with_bbpb_twins`, utils.py:724) |
| `compute_da.py --pool predictivity_schemes` | 50.9 min | still running (*est.* ~100) | |
| `compute_da.py --pool predictivity` | 34.9 min | 69.5 min | |
| `compute_da.py --pool predictivity_seeds_train` / `_test` | 4.3 / 2.1 min | 4.9 / 3.8 min | |
| **compute_da.py, all five pools** | **152.7 min (71 %)** | ***est.* ~290 min** | |
| `scale_convergence.py` ×4 variants | 8.6 min | not reached | |
| `analyze_snr_variants.py` + `snr_definition_postprocess.py`, ×2 pools | 8.3 min | not reached | |
| `tokens_seen.py` | 2.8 min | 6.9 min | ×2.5 |
| `finetasks_criteria.py`, `compare_seed_splits.py`, `smooth_subtasks.py`, rq00 `panels.py`, `regimes.py`, `compare.py` | 2–3 min each, 15.3 min together | | |
| the other ~55 steps | 0–1.5 min each, ~32 min together | | most are about one loader call (below) |
| all `run_all_predictivity.sh` steps | 216.4 min | 213.5 min so far | |
| after the pipeline: paper figures and tables, deck figures, scaling plot, report PDF, compendium, checks, facts | ~7 min | | job elapsed 223.5 min minus 216.4 min of steps |
| separate `ladder-curves` job (`run_apertus.py` with grids) | own job, 4 h limit | | ~85 min in an older `snr-analysis-*.log` (not re-checked) |

**Measured 2026-10-07** (`.../2026-10-07.regen-3601965.log`, the full refresh after §3.1, §3.3 and §3.4, frozen 10-06 report, `COMPUTE_DA_WORKERS=64`, four pools): the 85 driver steps took 217 min (job 3 h 41 min). `compute_da.py` is no longer the cost, 1.6 min for the four pools (44 s `predictivity_seeds`, 40 s `predictivity`, 5–6 s each holdout). The two rq04 surrogate steps now dominate: `search.py` 84 min and `catalogue.py` 28 min, 52 % of the steps. Next are `per_item_ladder.py` (10.4 min), `above_chance_items.py` (7.2 min), rq01 `regimes.py` (5.9 min), `finetasks_criteria.py` (5.2 min) and `scale_convergence.py` (4.1 min for the headline call).

**The loader is paid on every step.** I measured it in the worktree, single-threaded on the login node:
- `load_predictivity_eval_results()` takes 39.4 s and produces a 4.43 M × 19 frame. A second call in the same process takes 38.1 s again, so nothing is cached.
- `build_snr_pool()` takes 46.6 s for `predictivity` (614 k rows, 1,714 tasks, 22 families), 45.9 s for `predictivity_seeds` (711 k rows, 36 families) and 48.8 s for `predictivity_schemes` (893 k rows, 30 families).
- Peak RSS is 8.6 GB.

About 80 steps each build at least one pool, so *est.* 50–60 min of the run is the same CSV being parsed, melted, filtered and twinned again and again. This is also why so many steps take a flat 40–60 s.

**Repeated work across pools.** `predictivity` (seed 1904, schemes A/B) is a subset of the families of both `predictivity_seeds` and `predictivity_schemes`, and the train/test pools are subsets of `predictivity_seeds`. `compute_da.py` treats each pool from scratch, so every family of the headline pool goes through checkpoint selection three times. Within one pool it happens three more times, once per pair set (§3.2).

**Critical path.** The steps are serial: publish (04:00), queue wait (today 3 h 18 min), the five decision-accuracy pools, the scripts that read their tables, then the documents. Nothing runs in parallel on a 288-core node that the run uses one core of. `nightly.sh` sets `OPENBLAS_NUM_THREADS=4` and no Python-level parallelism.

**Steps failing every night.** These failed on 2026-10-04, so their outputs were already stale (`run_all_predictivity.sh` FAILED list):
- `by_L.py` ×3
- `agreement.py`
- `scaling_vs_ranking.py`
- `public_ladders.py`
- `finetasks_criteria.py`
- `per_item_ladder.py`

## 3. Code optimizations, ranked by minutes saved per unit of effort

### 3.1 Rewrite the decision-accuracy kernel as one score cube per task (saves ~150–290 min; effort about 1 day)

**Where.** `analysis/rq02_decision_accuracy/compute_da.py`:
- `compute_size_decision_accuracy` (lines 106–135)
- `compute_ckpt_decision_accuracy` (lines 138–185)
- `_scores_at` / `compute_early_small_decision_accuracy` (lines 191–243)
- the loop in `run` (lines 296–327)
- `analysis/utils.py:452` `pair_agreement`

**Why it is slow (read from the code).** For each pair set (3 in a seeded pool) and each of 1,714 tasks, `run` makes:
- 5 + |scaling pairs| size-DA calls,
- 9 fractions × 6 buckets = 54 checkpoint-DA calls,
- 6 buckets × 10 fractions = 60 `_scores_at` calls for DA-goal.

That is about 130 calls per (task, pair set), and each one:
- calls `add_family_column(df)` again (lines 122, 154, 223),
- re-filters the task frame by bucket,
- loops over families in Python with `groupby` + `sort_values` + `iloc`,
- runs `pair_agreement`, which walks the whole pair list in Python (`[(a, b) for a, b in pairs if a in common and b in common]`, then one `np.sign` per pair).

For `predictivity_seeds` that is about 3 × 1,714 × 130 ≈ 670 k pandas group-bys per run.

**The change.**
1. **Build one cube per task.** The checkpoint the three definitions read is the same: the early checkpoint of checkpoint-DA (lines 164–177) and `_scores_at(b, frac)` (lines 199–207) use one rule, and DA-size reads `_scores_at(b, 1.0)`. So per task, build one cube `S[family, (bucket, frac)]` for frac ∈ {0.1 … 0.9, 1.0} in one vectorised pass (nearest step to `frac × max_step` among non-final steps, within `CKPT_TOL`), plus the matching `compute`.
2. **Compute pair signs once per pair set.** Turn each pair set into index arrays `(i, j)` and compute `sign(S[i] − S[j])` once, an (n_pairs × n_cols) matrix.
3. **Read every DA column from that matrix.** Every column (size, scaling, ckpt, goal) is `mean(sign[:, p] == sign[:, r])` over the rows where both columns exist, with `n_pairs` = that row count.
4. **Reuse across pair sets.** The cube does not depend on the pair set, so the three pair sets reuse it.

**Expected saving.** The kernel goes from ~670 k Python group-bys to ~10 k small numpy ops. *est.* < 5 min for all pools (not yet benchmarked; §6 step 2).

**Done 2026-10-07 (measured, login node, one core, pool frame pre-built so the load is excluded).** `compute_da.py` now builds the cube once per pool (`_ckpt_cube`) and reads every column from per-pair-set sign matrices (`_task_rows`); the per-call kernels stay for `by_L.py`, the tests and any task with two rows at one (bucket, family, step), which they compute as before. For the first three rows the old and new code ran on the same pre-built frame and all three CSVs are byte-identical (`cmp`); the stdout logs were compared on `predictivity_seeds_test` only and match apart from the output path. The full `predictivity` row was not compared against the old code (only 4 workers against 1 worker, identical), and its old time is the nightly wall time, which includes the ~47 s pool build:

| Pool | old | new | speed-up |
|---|---:|---:|---:|
| `predictivity_seeds_test` (1,714 tasks) | 261.1 s | 1.2 s | ×220 |
| `predictivity_seeds_train` (1,714 tasks) | 374.3 s | 1.5 s | ×250 |
| `predictivity`, 60-task subset | 207.4 s | 0.7 s | ×300 |
| `predictivity`, all 1,714 tasks | 69.5 min (nightly 10-06, pool build included) | 17.9 s (15.9 s with 4 workers) | ~×230; 4 workers vs 1 worker only, old-vs-new `cmp` pending on Slurm |

No task in these pools has duplicate (bucket, family, step) rows, so none took the per-call path. `tests/test_early_small.py::CubeMatchesPerCallKernels` checks the cube against the per-call kernels on random ladders with tied distances, tied and NaN scores, single-checkpoint runs and runs with no checkpoint near a fraction.

**Risk to results.** It can be bit-identical. DA is an integer count over an integer count in both versions, and `np.mean` of a bool array sums exactly in float64. Four details have to be replicated:
- **Tie-break on equal distance.** `argmin` on the step-sorted frame picks the smaller step.
- **First of duplicate final steps.** `idxmax` takes the first row of a duplicated max step.
- **NaN scores.** Today a family with a NaN score still counts as present: the pair is counted and scored as a miss, because `NaN == NaN` is False. The cube must keep "present with NaN" apart from "absent".
- **Rule 5 bookkeeping.** The `_FEW_PAIRS` log and the n_pairs table must stay the same.

The `compute` means in the early table must be taken over families in sorted order, as now.

**How to verify.** Run old and new on the same frame and compare the three outputs byte for byte with `cmp`:
- `da_all_per_task_both_axes.csv`
- `da_all_n_pairs_per_task_both_axes.csv`
- `da_goal_early_small_per_task_both_axes.csv`

Add a unit test next to `tests/test_metrics.py` that checks the cube kernel against the current functions on random frames with ties and NaN.

### 3.2 Compute once across nested pools (saves the remaining pool repeats; effort about half a day, after 3.1)

A family's checkpoint selection depends only on its own rows. So build the cubes once on the union frame (`predictivity_seeds`, which holds every family of the four pools since 2026-10-05) and evaluate each pool as (its families, its `pair_sets`, its buckets, its `_scaling_da_pairs`) on the shared cubes, all in one process.

Two things are pool-specific and must be recomputed per pool: `pair_sets`'s fallback when the pool has no grid-seed cell (utils.py:408–440) and `_scaling_da_pairs`'s "≥ 2 shared families" (compute_da.py:246–260). Results stay bit-identical. Verify the same way as 3.1.

### 3.3 Cache the built pool frames as parquet, keyed by content hash (saves *est.* 45–55 min; effort about half a day)

Add a cache in `build_snr_pool` (utils.py:249) / `load_predictivity_eval_results`:
- write the filtered, twinned frame to `data/cache/<pool>-<hash>.parquet`;
- key the hash on: the bytes of `ladder_report.csv`, `bench_bpb.csv`, `configs/models.json`, `configs/tasks.json`, `configs/languages.json`, and the source of `snr/download/ladder.py` and `analysis/utils.py`.

The first call in a run pays ~45 s, and every later call reads parquet in a few seconds (*est.*; not measured).

**Done 2026-10-07, at the loader instead of the pool** (the pool code is being rewritten): `load_predictivity_eval_results` keeps its melted frame as a pickle under `src/signal-and-noise/.cache/` (git-ignored), one file per argument set, keyed by the SHA-256 of the report CSV, `ladder.py`, `pretrain/{ladder_report,launch_trainings,pretrain_progress}.py`, the hyperparams and `configs/*.json`. Pickle, not parquet, so every dtype round-trips; `assert_frame_equal(check_exact=True)` holds for the frame and for the `predictivity` and `predictivity_seeds_train` pools built from it. Measured: uncached 37.3 s, miss (build + write 1.2 GB) 39.3 s, hit 0.7 s warm; `build_snr_pool` drops from ~47 s to 11.1 s (`predictivity`) and 2.0 s (`predictivity_seeds_train`). `SNR_CACHE=0` bypasses it, `SNR_CACHE_DIR` moves it. `bench_bpb.csv` is not a key: the twins are added in the pool step, after the cache. The file holds the key first and the frame second, so a stale or unreadable file is rebuilt without unpickling the frame.

### 3.4 Run the per-task loop in parallel on the compute node (on top of 3.1; effort about 2 h)

The job gets a full node (sacct: 288 CPUs) and uses one. Tasks are independent in `compute_da.py`, and the same holds for the per-task loops in the SNR variants, `scale_convergence.py` and `cross_task.py`. Use `ProcessPoolExecutor` with a fork-shared frame and a worker count from `SLURM_CPUS_ON_NODE`, capped at 4 off Slurm (login-node rule). *Done for `compute_da.py` 2026-10-07*: `--workers N` or `COMPUTE_DA_WORKERS` (default 1), fork-shared cube; output identical to one worker. After 3.1 it saves little (17.9 → 15.9 s on `predictivity` with 4): the cube build is serial. Results stay bit-identical if the output is re-sorted by (task, axes) as now. This matters little once 3.1 lands, but it is a safety margin for the next doubling of tasks.

### 3.5 Read the next-heaviest steps for the same pattern (not done)

Candidates, by measured time:
- `scale_convergence.py` (8.6 min over 4 calls)
- `analyze_snr_variants.py` + `snr_definition_postprocess.py` (8.3 min)
- `tokens_seen.py` (6.9 min today)
- `finetasks_criteria.py`, `compare_seed_splits.py`, `smooth_subtasks.py` (2.6–3 min each)

These have not been read for quadratic loops yet.

## 4. Pipeline-level changes

1. **Replace the mtime freshness check with a content hash.** `fresh()` in `run_all_predictivity.sh` compares mtimes, and `nightly.sh` passes `FORCE=1` every night because the fetched report carries its commit time. So the cached decision-accuracy and SNR tables are recomputed every night even when neither the report nor the code changed. Instead, write a sidecar `.inputs.sha256` (same inputs as 3.3, plus the script's own source) next to each cached table and skip when it matches. Then `FORCE=1` is no longer needed nightly.
2. **One process for the five decision-accuracy pools (3.2).** It writes the same five folders.
3. **Run independent steps in parallel.** After the decision-accuracy tables exist, the blocks that only read them are independent of each other: the noise/SNR variants and seed holdout, design decisions, language transfer, subset selection, benchmark design, size generalisation. Run them as background jobs inside the allocation (`xargs -P` or `&` + `wait`), with each block's internal order kept. Bug #16 of `src/signal-and-noise/CLAUDE.md` (a figure drawn before its input) means only true leaves may run concurrently. Map the dependency edges from the driver comments before doing this.
4. **A fast core plus an on-demand tail.** The nightly runs the tables and figures the paper reads (§5), then `verify_paper_results.py`, `make_appendix_tables.py`, `make_surrogate_tables.py`, `facts.py` and `check_rules.py`. Exploratory and viewer outputs move to a weekly or by-hand target (`run_all_predictivity.sh --full`).
5. **Drop the deck steps from the nightly.** The user says the deck no longer needs updating, so these go:
   - the nine `documents/figures/fig_*.py` loop in `refresh_analysis.sh` step 3,
   - the slides half of the figure check,
   - the slidev block.

   This is within the ~7 min post-pipeline cost; `fig_setup` and `fig_from_analysis` may feed the report PDF, which still has to be checked (§5).

## 5. Analyses and figures to drop or demote (incomplete)

The consumer map (paper `\includegraphics` / `\input`, `make_rq_figures.py`, `verify_paper_results.py`, the appendix and surrogate table makers, README image links, `build_report.py`, `facts.py`, `check_rules.py`, and other scripts' inputs) **was not built**. The candidates below rest on the driver's own comments and the 2026-10-04 failures, not on a consumer grep, and must be confirmed before anything moves.

| Candidate | Proposal | Evidence so far |
|---|---|---|
| `run_apertus.py` grids (separate `ladder-curves` job) | demote to on-demand | the driver and `nightly.sh` comments say "~140 figures nothing else reads"; ~85 min |
| deck figures `documents/figures/fig_*.py`, slidev | drop from nightly | user: the deck no longer needs updating |
| `by_L.py` ×3, `agreement.py`, `scaling_vs_ranking.py`, `public_ladders.py`, `per_item_ladder.py`, `finetasks_criteria.py` | fix or demote | failed on 2026-10-04, so their outputs were already stale; if no paper input reads them, demote |
| `da_explainer.py` | demote | toy explainer, "no measured number" (driver comment) |
| `pair_axes.py` | demote | "(exploratory)" in the driver |
| `scale_convergence.py --by L --langs L8 [--common-tasks]`, `by_language.py`, `language_tier.py` | demote unless the paper uses them | extensions; per the 2026-09-30 snapshot they "do not order the per-L lines" (`src/signal-and-noise/CLAUDE.md`) |
| `catalogue.py` + `search.py` (~210 surrogates) | check the cost and the consumer | new since 10-04; 28 + 84 min on 10-07, the largest two steps |
| the decision-accuracy, SNR and surrogate tables per pool, seed holdout, `reliable_tasks.py`, `paper_rq2.py`, `scale_convergence.py` (headline), gate, scaling | keep | read by the paper path (`make_rq_figures.py`, `verify_paper_results.py`); to be confirmed by grep |

## 6. Execution order and verification

1. [ ] **Snapshot today's outputs as the reference.** Use the worktree, not the main checkout, while jobs run there.
2. [ ] **3.1 kernel.** (partly done 2026-10-07; numbers in §3.1: `cmp` passed on `predictivity_seeds_test`, `predictivity_seeds_train` and a 60-task `predictivity` subset; the full-pool old-vs-new `cmp` is still to do on Slurm) Benchmark old against new on `predictivity` restricted to ~30 tasks, then the full pool. `cmp` the three CSVs.
3. [ ] **3.2 shared cubes over the four pools (five until 2026-10-05).** `cmp` all twelve CSVs.
4. [ ] **3.3 loader cache (pickle).** (partly done 2026-10-07, at the loader, as a pickle; §3.3: `assert_frame_equal` passed for the loaded frame and the `predictivity` and `predictivity_seeds_train` pools; the full driver run is still to be timed on Slurm) `assert_frame_equal` per pool, then time one full driver run.
5. [ ] **§4.1 hash freshness.** Remove `FORCE=1` from `nightly.sh`.
6. [ ] **§5.** Finish the consumer map. Move the demoted steps behind `--full`, and drop the deck steps from `refresh_analysis.sh`.
7. [ ] **§4.3.** Run the leaf blocks in parallel, then re-check the orphan list and `check_rules.py`.
8. [ ] **Full verification.** Run the refresh in an allocation. Every regenerated CSV under `analysis/` and `documents/paper/` must be byte-identical to the snapshot of step 1 (`git diff --stat` empty for the tables; PNG/PDF deterministic under `SOURCE_DATE_EPOCH=0`). Record the wall time per step.

## Not done yet

- A cProfile of `compute_da.py` on a ~30-task subset, and micro-benchmarks of the cube kernel and a vectorised `pair_agreement`. The savings in 3.1–3.4 are estimates.
- Reading `scale_convergence.py`, the SNR variants, `tokens_seen.py`, `finetasks_criteria.py`, `compare_seed_splits.py` and `smooth_subtasks.py` for quadratic loops.
- The consumer map per script output (paper `.tex`, `make_rq_figures.py`, `verify_paper_results.py`, `make_appendix_tables.py`, `make_surrogate_tables.py`, READMEs, `build_report.py`, `facts.py`, `check_rules.py`, other scripts' inputs), and the list of outputs nothing consumes.
- The keep / demote / drop ranking of every step in `run_all_predictivity.sh`, with evidence.
- ~~The cost of the steps added since 2026-10-04~~ measured 2026-10-07 (§2): `search.py` 84 min, `catalogue.py` 28 min, `bench_bpb_da.py` 2.0 min, `reference_solved.py` 1.4 min, `recipe.py` 1.2 min, the probe `compare.py` 1.1 + 0.5 min.
- The cost of each post-pipeline step individually, and whether the report PDF needs any `documents/figures/fig_*.py` output.
- Checks that the parquet round-trip preserves dtypes, and whether the frame has NaN scores or duplicate (family, bucket, task, step) rows, which the cube kernel must replicate.
