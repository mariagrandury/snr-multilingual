# Retraining the 90M and 175M rungs at their own batch

**Decided 2026-09-23.** The two smallest rungs are retrained at global batch 84
and 168 against the ladder's 504, holding each rung's token budget. This
document records what changed in `src/pretrain/`, what has *not* changed yet,
and the order the rest has to happen in so the analyses read the new models and
not the old ones.

Evidence and the option comparison live in
[`90M-rung-anomaly.md`](90M-rung-anomaly.md); the figure is
[`90M-ladder-fixes.png`](90M-ladder-fixes.png). This is option 2 of the three
weighed there.

## Why, in one paragraph

AdEMAMix's slow EMA averages over a fixed `1/(1-beta3)` = 10,000 optimizer
steps. A rung's *length* scales with its own token budget, so the ladder spans
4,500 steps at 90M to 81,000 at 1.7B — 0.45× to 8.1× that window. Below ~1× the
slow average never leaves its warmup regime: nine of ten 90M runs degrade
monotonically after ~17% of training and end 1.2–1.9 nats above their own best,
and 175M lands +0.26 off the power law fitted on 350M–1.7B. Holding the token
budget and cutting the batch multiplies the step count by the same factor, which
shrinks the fixed window to a fraction of the run. **It is the steps that fix
it, not the smaller batch**: the step-*matched* batch cuts (`diag-90M-gbs252`,
`diag-90M-gbs84`, same 4,500 steps, ⅓ and ⅙ the tokens) diverge exactly like the
original.

| rung | batch | steps | memory / run | final loss | was |
| --- | ---: | ---: | ---: | ---: | ---: |
| 90M | 84 | 27,000 | 0.37× | **2.715** | 5.781 (diverged) |
| 175M | 168 | 25,620 | 0.39× | **2.565** | 2.912 (+0.26 off trend) |

168 rather than 84 at 175M: it runs on 6 nodes (DP 24) and Megatron needs
`GBS % DP == 0`, so 84 is not a valid layout there.

## Done: `src/pretrain/`

**Superseded on 2026-09-23.** `configs/models.json` has since been synced
(step 2, without `--prune`) and the trainings of step 1 have been launched.
What follows records the code change itself.

- **`launch_trainings.py`**
  - `GBS_BY_SIZE = {"90M": 84, "175M": 168}` and `cell_gbs(size)` — the one
    place the exception lives. An import-time assertion checks both constraints
    (`GBS % gbs == 0`, `gbs % DP == 0`), so a typo is a traceback and not 60
    failed jobs.
  - `scale_for_gbs(cfg, gbs)` — the schedule rescaling, extracted from the
    `--gbs` branch so the launcher and `sync_models_json` share one definition.
    Holds D = 100 × N: iters, warmup and decay all scale by `k = 504/gbs`.
  - `mix_label` / `exp_name` gain a `-b<N>` part: `lm-90M-L2-b84-deep-seed1904`.
  - `iter_ms(size, arch, gbs)` — step time is *not* proportional to batch; a
    fixed per-step cost survives and dominates at these rungs. Fitted
    `t(gbs) = t(504)·(gbs/504 + 0.20)` from the two clean measurements, taking
    the larger constant so the walltime over-estimates rather than walls a job.
  - `--gbs` keeps forcing a `diag-` name and stays the diagnostic override.
- **`sync_models_json.py`** — `TOKENS_PER_ITER` was hardcoded at `504*4096`;
  it now reads the cell's batch, and the config goes through `scale_for_gbs`.
  Without this, models.json would have recorded 4,500 iters at the wrong token
  rate and the watcher would look for checkpoints at iterations that never
  existed. `family` stays size-free on purpose — it is the cross-size ladder
  identity, so `-b<N>` must not enter it or every rung becomes a singleton.
- **`pretrain_progress.py`** — `NAME_RE` captures the optional `-b<N>`. The
  group is *captured*, not skipped: the old batch-504 runs are still on disk
  under the name without it, and a pattern that merely tolerated the suffix
  would read both as the same cell.
- **`ladder_report.py`** — `LOG_RE` / `CELL_RE` likewise, plus `on_grid(m)`,
  which rejects a run whose batch is not its rung's current one. `_key` has no
  batch field, so without this both versions of a cell land on one key and
  "newest log wins" would decide the ladder by job id.

Verified: token budget identical at every rung (9.29B / 17.63B / … unchanged);
`--size 90M` gives 27,000 iters, GBS=84, MBS=7, LR 0.0014276 *unchanged*,
20 saves; `--size 175M` gives 25,620, GBS=168, MBS=7, LR 0.00121655 unchanged,
20 saves; both `[fresh]`, not `resume`; `due_iters` returns 12 checkpoints at
both rungs, the same as 350M; grid 180 runs after BT3's retirement, 58 `-b`;
`check_rules` 0 findings; `sync_docs` in sync.

### The hazard the rename removes

With iters going 4,500 → 27,000, a retrain under the *old* name would not have
overwritten the old checkpoints — it would have **resumed** them. The launcher
reads the old final checkpoint at 4,500 as a mid-run checkpoint of a 27,000-step
run and continues it, silently carrying a batch-504 optimizer state into a
batch-84 schedule. The `-b<N>` name makes that structurally impossible: a
different name is a different checkpoint dir.

## Not done: everything downstream

In dependency order. Each step's *verification* is what says it is finished.

### 1. Launch the 58 cells — done 2026-09-23

26 cells at 90M (3 nodes) + 32 at 175M (6 nodes) — 58, not 60, since BT3 was
retired from the grid the same day. Budget roughly 950–1,000
node-hours against ~570 for the originals: +39–40% per run, measured at 175M as
16.9 against 12.3 node-hours. `--partition preemptable` is what sets
`PRETRAIN_CHAIN=1`.

**`--size 90M` alone is not enough.** The launcher's filters default to scheme
A, `deep` and the single seed, so `--size 90M` matches 6 of the 26. The 58 cells
need one invocation per (scheme, arch); `--size 175M` picks up its own seed
triples without a `--seed` flag.

```bash
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
for f in "" "--arch shallow" "--scheme AT3" "--scheme AT3 --arch shallow" \
         "--scheme B" "--scheme B --arch shallow" \
         "--scheme ZH" "--scheme ES" "--scheme DCLMP" "--scheme FWEB"; do
  for s in 90M 175M; do
    python3.11 launch_trainings.py cscs --size $s $f --partition preemptable
  done
done
```

Counts per invocation, 58 in total:

| filter | 90M | 175M |
| --- | ---: | ---: |
| *(scheme A, deep)* | 6 | 12 |
| `--arch shallow` | 6 | 6 |
| `--scheme AT3` | 3 | 3 |
| `--scheme AT3 --arch shallow` | 1 | 1 |
| `--scheme B` | 3 | 3 |
| `--scheme B --arch shallow` | 3 | 3 |
| `--scheme ZH` / `ES` / `DCLMP` / `FWEB` | 1 each | 1 each |

`DCLMP` and `FWEB` will skip `[no data]` until their builds stage — that
is expected, and re-running the same command later picks them up.

**Verify:** the first finished 175M-L2-deep-seed1904 lands near **2.565** and
90M-L2-deep-seed1904 near **2.715**. If they do not, stop — the diagnostic runs
were single-seed and on one cell, and the grid path is not the `diag-` path.

### 2. `configs/models.json`

`python3.11 src/pretrain/sync_models_json.py --prune` adds the 58 `-b<N>`
entries and removes the 60 old names plus the 6 BT3 ones, leaving models.json
at exactly the 180 names the grid produces.

**Verify:** exactly 180 entries carry `source: snr-pretraining-predictivity`,
with nothing missing and nothing stale; a `-b84` entry reads
`num_iters: 27000`, `tokens_per_iter: 344064`, `tokens: 9289728000`;
`tokens` is unchanged against the entry it replaces.

### 3. The eval watcher

`auto_evals_cscs.py` walks `predictivity_cells()` and keys off `exp_name`, so
it follows the rename automatically — but it DID need a code change: it read
`schedule_for(configs[size])[0]`, the unscaled 8,540, and `due_iters` then
returned 3 checkpoints instead of 12, silently breaking the ten-checkpoints
rule for the whole 175M rung. It now calls `cell_schedule`, as do
`pretrain_progress._targets` (which would otherwise have called a 90M run
*done* at 4,500 of 27,000) and `sweep_cells`. Two further consequences:

- 30 of the 40 old checkpoint dirs start printing `on disk but not a grid cell
  — not counted` (`pretrain_progress.py`, not the watcher); the 10 old 90M ones
  are skipped silently because 90M is not in `EVAL_SIZES`;
- **a running watcher will not see any of this.** It imports `DATA_SCHEMES` and
  the grid once at startup and holds them in memory. Restart it after the
  rename, or it will keep evaluating the old names and ignoring the new ones.

`EVAL_SIZES` still excludes 90M (`launch_trainings.py:178`) — see step 5.

### 4. `src/pretrain/` docs

`README.md`, `CLAUDE.md` and `plan/small-to-large-predictivity-training-plan.md`
describe a ladder with one global batch. The generated blocks do not move (the
grid moved 186 -> 180 with BT3's retirement, so they were regenerated), plus
the prose: the sizes table, the cell-name
grammar, and the `90M-rung-anomaly.md` cross-reference, which should now say the
anomaly is fixed rather than open.

### 5. Bringing 90M into the analysis

Decided in principle on 2026-09-23; it is the largest downstream change and it
re-opens every published number, so it is staged last and separately.

The 90M rung is excluded in four places, and they are not independent:

| where | what it says |
| --- | --- |
| `launch_trainings.py:178` | `EVAL_SIZES = [s for s in LADDER if s != "90M"]` — 90M is never *evaluated*, so no results exist |
| `analysis/utils.py:298` | `ANALYSIS_SIZES = [s for s in EVAL_SIZES if …]` — inherits the exclusion |
| `analysis/check_rules.py:49` | `FORBIDDEN_SIZES = ["90M"] + …` — enforces it |
| `analysis/RULES.md` rule 10 | "Sizes 175M–1.7B" — states it |

Order matters: **`EVAL_SIZES` first**, because until 90M is evaluated the other
three changes would only produce empty columns. Then a full eval pass over the
27 new 90M cells (10 checkpoints × the `auto` group each), then `ANALYSIS_SIZES`
follows automatically, then `check_rules.py` and rule 10's wording.

Everything keyed by size gains a row or a column: every `rqNN_*` per-size table,
`scale_convergence*`, the per-language and per-benchmark grids, the first-clearing
-size and min-predictive-size figures, `documents/ladder-facts.json` and the
deck. Regeneration is `scripts/refresh_analysis.sh` with `FORCE=1`.

Two things to settle before that pass, not after:

- **Does the ladder's left anchor move the fits?** That is the whole reason the
  rung was investigated. Record the fitted α on 350M–1.7B before and after
  adding 90M; a large move is a finding, not a bug, but it must be stated.
- **The 3B rung stays above the reference** and keeps opting in with
  `above_reference=True`. Adding a rung below the reference does not change
  that, and rule 10's wording has to keep the two cases distinct.

### 6. What is *not* proposed here

- **Retiring the old checkpoints.** The 40 diverged runs stay on disk. They are
  the evidence for the anomaly appendix, and nothing reads them any more:
  `on_grid` gates both the loss side (`_train_logs`) and the eval side
  (`_cell_parts`), and the grid no longer names them. The eval-side gate is the
  one that matters most — `plot_benchmarks` groups by (size, arch, scheme, L),
  which has no batch field, so without it the old 175M checkpoints would have
  been AVERAGED into the new ones in a single line. No deletion is proposed,
  now or later.
- **Changing any rung above 175M.** 350M and up are at 1.7–8.1× the optimizer
  window and show none of the signature; their batch, schedule and names are
  untouched.
- **`beta3` as a fraction of the run** (option 1, and option 3's second knob).
  It reaches the same place at 90M and costs no extra time, but it changes an
  optimizer constant across the ladder. Option 2 was chosen because every
  optimizer constant stays identical at every rung; if option 2's grid runs do
  not reproduce 2.715 / 2.565, option 1 is the fallback and this document's
  step 1 is where that decision returns.
