# Status 09-04 — analysis side

The signal-and-noise module now runs on the predictivity ladder (branch
`feat/snr-update`). Nothing here changes the training plan; it changes what
happens to the ladder report once it is published.

## What changed

- `src/signal-and-noise` reads one file, `ladder_report.csv`, the wide table
  `ladder_report.py` publishes to `msnr-data/ladder-report` (loader:
  `snr/download/ladder.py`). Four pools over it in `configs/models.json`:
  `predictivity` (the grid, seed 1904), `predictivity_seeds` (every seed),
  `predictivity_seeds_train` / `_test` (the ×3 cells split by seed).
- rq00–rq05 run unchanged on the ladder; rq01 (decision accuracy) and rq06
  (proxy size × L) are new directories. `bash run_all_predictivity.sh` runs
  everything; each RQ README takes its numbers from its script.
- Fixed on the training side: the `trained` flag in `ladder_report.py`
  (a first-two-letters match — Japanese, Spanish, Turkish and others were
  never "trained", Estonian was whenever Spanish was), the watchers' due
  rule (the 1B cells on the 2,287 grid were never evaluated), the FLOPs
  convention (6 (N_non-emb + d V) D everywhere, with a recorded basis), the
  registry (the scheme-B and 1B-seed entries), the seed order in
  `transform_effects`.
- The report: `documents/snr_predictivity_report.pdf`.

## Blocked on

- `msnr-data/ladder-report` is not on the Hub yet, so every README's results
  block is a placeholder. Once `ladder_report.py --plot --publish --push-hf`
  has run: `cd src/signal-and-noise && bash run_all_predictivity.sh` (or
  `SNR_LADDER_DIR=<capstor copy> bash run_all_predictivity.sh`), then commit
  the regenerated READMEs.
- Seed noise needs the ×3 replicate cells (unstarted on 09-01); rq06's
  effect-vs-noise table and rq02's seed holdout are empty without them.
- rq03 needs the AllenAI CSV, a git-lfs pointer in a plain clone
  (`git lfs pull`).

## Next

1. Publish the ladder report, run the pipeline, read rq06's proxy grid
   against the plan's predictivity question.
2. Decide on the legacy code listed in `src/signal-and-noise/CLAUDE.md`.
3. The proposed RQs in the report ("Proposed research questions").
