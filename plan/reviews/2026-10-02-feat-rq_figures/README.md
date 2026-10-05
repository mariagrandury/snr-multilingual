# Pre-merge review of `feat/rq_figures` (2026-10-02)

The independent review of the branch before its merge into `main`, run with
the `review-snr` skill on HEAD d309cb72, kept here so its evidence survives
the session that produced it.

| file | what it is |
|---|---|
| [REVIEW.md](REVIEW.md) | the report: scope, verdict, every bug with its evidence, docs, verification, comparability, proposals |
| [agentC/NUMBERS.md](agentC/NUMBERS.md) | hand-written numbers of rq00–rq02, rq04, rq06, rq10, `RULES.md`, `CLAUDE.md` and the paper checked against the committed CSVs (56 checks) |
| [numbers2/NUMBERS2.md](numbers2/NUMBERS2.md) | the same for rq03, rq05–rq09, `analysis/README.md` and `CLAUDE.md` (87 checks) |
| `agentA/` | launchers, reports and comparability (`FINDINGS.md`, `COMPARABILITY.md`) |
| `agentB/` | the last 13 commits' analysis code (`FINDINGS.md`) and the scripts behind its checks |
| `agentC/` | names, orphans, links and generated blocks (`FINDINGS.md`), with the extracted hand text per README (`hand_*.txt`) |
| `agentE/` | the changed Python and shell run on fixtures (`FINDINGS.md`, `sub5a/`, `sub5b/`) |
| `NOTES.md`, `names.py`, `orph.py`, `pc/` | the coordinator's notes and the name / orphan checks |

The scripts carry the scratch paths of the session that ran them and are kept
as a record of how each number was obtained, not to be re-run as they are.
Copies of repo files the helpers diffed against (`launch_trainings.py`,
`models.json`, `tasks.json` snapshots, file listings) were left out: git
holds them.

## What was done with it

Applied in the working tree after the review (uncommitted until approved):

- launchers: the Azure call passes the rung's batch; `run_tokens` and
  `scan_runs` at the rung's own batch; `preempt_drain.sh` ranks by
  `cell_schedule`; `nightly_ladder.sh` stops when `ladder_report.py` crashes;
  `make_rf_tasks.py` no longer recreates the retired `auto_rf*` groups;
  `derive_task_options.py` writes through the lost-update guard.
- rq04: threshold-filter rows out of the BH family; DA-ckpt cells inside the
  noise window dropped against the window surrogates; the catalogue's float
  guards, `task_chance`, twins excluded as peers, Kendall's W tie-corrected;
  the "valid test" wording corrected; `pseudo_ref_da` labelled as reading 1B.
- rule 1: `reformulations_gate.py`, `public_ladders.py` and rq05's benchmark
  population gated through `passes_gate`.
- `compare.py` reuses the cluster's twin accuracies when the harness results
  are absent and prints `sig=n/a` instead of `sig=0` for untested cells;
  `per_item_ladder.py` writes nothing without its store; the probe comparison
  joined the driver and `check_rules` checks a block's `--tag`.
- `scale_convergence --by transformation`: per-axis lines filtered on the
  mono-axis reliability; `by_L` labels and pair set; `da_explainer`'s tie null
  attributed correctly; rq06's README block stops embedding the transfer
  figures nothing draws any more.
- `da_all_reliable_by_language_both_axes.csv` renamed `_multi_axes` (it holds
  multi-axis rows only).
- every stale number of the two audits corrected in the READMEs, `RULES.md`,
  the `CLAUDE.md` files and the paper (RQ1, RQ3 rewritten from
  `rq3_surrogates.csv`, the reformulation paragraph); four missing bib
  entries added and three cite keys fixed.

Decided by the user on 2026-10-03 and applied:

- `arc_mt` stays in `auto`, now listed by name rather than reached through
  the `arc_` prefix (the task selection is unchanged, 1,057 tasks).
- the dead `--reformulated` flag is removed from `auto_evals_cscs.py`.
- `above_66_one` is renamed `above_66_own` (each panel filtered on its own
  axis; "either" would have been wrong).
- rq06 keeps its rule-2 exception: the transfer-decision table is computed in
  rq06 on every language's BPB, so `transfer_da_all_*` are regenerated
  instead of deleted.
- CSVs are written at twelve significant digits; `ladder_report_bpb.png`
  moved to LFS.

Still open: whether the 90M/175M cells at A-L15, A-L50 and B-L15 read the
92B FineWeb-2 rebuild (a cluster log check), the batch confound in rule 10,
recomputing `rf_significance.csv` on the cluster, rq06 `language_panel.py`
keeping one task per language where a family has several, the rq05
`intervention_da_size_by_*` names, the legacy-pool `compute_da.py` path, two
unresolved cite keys and three placeholder `\ref`s in the paper, and the
two header-only rq05 per-group tables nothing writes any more.
