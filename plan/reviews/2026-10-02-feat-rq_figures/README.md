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
| `agentB/` | the last 13 commits' analysis code (`FINDINGS.md`) |
| `agentC/` | names, orphans, links and generated blocks (`FINDINGS.md`) |
| `agentE/` | the changed Python and shell run on fixtures (`FINDINGS.md`, `sub5a/findings.md`, `sub5b/findings.md`) |
| `NOTES.md` | the coordinator's notes |

Only the write-ups are kept. The helper scripts, fixtures and raw outputs
behind them (141 files: the per-agent `*.py` / `*.sh`, `.diff` / `.txt` / `.out`
captures, the `hand_*.txt` extracts, `names.py`, `orph.py`, `pc/`) were removed
on 2026-10-05, once every finding had been checked against the tree; they carry
the scratch paths of the session that ran them and are recoverable from commit
f4c81050.

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

Closed on 2026-10-05: the 90M/175M cells do not read the 92B FineWeb-2
rebuild (no `-b84`/`-b168` training log names `data-92B`; their FineWeb-2
draws stay under the 52B build); the two header-only rq05 per-group tables
are deleted (rq06's `transfer_da_all_by_group_mono_axis.csv` replaces them).

Still open (re-checked against the tree on 2026-10-05): the batch confound in
rule 10 (the 90M/175M LR is deliberately not rescaled for their smaller
batch), recomputing `rf_significance.csv` on the cluster (the nightly refresh
does), `auto_evals_azure.py` reading `groups.auto` with CSCS-only task YAMLs
(#8), the orphan listing's remaining noise (#15), the rule-16 names of the
legacy pools, `rq2_…_either_transformation` and the rq05
`intervention_da_size_by_*` tables (#17), rq06 `language_panel.py` keeping one
task per language where a family has several (#25), the legacy-pool
`compute_da.py` path (#26), `scale_convergence`'s `[nan, nan]` and the rq00
score curves off the shared grid (#28), and in the paper four unresolved cite
keys (`bhagia_establishing_2024` against the bib's `_2025`, `he2025scaling`,
`hoffmann_training_2022`, `wang2026metaeval`) and five unresolved `\ref`s.
