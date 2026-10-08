# Pre-merge review: feat/rq_figures -> main (HEAD d309cb72)

Produced by following the `review-snr` skill section by section (`.claude-shared/skills/review-snr/SKILL.md`; the Skill tool loaded `.claude/skills/review-snr`, which is a stale COPY, not the symlink CLAUDE.md prescribes — it lacks the "analysis rules first" paragraph, so I applied the `.claude-shared` text).

Skill steps I could not carry out, and why:
- **Section 3 (fix directly) only partly.** Seven files were fixed (list below). After that the permission classifier refused every further write, including a scratch test of one fix, so the rest is listed as "to fix" with the exact change.
- **Section 2 "run it" only partly.** A trimmed driver run in a scratch clone was refused by the classifier; the verification pipeline in the checkout was stopped inside rq02. Writer/reader name consistency is therefore verified statically, not by execution.
- **Section 5 (mark) deliberately not run**, as instructed. No commit, stage, push, `git mv` or delete.
- No launcher/watcher dry-run (cluster paths), no `compute_da.py`, `by_L.py`, `catalogue.py`, `search.py`, no driver.

CONFIRMED = reproduced by me or by a helper on committed files / a synthetic input. PLAUSIBLE = read, not run.

## 1. Scope
- **44 commits, no merges.** `git rev-list --count origin/main..HEAD` = 44. `origin/main` = 64ac5b1a = the remote's main (`git ls-remote`), and it is an ancestor of HEAD, so the merge is a fast-forward with no conflict.
- Local `main` (83ff0dca) is 321 commits behind `origin/main`, so `main..HEAD` = 365. "33" matches no cut I could find (31 commits precede the last 13; 28 are dated since 2026-09-23).
- `origin/feat/rq_figures` = 7046ef22: the last two commits (68d40d33, d309cb72) are unpushed.
- Left out: nothing was pending besides the verification pipeline's rewrites, which the coordinator restored.

## 2. Verdict
**Merge after fixes.** Nothing breaks the comparability of trained cells and the tree is internally consistent (`check_rules` 0 findings, tests pass, names resolve), but two published claims rest on invalid numbers (rq04 search significance counts; the rf significance table) and one launcher path would mis-train a grid cell on Azure.

## 3. Bugs found (most severe first)

### Launchers and reports
1. **Azure launch of a 90M/175M cell trains at batch 504 under a `-b84`/`-b168` name** — `src/pretrain/launch_trainings.py:1593` (49e24dad). The CSCS call at :1568 passes `gbs=`, the Azure call does not, so `megatron_args.sh:34` falls back to `GBS=504` with the 6x/3x longer schedule. Helper evidence: `lm-90M-L2-b84-deep-seed1904 | azure env: TRAINING_STEPS 27000 GBS <unset -> 504> => 55.74B tokens | cscs 9.29B`. I confirmed the missing argument by reading. Latent (only if 90M/175M are launched on Azure), but the result would pass `on_grid`. **Not fixed** (writes refused). Fix: `cell_env(cfg, c["size"], c["seed"], exp, blend, gbs=None if gbs == GBS else gbs)`.
2. **`run_tokens` used batch 504 with the rung-scaled target** — `launch_trainings.py:1499`, `pretrain_progress.py:754` (49e24dad). 90M draw read as 55.74B against 9.29B true. **Fixed.**
3. **`pretrain_progress.scan_runs` counted retired batch-504 dirs as unfinished runs of the current cell** — `pretrain_progress.py:257`. `NAME_RE` captured `gbs` and nothing read it. **Fixed**, checked on a fixture (old dir skipped, `-b168` and 600M kept).
4. **`scripts/nightly_ladder.sh:108-109` runs the 2 h analysis on yesterday's report when `ladder_report.py` crashes before writing its CSV** (53bc979b). Helper reproduced on a fixture: `report verified ... identical to today's run`, `FAILED` only at the end. Not fixed. Fix: `(( LR_RC == 0 )) || { FAILED+=(...); finish 1; }`.
5. **`arc_mt` (a probe candidate) is evaluated as an `auto` benchmark** — `src/evals/scripts/utils/configs.py:539` matches by prefix `arc_` (0f9781ad). The published report carries it at 12 checkpoints per cell. Additive, but unscreened and in the analysis pools. Your choice: list it in `auto`, or register it under a non-prefixed key.
6. `scripts/preempt_drain.sh:126-130` ranks evals with `schedule_for` (4500 steps at 90M), so `-b84` evals rank as "final" from iter 5400. PLAUSIBLE. Fix: `cell_schedule(cfg, size)[0]`.
7. `src/evals/.../make_rf_tasks.py:333-335` re-creates the retired `auto_rf`/`auto_rfgm` groups on every run; `derive_task_options.py:141,180` writes `tasks.json` without the lost-update guard the other generators got. PLAUSIBLE.
8. `auto_evals_azure.py:67` reads `groups.auto`, which now holds tasks whose YAMLs exist only on CSCS: the Azure gate can never empty. PLAUSIBLE, not executed.
9. Dead `--reformulated` flag in `auto_evals_cscs.py:854-861, 921-929` (always exits: both groups are gone).

### Analysis — the last 13 commits
10. **rq04 `search.py:642-647`: threshold-filter rows are tested and sit in the BH family** (c60d29b7). A filter is built from the top-K surrogates chosen on the same truth and data. Helper's null simulation: 5.5 % of screen rows at p < .05, 56 % of filter-indicator rows. On the committed CSV 39,576 of the 75,442 "hold at q < 0.05" rows are filter rows; without them the non-filter count is 30,887 not 35,866, and 18 bold cells of the paper tables lose bold. Fix: `has_p = main_rows["p"].notna() & (main_rows["subset_type"] != "filter")`; needs `search.py` + `make_surrogate_tables.py` re-run.
11. **rq04 `search.py:96`: DA-ckpt at 80 % and 90 % shares its checkpoints with four window surrogates not in `CIRCULAR`** (`window_kendall`, `sign_consistency_window`, `kendall_w_window`, `dior`). ρ with DA-ckpt jumps from 0.31–0.62 (10–70 %) to 0.79–0.82 (80/90 %); the README's "window sign consistency for DA-ckpt (0.73)" is partly circular. Needs `search.py` re-run.
12. **rq04 `search.py:14-15` (and `app_snr_new.tex:299`, `literature.md:111`): "one point per cluster makes the points independent, so its p is a valid test" is false.** 239 clusters are 21 benchmarks x 34 languages; ICC by benchmark 0.77 for `n_items`. Text fix, no number changes.
13. rq04 `catalogue.py`: zero-variance guards are `> 0` on float sums (`gain_over_noise` = 4.8e14, `cronbach_alpha` > 1 in 20 rows) at :268, 274, 233-236, 330, 416; chance is `1 / n_opt` not `task_chance` at :283 (truthfulqa_mc2 0.143 against 0.449 — rule 1); `language_consensus`/`item_total_corr` use a task's own `rf_` twin as a peer at :427-432; Kendall W has no tie correction at :237. All CONFIRMED by the helper; all need `catalogue.py` + `search.py`.
14. rq04 README:399 and `search.py:611` say "proxy-only" while `pseudo_ref_da` reads 1B (rule 11's declared exception is labelled in the paper, not in the README). No other surrogate reads the reference.
15. **`refresh_analysis.sh` orphan listing (7046ef22)**: printed a stray `  ORPHAN ` line on an empty list (reproduced), and listed 256 frozen 36-sweep files every FORCE run (`rq08/per_sample/` 248, `seeds_28_1797__vs__seeds_1904` 8). **Fixed.** Still open: `mktemp` file never removed; `make_rq_figures.py` skips unchanged copies, so `documents/paper/figures/*` would be listed as orphans; files no driver step writes (`allenai_snr_variants_per_task.csv`, `length_features.csv`, `sample_items.json`, `finetasks_overlap.csv`, the probe tables) are listed every run.
16. `check_rules.check_generators` (7046ef22) tests only whether the script's basename occurs in the driver. It passes `auto:rf-compare-probe (compare.py --tag probe)`, which only `probe.sh` runs and no driver calls: a rule-17 violation the checker cannot see.
17. Rule 16 names that still break the rule (d309cb72), none listed by `--names`:
    - `da_all_reliable_by_language_both_axes.csv` holds multi-axis rows only (600 rows).
    - The legacy pools' `da_all_per_task_both_axes.csv` (`all/external`, `custom_swissai_hf`, `seeds_*`) have no `axes` column.
    - `rq2_da_all_above_66_one_*` uses a filter token rule 16 does not define, and `..._above_66_either_transformation_*` puts filter before breakdown, the reverse of `scale_convergence_da_size_transformation_above_66_*`.
    - rq05 `intervention_da_size_by_{benchmark,language}_mono_axis.csv` hold all ten fractions.
18. Orphans that d309cb72 renamed as if current: rq06 `transfer_da_all_lines_mono_axis.{png,csv}` and `transfer_da_all_by_L_mono_axis.{png,csv}` were last generated 2026-09-20 (c28b4333). Their source, rq05's `intervention_da_all_by_group_ckpt10_mono_axis.csv`, has 0 rows (confirmed), so `rq06/panels.py:53` returns early every run while the README block embeds them.
19. `snr/metrics.py:72` docstring was rewritten by the rename's sed, making a dated sentence anachronistic and `snr/` differ from origin/main. **Fixed** (file is byte-identical to origin/main again).

### Analysis — earlier commits
20. **`rq00_task_reformulation/compare.py`: the committed significance table is void.** `rf_significance.csv` has `p` NaN in 2979 of 2979 rows and `sig` False everywhere (confirmed), because the per-item logs exist only on the cluster; the README prints `sig=0` in every cell, which reads as "no gain is significant".
21. **Driver order (a0a1e953): two rq00 steps read tables written later** — `above_random_example.py:145` (rq02 DA) and `reformulations_gate.py:107-113` (rq01 regimes, rq02 DA and reliable tasks) ran in the rq00 block. Under FORCE=1 they drew the previous refresh's tables. **Fixed** (moved after `reliable_tasks.py`).
22. `reformulations_gate.py:124` re-derives the gate as `== 1`, so NA (BPB, lambada) fails: 41 tasks dropped, DA-size mean at 90M 0.507 (n=230) against 0.563 (n=271) through `passes_gate`. `public_ladders.py:113-116` reads the ladder's DA-size ungated (0.555 against 0.578 gated). Rule 1.
23. `scale_convergence.py:646-651`: under `--by transformation --axes multi-axis` the per-axis lines are mono-axis pairs filtered by multi-axis reliability and named `_multi_axes` (above_66_size keeps 93 tasks against 72). Feeds the left panel of `rq2_da_all_above_66_*_transformation_multi_axes`.
24. `rq05/analyze.py`: the benchmark population of the intervention DA is never gated and says so nowhere (rule 1; pre-existing on main). `bpb_all` is not in `POPULATIONS`, so two by-group tables are written with 0 rows every run.
25. `rq06/language_panel.py:107` keeps one arbitrary task per language where a family has several (arc en, INCLUDE v2 es/fr/zh, belebele zh).
26. `compute_da.py:287` raises `KeyError` on every legacy parquet pool (40af906e): `design_axes` needs `L, arch, scheme`. Only bites if the 36-sweep is regenerated; CLAUDE.md says those pools "remain runnable".
27. `rq08/per_item_ladder.py` writes a column-less `per_item_summary.csv` (1 byte, committed) when the cluster-only store is absent, and the driver runs it unconditionally. `build_per_item_store.sbatch:9` logs into a directory that does not exist on a fresh clone.
28. Smaller, all CONFIRMED by a helper: `by_L.py:335` labels an L below `MIN_PAIRS` "(no pairs yet)"; `by_L.py:357` the mono-axis figure does not state its pair set; `da_explainer.py:270` hard-codes "≈ 0.47 … the seed null" where the CSV gives 0.56–0.58; `scale_convergence.py:569` prints `[nan, nan]`; `rq00/panels.py:195` score curves are not on the shared grid; `rq00/panels.py:124` filter comment is stale (99c8346f gave every task an item count in the same commit).
29. **`autodoc.replace_block` swallowed text when one block key is a prefix of another** (`scale-convergence` / `scale-convergence-L8`, added on this branch). Reproduced on a synthetic README: the earlier block and the hand text between them are replaced. Harmless today only because the short key comes first. **Fixed**; the new pattern matches the same spans on all 128 existing keys.

## 4. Fixed directly (unstaged; `git diff` shows exactly these 7 files)
| file:line | what was wrong |
|---|---|
| `src/pretrain/launch_trainings.py:1499` | `run_tokens` multiplied the rung-scaled target by 504 |
| `src/pretrain/pretrain_progress.py:257` | retired batch-504 run dirs counted as runs of the current cell |
| `src/pretrain/pretrain_progress.py:754` | same `run_tokens` error |
| `src/signal-and-noise/run_all_predictivity.sh:77-82 -> 127-132` | two rq00 steps ran before the tables they read |
| `src/signal-and-noise/run_all_predictivity.sh:2-3, 111` | header said L up to 100, five schemes, "AT3/BT3" |
| `src/signal-and-noise/analysis/autodoc.py:52` | prefix-key block swallow |
| `scripts/refresh_analysis.sh:129-131, 135, 139` | orphan listing: frozen pools, stray line |
| `src/signal-and-noise/snr/metrics.py:72` | docstring restored to origin/main |
| `.claude-shared/skills/review-snr/SKILL.md:42` | "no 90M" contradicted rule 10 |

Re-checked after the edits: `py_compile` and `bash -n` pass; `check_rules.py --quiet --names` = 0 findings. The `replace_block` fix was tested before applying (pattern comparison on every README); its post-edit fixture test was refused. I did not verify that no step between rq00 and rq02 reads the two moved steps' outputs (that grep was refused).

Nine rq01 tables a helper had rewritten in the checkout differed from HEAD by at most 9e-15; I restored them with `git checkout --`.

## 5. Docs (stale; not updated — writes refused)
- `analysis/RULES.md:57` "34 of the Gemini-rewritten ones" -> 33 (mask: 33/43).
- `RULES.md:96, 104, 108-110`: "17 languages", "best per-language r is 0.24", "r = 0.194 at n = 324 / 0.138 at n = 2906" are not reproducible from HEAD tables (34 languages; pooled n = 1,524).
- `RULES.md:147-149` "ES is capped at 1B" contradicts rule 9 and its own table; `analysis/README.md:35, 50, 81` (rq10 "waiting", "175M–1.7B", "L2 ES stops at 1B").
- `rq02 README:1060-1075` seed bullets contradict the auto table 8 lines above (175M 0.72/0.78/0.83 written, 0.60/0.68/0.74 in the CSV); `:269` old names; `:204` dated 09-23 numbers.
- `rq04 README:60` states snapshot 2026-09-23; `:276-296` FineTasks findings off (13 %/38 % written, 23 %/43 % in the CSV; "no statistic exceeds 0.33" false, `r2_trajectory` 0.35); `:779` `surrogate_by_language.csv` does not exist.
- `rq01 README:122` "306 points, 25 families" -> 346, 26; `:386` `rq_facts.json` does not exist.
- `rq00_task_reformulation/README.md`: no snapshot date; heading "data not yet produced" is false.
- `src/signal-and-noise/CLAUDE.md:226, 231, 245`: `175M…`, "five schemes", `small_sizes 175M–1B`. `configs/models.json` pool description still lists BT3.
- `src/pretrain/README.md:4, 97, 166`, `src/pretrain/CLAUDE.md:39, ~205, 318`, `azure/README.md:536`, `src/evals/README.md:117`, `src/evals/CLAUDE.md:227`: 3B settings, "eight" schemes (seven), "every size but 90M", "every 2nd checkpoint", bbq still a candidate.
- Paper: 11 unresolved `\cite` keys in sections `main.tex` inputs (e.g. `hoffmann_training_2022`, `hagele_scaling_2024`, `bhagia_establishing_2024` vs bib `_2025`); 4 unresolved `\ref`; `app_snr_new.tex` keys all resolve, no duplicate bib keys. `main.tex` sits in `sections/` and inputs `sections/...`; `math_commands.tex` and the ICLR style are not tracked. `verified_results_provenance.json:29-30` still hashes pre-rename paths, so `verify_paper_results.py` has not run since d309cb72.
- Links: 0 dead markdown links and 179 image embeds all resolve.

## 6. Verification
Ran, on a clean sparse checkout of HEAD with the committed LFS content:
- `check_rules.py --names`: 0 findings, exit 0. `unittest discover -s tests`: 17 tests OK.
- `py_compile` on all 88 changed `.py`, `bash -n` on all 14 changed `.sh`, JSON load of changed configs: clean.
- `make_surrogate_tables.py` and `make_appendix_tables.py` re-run: no diff against the committed `.tex`, so both generated blocks match their generators.
- d309cb72: 395 renames at 100 % similarity plus 68 modified files. The four `rq00_task_reformulation` outputs are valid (PNGs open, SVG parses, CSV 2,979 rows) and the pipeline's re-run of `compare.py` left that folder unmodified.
- Names: every artifact name written as a literal resolves to a tracked file. For the f-string families of rq02 (scale_convergence, by_L, paper_rq2) I enumerated 219 expected names from the code and the driver's invocations: 0 written-but-untracked, 0 tracked-without-writer. `grids._csv_path` and `check_png_csv` use the same regex.
- rq00 and rq01 outputs regenerated by the pipeline before it was stopped differ from HEAD only in the last float digit (11 tables, max 9e-15).
- Open item (a): **no staleness.** All 354 `rfgm_belebele` mask cells went NA -> 1, none to 0, and every reader in rq03/rq05–rq10 treats NA and 1 alike; rq03's DA columns equal rq02's exactly.
- Merge readiness: no secrets, no `/Users/` or `/private/tmp` paths, no junk files; all 1,708 LFS objects present locally.

Could not run: any driver, the long generators, launchers and watchers, `auto_evals_azure.py`, anything on the cluster. My first attempt at a full worktree filled the disk (rolled back by git; the pipeline's log showed no error).

## 7. Comparability
**One flag, no break.** `49e24dad` changes batch and step schedule at 90M/175M (504 -> 84/168, 4500 -> 27000 steps), done as a full re-run under new names; old runs are excluded by `on_grid` and the published report holds 56 new-batch and 0 old-batch cells. What remains is yours to own: the LR is not rescaled, so size is confounded with batch at the two smallest rungs, and RULES rule 10 does not call it a limitation.

Everything else is additive or plumbing: hyperparams byte-identical for every size; 0 of 113 common cells change any training key; `auto` grows 14 -> 39 benchmarks with none removed; 343 task YAMLs added, 0 modified; `bbq` leaving `auto_probe` is the only removal, from a group born on this branch; the twelve-checkpoint due set is a subset of the old one at every size. Unverified (cluster): whether any `rfgm` eval predates the validator fix of 31565675.

## 8. Proposals (ranked)
1. Fix the Azure `gbs` call (bug 1) before any Azure launch.
2. rq04: apply bugs 10, 11, 13, then `python analysis/rq04_surrogates/catalogue.py --pool predictivity && python analysis/rq04_surrogates/search.py --pool predictivity`, then `make_surrogate_tables.py`. Stale until then: `surrogate_*.csv`, the rq04 README counts, both paper tables.
3. Recompute `rf_significance.csv` on the cluster, or have `compare.py` write "not computed" instead of `sig=0`.
4. Gate rq05's benchmark DA, `reformulations_gate.py` and `public_ladders.py` through `passes_gate`; regenerate with their own scripts (fast) and re-read the README numbers.
5. Decide `arc_mt`; state the batch confound in RULES rule 10 and the paper.
6. Renames, your call: `git mv .../da_all_reliable_by_language_both_axes.csv .../da_all_reliable_by_language_multi_axes.csv` (+ writer `reliable_tasks.py:335`); `rq2_da_all_above_66_one_*` -> `..._above_66_either_*`.
7. Flag or remove the four rq06 `transfer_da_all_*` orphans; make `check_generators` compare the marker's full command.
8. Squash the wip commits (`a0a1e953`, `53bc979b`, `8a695079`, `34a8990b`, `0f15cefc`) only if you rewrite history anyway: three unpushed commits aside, the branch is published, so I would not.
9. Write CSVs with a fixed float format so a re-run on another machine does not produce 1e-15 LFS churn. Move the 15.6 MB `src/pretrain/ladder_report_bpb.png` to LFS.

## 9. Proposed commit message (for the seven fixed files)
```
review: fix batch-scaled run tokens, driver order, autodoc block match

- launch_trainings / pretrain_progress: run_tokens at the rung's own
  batch; scan_runs skips the retired batch-504 dirs
- run_all_predictivity: above_random_example and reformulations_gate
  run after the rq01/rq02 tables they read; header matches the grid
- autodoc.replace_block: match the BEGIN marker whole (a key that
  prefixes another swallowed the text between them)
- refresh_analysis: orphan listing skips the frozen pools, no stray line
- snr/metrics.py docstring back to upstream; skill text follows rule 10

check_rules 0 findings; py_compile and bash -n clean.
```

## Last 13 commits
| commit | verdict |
|---|---|
| 774339c2 pair-set twins | OK; names consistent |
| 99c8346f script fixes | OK; stale comment at `rq00/panels.py:124` |
| 37fb9da9 paper copies | OK |
| c60d29b7 rq04 catalogue + search | **fix before citing**: bugs 10–14 |
| 32e6d55a doc corrections | OK |
| 4ba46b8a READMEs/RULES | stale numbers (section 5) |
| b031418b paper | unresolved cites/refs |
| 14798269 outputs regenerated | OK; rq00/rq01 reproduce to 1e-15 |
| cf3ca980 delete renamed | OK; nothing reads the deleted names |
| b64e4185 appendix + generator | OK; generated blocks match |
| 7046ef22 rules 16/17 | orphan listing fixed; checker gaps (15–17) |
| 68d40d33 remove 29 orphans | OK; rq06 orphans remain (18) |
| d309cb72 rule-16 names | 395 pure renames; bugs 17–19; "driver re-run confirms" not verified |
