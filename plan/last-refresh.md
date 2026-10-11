# Last refresh before the figures and paper-story work

Quick status of the remaining tasks. Not committed; updated as agents report.
Last update: Sat 10 Oct, 22:30 CEST. ETAs are estimates (CEST).

## Steps to the full final refresh

| # | Step | Status | ETA |
|---|---|---|---|
| 1 | Website: findings pages, landing, ladder, new views | done, committed `ecba4e16` | — |
| 2 | No eval watcher is running since 18:58 (all evals are done); the 00:00 nightly starts a fresh one on the fixed lock code. rq06 palette crash, rq05 Muon/SwiGLU decisions, curves split into two 10 h jobs, purged-checkpoint fallback in the watcher | done, uncommitted (lands in step 11) | — |
| 3 | Promoted benchmarks (piqa, winogrande, boolq, …), the all-languages backlog, the 350M K=1 gap (12 jobs) and the six 3B probe evals | done: all eval jobs finished (last 17:52) | — |
| 4 | social_iqa | removed from `auto` (your call: not worth ~2,600 jobs before the refresh); its task entry and labels stay | — |
| 5 | Muon 1B/1.7B benchmark evals | done (no checkpoint missing) | — |
| 6 | Muon 1B/1.7B bpb: the 26 missing tenths, scored by 26 one-checkpoint jobs | done (all completed ~19:30) | — |
| 7 | Prose pass with /draft + AGENTS.md (READMEs, generated appendix pages, slides); appendix generators rerun; no overflowing page | done, uncommitted | — |
| 8 | Setup appendices rewritten (terminology, ladder, data mixtures, evaluation, chance threshold, ranking agreement); ladder tables: no Muon rows, short seeds caption; 3B K=50 kept (it is trained and rq10 uses it) | done, uncommitted | — |
| 9 | Model card (self-contained) + new Dataset Card appendix | done, uncommitted | — |
| 10 | Review of the uncommitted tree and of the 29 unpushed commits | done; fixes below go in before committing | — |
| 11 | Pre-commit fixes and the consistency pass on the current values: website, figures + slides, paper main text + snr_predictivity_report.md done; analysis READMEs and research-questions.html running; Limitations section now \\input; then the atomic commits | running (2 agents) | commits after 13's refresh is submitted |
| 12 | Push `feat/rq_figures` as is (fast-forward; no new anonymity leaks) | after 11 | Sat 10 Oct ~21:15 |
| 13 | Report published (20:16), bBPB store rebuilt (20:29), majority baselines re-derived (done); the refresh + two curves jobs are submitted automatically (detached, status in logs/manual-refresh/status.txt) as soon as the three fix agents finish | waiting on 13b | refresh ~23:30–03:30, curves Sun ~13:30 |
| 13b | Figure-review fixes: main text + agreement applied; further, measurability and rq09 (item count + skill) being finished (the session moved login node at ~22:10 and lost the /tmp scratchpad; agents rebuild their remaining patches from context, shared scratch now on iopsstor) | applying | Sat 10 Oct ~23:15 |
| 14 | Post-refresh delta: re-check the values logged by the consistency pass (scratchpad/consistency_*.md) against the refreshed outputs, new-task DA re-check, final review section below | after 13 | Sun 11 Oct ~01:00 (curves-only figures ~10:00) |
| 15 | Commit 14 | after 14 | Sun 11 Oct ~05:30 |

## How the new evals changed the key findings (preliminary)

Compared: the 2026-10-08 report the paper quotes against the 2026-10-10 04:31
report (promoted benchmarks partly in, all-languages backlog partly in, Muon up to
600M). Final numbers come with the refresh tonight (Muon 1B/1.7B, the 140 late
cells, the 3B probe, social_iqa).

- **More of the suite is measurable, not more reliable.** The promoted
  benchmarks pass the gate often, so the trained tasks above chance rise from
  37 % to 40 % at 90M and from 60 % to 64 % at 1.7B (377 and 599 of 933, was
  311 and 507 of 841), and 607 → 772 of 927 at 1.7B → 3B (was 522 → 687 of 841).
- **Decision accuracy does not improve.** Accuracy DA-size against 1.7B stays
  at about 0.5 (3B check 0.48–0.53, was 0.47–0.52); the twins move by +0.01.
  Per-language BPB stays high but its weakest proxy against 3B drops from 0.81
  to 0.78 as more languages enter.
- **Scaling is slightly less clean.** Tasks predictable along size and
  training: 78 % → 74 % (311 of 420), median size-fit R² 0.88 → 0.87; the new
  tasks are noisier than the old core.
- **SNR as a predictor gets weaker.** On accuracy alone SNR's ρ with DA drops
  from 0.35–0.43 to 0.31–0.36 and leads only up to 350M (at 1B the inverted noise
  leads); the new families (kobest, noreval_extra, toksuite) are the lowest-SNR
  ones, and benchmark tasks below the noise at 1.7B rise from 7 % to 13 %.
- **SNR in the appendix catalogue holds, a bit weaker.** Best proxy-only ρ with DA
  0.82–0.83 → 0.80–0.81; rel-std SNR 0.79 → 0.75; on per-language BPB it
  improves, 0.27 → 0.39.
- **External check gets weaker but broader.** With 9 shared tasks instead of 6,
  the agreement with DataDecide drops from ρ 0.83 to 0.77, still significant.
- **bBPB gains ground:** in the recipe (rq11) bBPB now beats the original
  accuracy at all five proxy sizes (original accuracy 0.55–0.59 → 0.53–0.57).
- **New and more interesting:** the rq05 BPB table now fills (temperature
  0.94–0.99, activation 0.92–0.99, depth 0.96–1.00 except 350M/600M), and the
  600M dip shows up for activation too. Activation pairs share one shape, so
  the dip is not only the depth pair's size mismatch: it points at the 600M
  rung itself (its seed draw), which the paper's 600M paragraph should say.

- **The new benchmarks are not reliable at small proxies yet.** At 1B several
  MMLU-translation languages and TokSuite-math beat the permutation null (DA
  0.65–0.79, FDR q ≤ 0.04), but no new task with DA-size ≥ 0.75 at 90M–350M
  survives FDR. WinoGrande, noreval_extra, KoBEST and canonical TokSuite are
  noise. Coverage at small sizes is still thin (2–9 cells), and the new tasks
  have no bBPB twins until the store is rebuilt tonight.

Net: the new evals widen coverage and make the measurability story stronger,
but they do not raise DA or SNR; the headline contrast (benchmark accuracy near
chance, per-language BPB high) is unchanged.

### Checked by the popular-vs-niche analysis (20:05)
- No key finding changed sign or conclusion; every shift comes from the 92 promoted tasks entering the population (recomputing on the old task set gives back the 10-08 values).
- Corrections to the bullets above: the README DA-size headline (0.51–0.57) covers only tasks with a bBPB twin, so it excludes the new benchmarks; "bBPB gains ground" in rq11 is really "accuracy lost ground" (bBPB unchanged, accuracy fell as untwinned tasks entered); the rq05 BPB table filled because of a code change (transformations with fewer than 3 items dropped), only the activation row is new data; at 600M SNR and inverted noise tie (0.3228 vs 0.3227).
- Not listed before: the rq04 per-language BPB surrogate panel changed (inverted noise ~-0.1 → +0.39 at 90M–350M); rq07's top-5 overlap with DataDecide drops from 5/5 to 3/5; rq13's English lead shrinks from 1.3 to 1.1 points.
- Popular vs later-added benchmarks: per task the original ones look better (DA-size 0.64 vs ~0.50), per benchmark family the gap mostly goes (0.57 vs 0.51–0.56, CIs overlap); it is carried by the LM-like originals (HellaSwag, LAMBADA, XStoryCloze, MultiBLiMP). Item count is the division that explains DA, DA-ckpt and R² best, also within families; skill (LM-like vs knowledge) is second.
