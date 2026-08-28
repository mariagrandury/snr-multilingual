---
name: review-snr
description: Review pending snr-multilingual changes (unstaged, staged, or committed-but-unpushed) for bugs, necessity, doc drift, runtime errors, and scientific comparability with already-trained models. Runs before every commit — the commit hook refuses a tree this skill has not marked as reviewed. Use whenever the user asks to review, commit, or check pending work.
---

# Reviewing changes in snr-multilingual

This repo drives a live experiment: ~150 pretraining runs and thousands of
eval jobs across CSCS and Azure, feeding one scaling/predictivity fit that
becomes a paper. A bad change here does not throw — it silently spends node
hours, or quietly makes two model families incomparable. Review accordingly.

The repo is `/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual`.
Sessions are often started from its parent, so `cd` there (or use `git -C`)
before every command below — never assume the cwd.

## 1. Establish the scope first

The user may mean unstaged, staged, or committed-but-unpushed changes — or
some combination. **Do not assume.** Read the state and say what you are
reviewing before you start:

```bash
git status --short          # unstaged + staged + untracked
git diff                    # unstaged
git diff --cached           # staged
git log --oneline @{u}..HEAD && git diff @{u}..HEAD   # unpushed commits
```

If the user gave a count ("the last 12 commits"), reconcile it with what is
actually there and say so if it differs — a commit may have landed since.
Review the whole pending set, not just the newest layer. Work the user has
in flight but did not ask about (other unstaged files, untracked scripts)
is out of scope: list it in the report's Scope line and leave it alone.

## 2. What to check

**Bugs.** Read every changed hunk against the file around it, not in
isolation, and check that every helper the hunk calls exists with that
signature (`grep -n "def name"`) — a diff can call a kwarg the helper never
had. The recurring shapes here:

- Dead caches / guards: a memo dict that is read but never written, a flag
  computed but never used, an `if` whose branch can't be reached.
- Shell: `stat -c%s` on a symlink reports the *link*, not the target (use
  `-L`); `[ -f ]` follows symlinks but `[ -L ]` doesn't; unquoted globs that
  silently match nothing; `$0` inside an sbatch script resolves to
  `/var/spool/slurmd/job<id>/slurm_script`, so script-relative paths break —
  pair them with an absolute `/iopsstor/...` constant. A script that prints
  `FAILED` but exits 0 has no failure path: Slurm and every caller see
  COMPLETED.
- Anything that submits jobs: check the idempotency gate and the *failure*
  path. A gate that only asks "did this succeed?" resubmits a permanently
  failing job forever, once per watcher pass.
- Anything that copies data: check size, idempotency on re-run, and whether
  the source is a symlink into another build.
- Doc examples that are commands: a `--seed`/`--size`/`--langs` value must
  exist in the grid (`SEED_SINGLE`/`SEED_TRIPLE`, `SIZE_LANG_SETTINGS` in
  `launch_trainings.py`) — these flags are *filters*, and a value outside
  the grid silently matches nothing. When the intended value is not
  guessable, report it; never pick one.

**Necessity and correctness.** Every hunk should be doing work. Flag
leftover blank lines, no-op reformatting, stale trailing-newline churn,
"just in case" defensive scaffolding, and helpers that duplicate something
already in the repo (grep before believing a helper is new). A block of
session-specific launch commands in a README is a to-do list, not
documentation — say so.

**Docs.** Every changed behaviour must reach the docs that describe it, and
no further — concise, not verbose. Check, at minimum:

- module/function docstrings whose stated contract the diff changed;
- `src/pretrain/README.md`, `src/pretrain/azure/README.md`,
  `src/evals/README.md` for user-visible flags, defaults and new scripts;
- the `CLAUDE.md` files when a *failure mode* or invariant changed;
- `plan/` docs when the grid, budget or schedule changed — and when a diff
  *corrects a number* there (a cost share, a timing), grep the same doc for
  the old figure: the sentence three paragraphs down that still carries it
  is the bug that survives;
- generated blocks (`<!-- BEGIN generated: ... -->`) — never hand-edit them;
  `pretrain_progress.sync_docs()` is the idempotency check: it must report
  "already in sync" for every doc, and `git status` must be unchanged after.

Delete comments the change made false. A new script that no README mentions
is an incomplete change.

**Run it.** Execute every changed script, not just the happy path:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr   # matplotlib, numpy live here
```

- Python: `python -m py_compile` is table stakes, not a test. Actually run
  the CLI. System `python3` is 3.6 and will reject `from __future__ import
  annotations` — that is the wrong interpreter, not a bug in the diff.
- Shell: `bash -n`, then run it against **fixtures in the scratchpad** with
  `SRC=`/`DST=`-style overrides. Never let a test touch real data. The
  scratchpad can be swept mid-session — rebuild fixtures if a path vanishes
  rather than trusting an earlier run.
- Launchers/watchers: `--dry-run` only. `launch_trainings.py cscs --dry-run`
  starts neither the auto-eval watcher nor the plot refresh, so it is safe.
  `auto_evals_cscs.py --dry-run` still calls `sync_models_json.sync()` —
  back up `configs/models.json` first and check `git status configs/`
  after. Its dry-run output IS the gate quantification: count the
  `submit: eval-` / `submit: convert` lines and read the held-back list —
  and remember that self-repair steps (`fix_missing_dataset`) return False
  on dry-run, so "held back" there can be a dry-run artifact; the real
  watcher's `auto_eval_errors.json` and the HF dataset cache are the truth.
- Plots: `pretrain_progress.update_plots / plan_table / eval_progress` all
  take `out_dir=` — render into the scratchpad and `Read` the PNGs, do not
  overwrite the committed ones. `data_progress.py --out <scratchpad>`.
- `auto_evals_azure.py` needs the `az` CLI, which the login node lacks:
  compile + read it against the CSCS twin's semantics, and say in the
  report that it was not executed.
- Edge cases that have actually bitten here: empty/missing data dir, a
  symlinked source, a build still in progress (checkpoint present), an
  unreadable capstor file, a re-run (idempotency), a failed copy (exit
  code!), a cell absent from the grid, an empty task or result set.
- Prefer a check that produces evidence over one that produces an opinion:
  compare a regenerated artifact against the committed one; quantify a
  behaviour change ("how many checkpoints does this new gate re-submit
  *today*?") instead of reasoning about it.

**Scientific comparability.** The hard question: after this change, can a
model trained tomorrow still be compared with one trained last month in the
same table?

Fine — additive, or affects only cost/plumbing:
- a new benchmark, task or language *added* to an eval group (every
  checkpoint gets topped up; the gate is task-level for exactly this reason);
- a new pretraining transformation as a *new* axis (tokenizer grid, a new
  arch family, a new scheme) run as its own cells;
- walltime, staging, logging, progress plots, job ordering, retries;
- new diagnostic runs, as long as they are named outside the grid's cell
  pattern (`diag-*`) so `NAME_RE` / `sync_models_json` never pick them up.

Not fine without a full re-run of everything affected — flag it, loudly:
- changing architecture, LR law, init, optimizer schedule, batch/seq, or
  token budget for a family that is already (partly) trained;
- changing the language list, data scheme or mixture ratios of an existing
  L*X* family;
- changing the tokenizer, seed set, or checkpoint schedule of existing cells;
- *removing* a task from an eval group, or changing a task's config, so
  older results and newer ones mean different things;
- anything that rewrites or deletes existing checkpoints or eval results.

A plan document that *proposes* one of these is not a violation, but say so
in the report: it is a decision the user has to make deliberately, and the
answer is almost always "re-run the whole affected family, don't swap it in".

## 3. Fix vs propose

- **Fix directly**: real bugs, stale/incorrect comments and docs, dead code,
  formatting churn the diff introduced. Then re-run the checks.
- **Propose in the report**: anything non-cosmetic that changes behaviour,
  adds a feature, costs cluster time, or changes what Slurm reports for a
  job (exit codes, requeue). Give the evidence and the one-paragraph
  design, and let the user decide.
- **Never**: delete checkpoints or eval results, `rm -rf`, `git push -f`,
  submit cluster jobs, or commit without showing the message first.

## 4. The report

Write it in the chat (not to a file unless asked). Structure:

1. **Scope** — exactly what was reviewed (n commits / staged / unstaged),
   and what pending work was deliberately left out.
2. **Verdict** — one line.
3. **Bugs found** — each with file:line, why it is wrong, the evidence
   (command output, counts), and whether you fixed it.
4. **Fixed directly** — the list, so the user can `git diff` it.
5. **Docs** — what was stale and what you updated.
6. **Verification** — what you ran and what it showed. Name the edge cases,
   and name what you could *not* run.
7. **Comparability** — explicit verdict against the list above.
8. **Proposals** — non-cosmetic, evidence-first, ranked.
9. **Proposed commit message** — whenever anything is uncommitted (including
   fixes you just made). Subject in the repo's style: lowercase, imperative,
   `area: what changed`, ≤ 72 chars. Body: why, and the verification that
   backs it. Print it; do not commit until the user approves it.

## 5. Mark the reviewed state — last, after every fix

```bash
bash /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/.claude-shared/hooks/review-snr-gate.sh --mark
```

This fingerprints the pending tree (tracked changes vs HEAD + untracked file
contents) into `.git/review-snr.ok`. The same script runs as a PreToolUse
hook on every `git commit` aimed at this repo and refuses the commit unless
the tree still matches — so run it **after** the last edit of the review,
never before. If the user (or you) edits anything afterwards, the gate says
so and the review has to run again; that is the point.

## 6. When the user asks to commit

Treat "commit X" as "review X, then show me the message". Run this review
first, report, mark, print the proposed message, and wait. The user may
edit it. If the gate refuses a commit, do not work around it (no
`--no-verify`-style tricks exist for it, and none should be added) — run
the review.
