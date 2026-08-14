# Plan: `refactor/shared_config` branch bug fixes

> Continuation of a long code review on the `refactor/shared_config`
> branch. Picks up after C1 was committed (`869b60d` —
> `pretrain: hoist is_valid_iter_dir as canonical checkpoint validity helper`).
> Below is the full remaining punch list with per-item specs, decisions
> already made, and the open question (H4) that blocks one item.

## Status overview

| Item | Title | Status | Notes |
|---|---|---|---|
| C1 | Hoist `is_valid_iter_dir` as canonical helper | **DONE** (commit `869b60d`) | 4 call sites delegate via `--is-valid` CLI |
| C2 | Marker-stuck-on-failed-resume (loose validity check passes some unloadable iters) | **SKIP** (user-decided) | Documented as KNOWN LIMITATION in docstring; no quick fix |
| C3 | Fail-fast in `evaluate.sbatch` on judge-API tasks without `CSCS_SERVING_API` | PENDING | Reads judge-task list from `configs/tasks.json` groups |
| H1 | (skipped per user) | **SKIP** | — |
| H2 | (skipped per user) | **SKIP** | — |
| H3 | Add 2 posttraining pools | PENDING | `posttraining_hf_reference`, `posttraining_local_distill` |
| H4 | HF-postraining eval launch path | **AWAITING USER DECISION** | Option A (doc-only) vs Option B (unify launchers) — see below |
| H5 | Force chat template via `CHAT_TEMPLATE_OVERRIDE` in mode handler | PENDING | All posttraining evals must use chat template |
| M1 | New `full_eval` / `main_eval` / `10_ckpts` / `da_ckpts` derivation | PENDING | Percentages-based; see spec below |
| M2 | Delete dead `HF_STAGING_BASE` constant | PENDING | Use `HF_LOCAL_BASE` only |
| M3 | Add `num_key_value_heads` to `models.json` | PENDING | Drives per-cell TP for vLLM (bug 14) |
| M4 | `cache-datasets.sh` minimal prewarm script | PENDING | One-shot HF dataset cache primer per task group |
| M5 | (skipped per user) | **SKIP** | — |
| L1 | (skipped per user) | **SKIP** | — |
| L2 | `convert-snr.sh` exit 1 when line_count==0 | PENDING | Fail-fast on empty plan |
| L3 | (skipped per user) | **SKIP** | — |
| L4 | Assert `full_eval`, `main_eval` ⊆ `all` in `_meg_stage` | PENDING | Logic sanity check |
| L5 | Log unrecognized NAMEs to `models_dropped.log` | PENDING | Visibility for `push_all_results.py` |
| L6 | `convert-snr.sh` mode 3 → use `iters_for()` instead of legacy `ITERS_*` lists | PENDING | Single source of truth for iter sets |
| **NEW** | seed-1797 / seed-28 marker drift + empty-shell `iter_*/` dirs | OPEN | See "Discovered during review" section below |

**Ordering for the new session:** C3 → H3 → H5 → M1 → M2 → M3 → M4 → L2 → L4 → L5 → L6. H4 needs the user's design call before starting. The marker-drift issue is independent — fix when ready.

---

## Branch context

- Branch: `refactor/shared_config`
- Repo root: `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/`
- Last commits before this session:
  - `859aaf5` — `tasks: split posttraining into safe + LLM-as-judge subsets`
  - `c5aee5b` — `hf_base_runner.sh: fix per-checkpoint idempotency SKIP misfiring on _eval_status.py crash`
- Commit added in this session:
  - `869b60d` — `pretrain: hoist is_valid_iter_dir as canonical checkpoint validity helper` (C1)

---

## Hard rules to keep (CLAUDE.md, memory)

- **Never `rm -rf`, `mv`, delete checkpoints / eval results, force-push.** Skip with a warning on corrupt state; never auto-clean.
- **Reuse existing code aggressively; keep new code minimal.** No defensive scaffolding, no "just in case" wrappers, no premature abstraction.
- **Use absolute `/iopsstor/...` paths** in cluster-side docs/scripts.
- Login node python is **3.6** (broken for 3.7+ syntax) → use `python3.11` (the `snr` conda env). Container has python 3.12 → use `python3`.
- Container does NOT inherit host env vars across `srun --environment=...` — thread anything new via `INNER_EXPORTS` (eval bug 5).

---

## C3 — Fail-fast on judge-API tasks without `CSCS_SERVING_API`

**Why.** Today (`evaluate.sbatch:98–99`) the script only WARNs when `CSCS_SERVING_API` is unset. If the user submits a posttraining eval that includes `harmbench` or any `aya_redteaming_*`, the run silently proceeds and crashes 12 h later inside lm-eval. The judge-task list is now a tasks.json group (added in `859aaf5`) so we can intersect at submit time.

**File.** [src/evals/scripts/evaluate.sbatch](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/evaluate.sbatch)

**Spec.**

1. Remove the existing warn block (lines 98–99):
   ```bash
   warn_if_empty CSCS_SERVING_API \
     "Optional key missing. LLM-as-a-judge benchmarks via CSCS serving API will fail."
   ```
2. Insert a fail-fast block AFTER `$TASKS` is fully resolved (i.e. after the split logic, immediately before the `if [ -f "$TABLE_METRICS" ]` block, around line 131):

   ```bash
   # Fail-fast: if $TASKS contains any LLM-as-judge task that needs
   # CSCS_SERVING_API and the key is unset, die NOW rather than 12h later
   # when lm-eval crashes mid-evaluation. The judge-task set is the
   # `posttraining_llm_judge` group in configs/tasks.json, so adding /
   # removing judge tasks doesn't need a code change here.
   JUDGE_TASKS_JSON="$(dirname "${BASH_SOURCE[0]}")/../../../configs/tasks.json"
   if [[ -f "$JUDGE_TASKS_JSON" ]]; then
       judge_tasks=$(python3 -c "import json; print(' '.join(json.load(open('$JUDGE_TASKS_JSON')).get('groups', {}).get('posttraining_llm_judge', [])))")
       requested=",${TASKS},"
       needs_judge=()
       for t in $judge_tasks; do
           [[ "$requested" == *",${t},"* ]] && needs_judge+=("$t")
       done
       if (( ${#needs_judge[@]} > 0 )) && [[ -z "${CSCS_SERVING_API:-}" ]]; then
           die "ERROR: \$CSCS_SERVING_API is unset, but \$TASKS contains LLM-as-judge tasks that require it: ${needs_judge[*]}. Set CSCS_SERVING_API (env or scripts/cscs_serving_api_key.txt) or remove these tasks from \$TASKS."
       fi
   fi
   ```

**Path note.** `evaluate.sbatch` lives at `src/evals/scripts/evaluate.sbatch`; `${BASH_SOURCE[0]}/../../../configs/tasks.json` resolves to the project-root `configs/tasks.json`. The `python3` call runs on the compute node's host shell BEFORE any `srun --environment=...`, so this needs python3 on the bare node (it is present).

**Test.** Stage a tiny TASKS list with `harmbench` and verify it dies with the expected message when `CSCS_SERVING_API` is unset:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals
unset CSCS_SERVING_API
TASKS="harmbench" sbatch --test-only scripts/evaluate.sbatch dummy dummy
# expect: "ERROR: $CSCS_SERVING_API is unset ... harmbench"
```

Also verify a non-judge TASKS list still passes (no false positive):

```bash
TASKS="mmlu" sbatch --test-only scripts/evaluate.sbatch dummy dummy
```

---

## H3 — Add 2 posttraining pools

**Why.** Posttraining evals have three distinct populations: reference HF models (e.g. apertus-8B, llama, mistral), local-distill checkpoints from the SNR distillation work, and the 36-cell SNR pretraining sweep itself (already covered by `pretraining_a06`). Today there's no pool for the first two.

**File.** [scripts/build_configs.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/scripts/build_configs.py) — where the pool enumeration lives.

**Spec.** Add two new pools to the `POOLS` mapping:

- `posttraining_hf_reference` — the HF reference models for posttraining comparison (e.g. swiss-ai/Apertus-8B-Instruct, allenai/Olmo-3-1025-7B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Llama-3.1-8B-Instruct, etc.). The exact list comes from the user — they were asking which HF posttraining evals already ran ("we launched 3 hf postraining evals" — H4 context).
- `posttraining_local_distill` — the local-staged distilled apertus posttraining checkpoints (the `apertus-0.6b-from8b-TOP256-long` / `apertus-1b-from8b-TOP256-long` family already staged at `/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints/`).

**Action items for the new session.**
1. Ask the user which models go in each pool (the H4 discussion will surface the HF reference list).
2. Add the pool entries to `build_configs.py` POOLS.
3. Update the launcher mode-switch (`src/evals/scripts/launch_evaluations.sh`) to map the two new pool names to the correct TASKS file + runner.
4. Add a corresponding runner stub under `src/evals/runners/` if one doesn't already cover them.

**Test.** `python3 scripts/build_configs.py --pool posttraining_hf_reference --dry-run` should print the expected NAME list with their backend args; same for the distill pool.

---

## H4 — HF posttraining eval launch path — **AWAITING USER DECISION**

**Context.** The user observed "we launched successfully 3 hf postraining evals" and asked to verify the path before generalizing. There are two plausible designs; the user explicitly asked to confirm BEFORE implementing.

**Option A — Doc-only fix.**
- Don't change code. Document in `src/evals/CLAUDE.md` exactly which launcher + runner combination was used for the 3 successful HF posttraining evals, and add a "to run a new HF posttraining eval, do X" stanza.
- Pros: zero-risk, preserves whatever incantation worked.
- Cons: every new HF posttraining eval requires hand-rolling.

**Option B — Unify launchers.**
- Add a `hf_posttraining_runner.sh` modeled on `snr_pretraining_hf_top.sh`, with the chat-template + judge-API plumbing baked in. Wire `launch_evaluations.sh` to dispatch by pool.
- Pros: H3 + H5 reuse cleanly; one runner per population.
- Cons: bigger diff; needs verification that the unified runner reproduces the 3 successful runs.

**Blocker.** User asked to confirm direction. **Re-ask at the start of the next session.** Don't start either path without an answer.

**Investigation pointers for whichever option wins.**
- The 3 successful jobs are in `sacct -u $USER -S 2026-05-15 --format=JobID,JobName%50,WorkDir%80` — filter by jobnames matching `eval-*posttraining*` or scan `eval_logs/` mtimes 2026-05-15+ for posttraining model dirs.
- Cross-reference the slurm `.out` / `.err` to see which TASKS file + backend + chat-template flag they ran with.
- The runners themselves are under `src/evals/runners/` (see `snr_pretraining_hf_top.sh`, `snr_pretraining_local_hf.sh`, etc.).

---

## H5 — Force chat template for all posttraining evals

**Why.** Posttraining evals MUST go through the model's chat template; otherwise they evaluate the wrong rendering. Today the mode handler does not force it.

**Spec.** In the mode dispatch (`src/evals/scripts/launch_evaluations.sh` or the posttraining runner depending on H4 outcome), when the mode is any of `snr-posttraining*` / `posttraining_*`, force:

```bash
export CHAT_TEMPLATE_OVERRIDE="true"
```

This env should reach the inner container — verify it's in `INNER_EXPORTS` in `evaluate.sbatch` (search for `INNER_EXPORTS=` — eval bug 5 context). If not, add it.

Then in `_run_per_task.sh` (or wherever the lm-eval CLI is constructed), ensure `--apply_chat_template` gets set when `$CHAT_TEMPLATE_OVERRIDE == "true"`.

**Test.** Submit a one-task posttraining eval with `--limit 4` and confirm the lm-eval invocation has `--apply_chat_template` in it (grep the slurm `.out`).

---

## M1 — New `full_eval` / `main_eval` / `10_ckpts` / `da_ckpts` derivation

**Why.** The user wants the checkpoint subsets used for eval reporting to be derived from percentages of total training, not a hardcoded list. This makes the spec apply uniformly to models trained for different total iters (50000, 800000, etc).

**Spec (verbatim from user).**

- `full_eval` = union of `dense_tail` and `10_ckpts`
- `10_ckpts` = 10%, 20%, 30%, …, 90%, 100% of total iters
- new `main_eval` = union of `dense_tail` and `da_ckpts`
- `da_ckpts` = 5 ckpts at 10%, 33%, 50%, 66%, 100% of total iters
- `dense_tail` remains as today (last 10% with up to 5 dense picks)

**File.** [src/evals/scripts/utils/configs.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/configs.py) — specifically the `_meg_stage` helper that derives subsets per model.

**Implementation sketch.**

```python
def _pct_iters(total: int, percentages: list[float]) -> list[int]:
    """Round each percentage of `total` to the nearest 2000-step grid
    (Megatron save cadence). De-dup and sort ascending."""
    grid = 2000
    out = sorted({max(grid, round(total * p / 100 / grid) * grid) for p in percentages})
    return out

def _meg_stage(model_total_iters: int, dense_tail_iters: list[int]) -> dict[str, list[int]]:
    ten = _pct_iters(model_total_iters, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    da  = _pct_iters(model_total_iters, [10, 33, 50, 66, 100])
    full_eval = sorted(set(dense_tail_iters) | set(ten))
    main_eval = sorted(set(dense_tail_iters) | set(da))
    all_iters = sorted(set(dense_tail_iters) | set(ten) | set(da))  # for L4
    return {
        "10_ckpts":  ten,
        "da_ckpts":  da,
        "full_eval": full_eval,
        "main_eval": main_eval,
        "all":       all_iters,
    }
```

**Touches.**
- Update every consumer that reads `full_eval` / `main_eval` (grep for them across `src/evals/`).
- The runner-side `iters_for(model, subset="full_eval")` should return the derived list.
- Add unit-style assertions: for SNR-canonical 50000-iter models, `10_ckpts == [5000, 10000, 15000, ..., 50000]` if grid=5000, OR `[6000, 10000, 20000, 30000, 40000, 50000, ...]` depending on how the rounding lands. Sanity-check against the canonical iter list `2000, 6000, 12000, 18000, 22000, 28000, 34000, 38000, 42000, 44000, 46000, 48000, 50000` in `src/evals/CLAUDE.md`.
- Distill models (`apertus-1b-from8b-TOP256-long` with iters 120000–800000) should auto-compute the right percentage iters without manual lists.

**Test.** Add a quick `__main__` block that prints `_meg_stage(50000, [42000, 44000, 46000, 48000, 50000])` and `_meg_stage(800000, [720000, 740000, 760000, 780000, 800000])` and eyeball the resulting subsets.

---

## M2 — Delete dead `HF_STAGING_BASE` constant

**Why.** Both `HF_STAGING_BASE` and `HF_LOCAL_BASE` exist; one is dead weight after the recent refactor. The single live constant should be `HF_LOCAL_BASE`.

**Action.** `grep -rn HF_STAGING_BASE src/` → delete the constant + every import. If `HF_STAGING_BASE` was used as a fallback, repoint to `HF_LOCAL_BASE`.

**Test.** `grep -rn HF_STAGING_BASE src/` returns empty. Project still imports cleanly.

---

## M3 — Add `num_key_value_heads` to `models.json`

**Why.** vLLM rejects model load if `tensor_parallel_size` does not divide `num_key_value_heads` (eval bug 14 in `src/evals/CLAUDE.md`). Today the per-size TP map is hard-coded in the launcher; centralising the KV-head count in `models.json` lets the launcher compute TP per-cell from declarative data.

**File.** [configs/models.json](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/configs/models.json)

**Spec.** Add `"num_key_value_heads": <int>` to every SNR Apertus entry (and to any other model that goes through vLLM). Per the CLAUDE.md table:

| Size | num_attention_heads | num_key_value_heads |
|---|---:|---:|
| 175M | 16 | 4 |
| 350M | 20 | 5 |
| 600M | 24 | 6 |
| 1B   | 28 | 7 |

For HF reference models, read off the upstream `config.json` (e.g. `hf download swiss-ai/Apertus-8B-2509/config.json` → `num_key_value_heads`).

**Consumer change.** In whichever runner builds the `sbatch --export=...` line, replace the hard-coded TP map with a lookup that computes `TP = greatest divisor of GPUS_PER_NODE that also divides num_key_value_heads`. For SNR sizes: 175M→4, 350M→1, 600M→2, 1B→1.

**Test.** `python3 -c 'from src.evals.scripts.utils.configs import load_models; m=load_models(); print({n: e.get("num_key_value_heads") for n,e in m.items()})'` returns the expected values.

---

## M4 — `cache-datasets.sh` minimal prewarm

**Why.** Cold-start of an 86-task sweep hammers the HF datasets API and triggers 429s (eval bug 12). A one-shot script that downloads every task's dataset into the local HF cache BEFORE the sweep prevents the cascade.

**File.** Create [src/evals/scripts/cache-datasets.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/cache-datasets.sh) (rename from "prewarm" per user — "call it 'cache-datasets.sh'").

**User instruction (verbatim).** "add a one-shot prewarm-datasets.sh to download all tasks in a group into the cache before the real sweep, but call it 'cache-datasets.sh', implement it with really minimal code, as simple as possible"

**Spec — really minimal.**

```bash
#!/usr/bin/env bash
# Prewarm the HF datasets cache for every task in a group from configs/tasks.json.
# Usage: bash cache-datasets.sh <group_name>   e.g. pretraining_full, posttraining
set -euo pipefail
group=${1:?usage: $0 <group_name>}
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual
source /users/mariagrandury/miniconda3/etc/profile.d/conda.sh && conda activate snr
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/$USER/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/users/$USER/hf_models}
python3.11 -c "
import json, sys
from datasets import load_dataset
tasks = json.load(open('configs/tasks.json'))
group_list = tasks['groups']['$group']
for t in group_list:
    repo = tasks['tasks'][t].get('benchmark', t)
    print(f'[cache] {t} <- {repo}')
    try: load_dataset(repo, trust_remote_code=False)
    except Exception as e: print(f'[cache] SKIP {t}: {e}', file=sys.stderr)
"
```

Adjust the `repo` resolution if `tasks.json` doesn't store the HF repo directly under `benchmark` — that's the field name to check. If the actual HF repo lives in a different field, use that.

**Test.** `bash cache-datasets.sh posttraining_llm_judge` should populate the cache for the 7 judge tasks; re-running it should be near-instant (datasets cached).

---

## L2 — `convert-snr.sh` exit 1 on empty plan

**File.** [src/pretrain/conversion/convert-snr.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/conversion/convert-snr.sh)

**Why.** When the generated plan file has 0 lines (no cells / iters matched), the script silently continues and submits zero work. Should fail-fast.

**Spec.** After the plan file is fully written (end of mode 2 or mode 3 — wherever `wc -l "$plan"` becomes computable), add:

```bash
line_count=$(grep -cv '^#' "$plan" || true)
if (( line_count == 0 )); then
    echo "[convert-snr] ERROR: plan file $plan is empty after filtering — nothing to convert." >&2
    exit 1
fi
```

**Test.** `bash convert-snr.sh --models nonexistent-model --dry-run` should exit 1 with the error.

---

## L4 — Assert `full_eval`, `main_eval` ⊆ `all`

**File.** [src/evals/scripts/utils/configs.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/configs.py)

**Spec.** Inside `_meg_stage` (after M1 lands), append:

```python
assert set(full_eval).issubset(set(all_iters)), \
    f"full_eval not subset of all: {set(full_eval) - set(all_iters)}"
assert set(main_eval).issubset(set(all_iters)), \
    f"main_eval not subset of all: {set(main_eval) - set(all_iters)}"
```

**Why.** Defensive sanity: if M1's `all` definition drifts, the assert catches it at import time rather than silently dropping iters at eval time.

---

## L5 — Log unrecognized NAMEs to `models_dropped.log`

**File.** [src/evals/scripts/push_all_results.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py)

**Why.** `parse_name` returns None when a NAME doesn't match any model in `models.json`. Today those names are silently skipped — invisible to the user. Log to `models_dropped.log` so the user can see what got dropped.

**Spec.** Wherever `parse_name(name)` returns None inside the bulk-rescue walk, add:

```python
with open("models_dropped.log", "a") as f:
    f.write(f"{name}\n")
```

Open the file once (append-mode) at the top of the bulk walk and write per drop; print a single summary line at the end:

```python
print(f"[push] dropped {n_dropped} NAMEs not in models.json — see models_dropped.log")
```

**Test.** Run bulk push with at least one stray NAME; verify the file exists and lists it.

---

## L6 — `convert-snr.sh` mode 3 → use `iters_for()`

**File.** [src/pretrain/conversion/convert-snr.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/conversion/convert-snr.sh)

**Why.** Mode 3 currently uses two legacy hardcoded lists `ITERS_SEED1904` / `ITERS_OTHER`. These duplicate logic that already lives in `iters_for()` in `src/evals/scripts/utils/configs.py`. Single source of truth wins.

**Spec.** Replace the `if [[ "$cell" == *seed1904* ]]; then iters=ITERS_SEED1904; else iters=ITERS_OTHER; fi` block with a Python inline that calls `iters_for(model)`:

```bash
# Pull the iter list from configs.iters_for (single source of truth)
iters=( $(python3.11 -c "
import sys; sys.path.insert(0, '$REPO_ROOT/src/evals/scripts/utils'); from configs import iters_for
print(' '.join(str(i) for i in iters_for('$cell')))
") )
```

Then delete the `ITERS_SEED1904` and `ITERS_OTHER` array definitions at the top of the script.

**Test.** `bash convert-snr.sh --dry-run` mode 3 should produce the same enumeration as before for non-1904 cells (9 iters) and seed1904 cells.

---

## Discovered during review — seed-1797 / seed-28 marker drift + empty-shell `iter_*/` dirs

**State on disk (as of 2026-05-17).**

- Across the 24 seed-1797 / seed-28 cells, **all late iters (40000–50000) are valid** torch_dist checkpoints (`.metadata + N × .distcp`).
- **iter_0036000** is a 0-file empty shell in **10 of 12 seed-1797 cells** (everything except 175M-fwEdu30 and 175M-fwEdu60).
- Recurring empty-shell iters across seed-1797 cells: `iter_0024000`, `iter_0026000`, `iter_0032000`, `iter_0036000`, `iter_0040000` + handful of off-grid (`iter_0022270`, `iter_0034969`, etc.).
- **seed-28** is mostly clean — only the 3× 175M cells have shell dirs.
- All shells were created in a tight window on **2026-05-16 18:44–20:20** (yesterday). Nothing touched them today.
- **No `apertus-*-edu*` SLURM training jobs ran in the past 7 days under your UID.** Whatever created the empty dirs was not a Megatron training resume.
- Several cells have `latest_checkpointed_iteration.txt` pointing at an empty-shell iter (e.g. `175M-fwEdu30-seed1797` marker = 22000 but `iter_0022270/` is a shell; `350M-fwEdu60-seed1797` marker = 32000 but `iter_0032033/` is a shell). A resume from those would hit `FileNotFoundError` on the first distcp.

**Suspected cause.** The empty-shell pattern matches `src/pretrain/CLAUDE.md` failure mode #2 (Megatron `--async-save` killed between metadata-write and shard-write). With no training jobs in the window, the most plausible non-SLURM trigger is something on the conversion path — `convert-snr.sh` or its inner Megatron load. The previous session's bash history shows multiple `launch_resumes.sh` and `convert-snr.sh --submit` invocations on 2026-05-16, but bash_history has no timestamps to pin which fired in the 18:44–20:20 window.

**Not data loss for late iters.** Eval at the canonical late anchors (44000, 46000, 48000, 50000) will not regress in any cell.

**Two independent fixes.**

1. **Make `pretrain_progress.py` report effective `latest_valid_iter`.** Today it reads `latest_checkpointed_iteration.txt`. Change it to scan all `iter_*/` dirs and return `max(valid)` per cell. The plot's "red/orange/green" lens then reflects on-disk validity instead of marker truth. This is low-risk: the marker is read-only there; we'd just override the reported number.

2. **Make `launch_resumes.sh` auto-rewind a marker that points at a shell.** Today `rewind_marker` only fires when the launcher already knows the desired `load_iter` is below the marker. Extend it to also handle the case where the marker itself is invalid (empty shell). In that branch, scan for the highest valid iter below the marker and rewind to it before submitting a resume. Behaviour stays inside the "never rm" rule — the empty shell dirs are left in place and overwritten when training next passes that step.

**Both fixes are conservative — neither deletes anything on disk.** Do them together: (1) for visibility, (2) for fix-forward. Run `bash launch_resumes.sh --dry-run` after (2) and verify the rewind-to-valid plan looks right before submitting.

---

## How to drive this in the next session

1. **Start with C3.** Self-contained, tested example above, no user input needed. Commit.
2. **Then H5.** Small. Commit.
3. **Ask user for H4 direction** (Option A vs Option B). Park H3 until H4 is decided — they're related.
4. **M1 is the biggest item.** It touches the per-stage subset definitions and every consumer. Plan it carefully: write `_meg_stage` first, add asserts (L4), then update consumers one by one. Verify against existing snapshots before committing.
5. **M2, M3, M4** are independent — do in parallel-ish.
6. **L2, L5, L6** are mechanical. Batch into one commit at the end if small.
7. **Marker drift** can be tackled anytime — it doesn't block the refactor work.

**Per-fix discipline.** Each fix gets its own commit with a descriptive message (the C1 commit message is a good template). Test before committing. Don't batch.
