# Predictivity sweep — Azure compute budget

Status 2026-08-14: **quota requests filed** (Spain Central NCadsH100v5, UK South
NDSH100v5, dedicated + Spot/low-priority counters).

> **Superseded on the ND region (2026-08-26).** The ND pool moved **UK South →
> Canada Central**: no H100 SKU on this subscription can use low-priority at
> all, so the UK Spot plan below is unreachable, and Canada Central is the
> cheapest allow-list-clear region (see `src/pretrain/azure/README.md`, "The
> four gates"). The prices, quota tables and terminal evidence below are kept
> as the record of what was measured on 2026-08-13/14 — read them as history,
> not as the current plan.

This sheet is the budget of
record for the small-to-large predictivity training plan
([small-to-large-predictivity-training-plan.md](small-to-large-predictivity-training-plan.md)),
computed from the 90M–1.7B deep ladder in `src/pretrain/hyperparams/hyperparams_deep.json`
(the reviewed source of truth since 2026-08-14; see Appendix B).
A rendered version lives at the "Predictivity Sweep Compute Budget" artifact
(claude.ai/code/artifact/3d1f4011-c824-4899-b045-a5dc1d66bb17).

> **Stale by ~10% (2026-08-28):** the Azure tables below price the earlier
> **51-run** grid; the grid is now **56 runs** per level (L=2 gained ×3 seeds
> at 175M/600M and the 1.7B row gained L=2 — see the cluster table further
> down, which already uses 56). Scaling the per-size rows: ≈ 2.32e22 FLOPs
> per level, fast mix ≈ **$130k** (headroom ≈ $70k), all-Spot ≈ $137k.
> A full re-derivation is on the todo list (low priority).

## Verdict

| | |
|---|---|
| Total compute | **5.95e22 FLOPs** · 153 runs (51/level × 3 intervention levels) · **1,722 H100-days** |
| Recommended plan | ≤600M on Spain Central low-priority + 1B/1.7B on ND96isr (UK South Spot as costed here; **now Canada Central dedicated** — see the note above) → **≈ $110k** |
| Grant | $200k → **≈ $90k headroom** (evictions, storage, dedicated tail, retries) |
| Deadline | Aug 31 (hard stop Sep 4) → needs **~141 concurrent H100s** for ~14 training days |

## Real prices (Azure Retail Prices API, 2026-08-13, USD, Linux)

Only the SKUs this subscription can actually deploy (see Appendix A.2/A.4).
"Low-pri" is the fixed-price meter Azure ML bills for `tier: low_priority`
clusters; Spot is the variable market meter. Discounts vs dedicated: 77–81%.

| Option | PAYG $/GPU-h | Spot | Low-pri | $/1e21 FLOPs (low-pri · spot) | Verdict |
|---|---|---|---|---|---|
| NC80adis H100 · Spain Central | 9.07 | 2.80 | **1.82** | **1,260** · 1,944 | cheapest $/FLOP — economy pool |
| ND96isr 8×H100 · UK South | 15.36 | **2.84** | 3.07 | 2,134 · **1,972** | the fast nodes — 1B/1.7B pool |
| NC80adis H100 · Switzerland N | 19.96 | 2.71 | 2.00 | 1,386 · 1,881 | alternate economy |
| NC96ads 4×A100 · Sweden C | 4.78 | 1.83 | 0.96 | 2,122 · 4,068 | A100 fallback (small runs) |
| ND96isr H200 · Poland C | 16.32 | 16.32 | — | — | Spot = PAYG, no discount — skip |
| ND96isr H100 · Norway E / Italy N | — | — | — | — | no meters for this subscription — skip |

Availability constraints that drove this (Appendix A.2/A.4): Sweden Central is
`NotAvailableForSubscription` for **every** H100 SKU; Spain Central and
Switzerland North are the only usable NC H100 regions in Europe; UK South is
the only usable ND H100 region with real meters; France's MI300X is unusable
(the swiss-ai Megatron/xIELU stack is CUDA-only).

## The workload (implemented ladder, trimmed grid)

Grid: L ∈ {1, 2, 8, 15, 30, 50, 100}, ×3 seeds (64/313/1904) at 175M and 600M in
the L ∈ {1, 30, 100} rows, 1.7B at L ∈ {1, 8, 30, 100} — 51 runs per
intervention level. FLOPs = 6 · N_eff · D with N_eff = non-embedding + d·V
output projection (V = 131,072 — it nearly doubles the smallest sizes'
effective compute); throughput 40% MFU (±15%).

| Size | Tokens (5×C) | FLOPs/run | wall 2×H100 | wall 8×H100 | $/run low-pri | $/run Spot | Runs/level | Level $ LP · Spot |
|---|---|---|---|---|---|---|---|---|
| 90M | 9.3B | 1.08e19 | 3.7 h | 0.9 h | 14 | 21 | 7 | 95 · 149 |
| 175M | 17.6B | 3.28e19 | 11.4 h | 2.8 h | 41 | 65 | 13 | 537 · 841 |
| 350M | 34.4B | 1.06e20 | 1.5 d | 9.2 h | 133 | 208 | 7 | 932 · 1,459 |
| 600M | 59.5B | 2.84e20 | 4.1 d | 24.6 h | 358 | 560 | 7 | 2,504 · 3,919 |
| 1B | 94.4B | 6.68e20 | 9.7 d | 2.4 d | 842 | 1,317 | 13 | 10,940 · 17,121 |
| 1.7B | 167.2B | 1.98e21 | 28.7 d | 7.2 d | 2,496 | 3,905 | 4 | 9,982 · 15,622 |
| **Σ / level** | | **1.98e22** | | | | | **51** | **24,990 · 39,112**† |

† level totals under a single meter; the recommended plan mixes them (below).
The 1B + 1.7B rows are **84%** of all compute.

## Scenarios (× 3 intervention levels)

| Scenario | Total | vs $200k grant |
|---|---|---|
| Hybrid: ≤600M Azure (Spain LP), big rungs at CSCS | $12k | ✓✓ |
| All on Spain low-priority (1.7B slow: 28.7 d/run) | $75k | ✓✓ |
| **★ Fast mix: ≤600M Spain LP, 1B/1.7B UK ND Spot** | **$110k** | ✓ ~$90k headroom |
| All UK ND Spot (everything at 8×H100 speed) | $117k | ✓ |
| Dedicated economy (all Spain PAYG) | $375k | ✗ |
| Dedicated fast mix (big rungs UK PAYG) | $592k | ✗ (fallback pricing only) |

## Reduced target and what to ask Azure for (2026-08-28)

The grid the sponsor ask is built on is now **21 × 1B + 7 × 1.7B** (28 runs),
not the full 45/15. Azure jobs are single-node (`torchrun --standalone`), so
nodes buy **concurrency, not per-run speed** — which makes GPUs-per-node the
binding constraint, and makes a single run's length a hard floor on the
schedule no matter how many nodes are granted.

Per run at 40 % MFU: **1B** = 2.4 d on 8×H100 / 9.8 d on 2×H100;
**1.7B** = 7.2 d / 29.0 d.

`ND96isr_H100_v5` (8×H100, 96 cores/node), assuming a 29 Aug start — every
date shifts 1:1 with the grant date:

| Nodes | Cores | 1B only | 1B + 1.7B |
|---|---|---|---|
| 1 | 96 | 19 Oct | 9 Dec |
| 4 | 384 | 13 Sep | 25 Sep |
| **6** | **576** | **10 Sep** | 16 Sep |
| **9** | **864** | 3 Sep | **12 Sep** |
| 11 | 1056 | — | 10 Sep |

`NC80adis_H100_v5` (2×H100) cannot carry the 1.7B rung **at any node count**:
one run is 29 d and cannot span nodes, so the earliest finish is ~27 Sep with
21 nodes or 28 or 100. It *can* carry the 1B rung — ~21 nodes (1,680 cores)
reaches 8 Sep — which makes it a usable fallback for that half only.

**The ask, in order of what unblocks most per core granted:** 6 ND nodes for
the 1B rung by 10 Sep, 9 for both rungs by 12 Sep; a single node still starts
the critical path. Two non-capacity blockers are worth raising separately
because they are not gated on the global GPU shortage: the SKU allow-list
(`NotAvailableForSubscription`, an enablement decision) and the low-priority
tier being disabled for H100.

Two prerequisites on our side, not Azure's: the ~2.5 TB tokenized dataset is
**not yet uploaded** to either workspace, and the binding fix in the job specs
is still awaiting a passing smoke run.

## A100 fallback — how long the big rungs would take (2026-08-26)

Added because H100 quota has not landed: after 13 days and 20 tickets the
subscription still holds **0 H100/H200 cores** (see the `quota_status.sh board`
report). A100 is the one GPU class this subscription is *not* allow-list
blocked from — `NC24/48/96ads_A100_v4` in 15 Azure ML regions, and
`ND96amsr_A100_v4` (8×A100 80 GB **+ InfiniBand**, the only A100 SKU worth
using for these rungs) in **canadaeast** and **swedencentral**.

Same convention as the tables above: FLOPs/run unchanged, 40 % MFU, bf16.
A100 SXM peak is 312 TFLOP/s vs H100's 989 → **A100 is 3.17× slower per GPU**.

| Rung | FLOPs/run | A100-days/run | 1 node (8×A100) | 4×A100 (NC96ads) | _vs 8×H100_ |
|---|---|---|---|---|---|
| 1B | 6.68e20 | 62.0 | **7.7 d** | 15.5 d | _2.44 d_ |
| 1.7B | 1.98e21 | 183.6 | **23.0 d** | 45.9 d | _7.24 d_ |

Whole-sweep totals (21 × 1B and 15 × 1.7B — 7 and 5 per level × 3 levels,
per `launch_trainings.py`; the ×3 seeds sit at 175M/600M, not 1B):

| Set | Runs | A100-days |
|---|---|---|
| all 1B | 21 | **1,302** |
| all 1.7B | 15 | **2,754** |
| all 1B + 1.7B | 36 | **4,056** |

Wall-clock against fleet size (runs are independent jobs, so they parallelize
freely; ignores eviction overhead):

| Fleet | all 1B | all 1B + 1.7B |
|---|---|---|
| 3 × ND96amsr (24 A100) — _today's low-priority allowance_ | 54 d | **169 d** |
| 4 × ND96amsr (32 A100) | 41 d | 127 d |
| 8 × ND96amsr (64 A100) | 20 d | 63 d |
| 16 × ND96amsr (128 A100) | 10 d | 32 d |
| 32 × ND96amsr (256 A100) | 5 d | 16 d |

**A100 is a schedule fallback, not a cost saving.** At canadaeast low-priority
($7.865/node-h = $0.98/GPU-h) the 1B+1.7B set alone is **$131k** (Spot:
$189k) — ≈ $2,190 per 1e21 FLOPs, ~11 % dearer than UK ND H100 Spot ($1,972)
and 3.2× slower. It only makes sense if H100 quota never lands.

The binding constraint is fleet size, not price: the low-priority allowance
that already exists (300 vCPU regional = 3 ND nodes, see below) gives **231
days** for 1B+1.7B. A100 only becomes viable with a real dedicated grant
(≥ 8 nodes), and at 16 nodes it roughly matches the original H100 schedule.

## CORRECTION (2026-08-26, later the same day): low-priority is impossible for H100

The section below was written from the quota counters and is **wrong in its
conclusion**. Tested empirically by creating the clusters: AML rejects every
H100 SKU for low-priority outright.

```
UnsupportedVMSizeForLowPriority: The VM size STANDARD_ND96ISR_H100_v5 is not
allowed for LowPriority. Please convert to Dedicated or use a different VM size.
```

Confirmed for `Standard_ND96isr_H100_v5` **and** `Standard_NC80adis_H100_v5`.
So the 300 granted `TotalLowPriorityCores` are unspendable on H100, and
**dedicated quota is the only path** — the 23 pending dedicated tickets are
the correct ask after all.

Three independent signals pointed the wrong way and should not be trusted
again: the quota counters (`lowPriority` limit `-1`, `TotalLowPriorityCores`
300), `az vm list-skus` (`LowPriorityCapable=True` for ND96isr_H100_v5), and
the retail price list (a "Low Priority" meter exists for it, $23.60/node-h).
The cluster also creates as `provisioning_state: Succeeded` regardless; the
rejection appears only in `properties.errors` and reaches jobs as _"cluster
has encountered unknown issue"_.

**The cheap probe** — costs $0, since `min_instances: 0` allocates nothing —
should be run against any new SKU/region/tier before planning around it:

```bash
az ml compute create --file <compute>.yml $AZ_ML_ARGS
az rest --method get --url ".../workspaces/$AZ_WS/computes/<name>?api-version=2024-10-01" \
  --query "properties.properties.errors"
```

### A100 low-priority: allowed, but out of capacity (probed 2026-08-26)

Probed all four combinations in the Canada Central workspace. The two failure
modes are completely different and must not be conflated:

| SKU | tier accepted? | node allocates? |
|---|---|---|
| `ND96isr_H100_v5` | ✗ `UnsupportedVMSizeForLowPriority` | never — config rejected |
| `NC80adis_H100_v5` | ✗ `UnsupportedVMSizeForLowPriority` | never — config rejected |
| `NC24ads_A100_v4` (1×A100) | ✓ no error | ✗ `OutOfCapacity` |
| `NC96ads_A100_v4` (4×A100) | ✓ no error | ✗ `OutOfCapacity` |

So the policy is **family-level: A100 may use low-priority, H100 may not.**
The H100 rejection is permanent and no ticket changes it. The A100
`OutOfCapacity` is *transient* — worth re-probing periodically and in other
regions, since it needs no quota at all. `OutOfCapacity` also explicitly
suggests dedicated VMs "to improve chances of capacity allocations".

`LowPriorityCapable` from `az vm list-skus` is wrong in **both** directions —
`True` for the rejected ND96isr_H100_v5, `False` for the accepted
NC96ads_A100_v4. Ignore it entirely; probe instead.

Untested, and the one worth testing next: `ND96amsr_A100_v4` (8×A100 + IB,
the only A100 shape worth using for the 1B/1.7B rungs) in **canadaeast** or
**swedencentral**. It needs a workspace in one of those regions.

### Dedicated vs low-priority fail at opposite ends (2026-08-26)

A *dedicated* cluster is quota-checked at **create** time and refused
outright, even with `min_instances: 0`:

```
ClusterMinNodesExceedCoreQuota: The specified subscription has a Standard
NCADSA100v4 family vCPU quota of 0 and cannot accomodate for at least 1
requested managed compute nodes which maps to 96 vCPUs.
```

A *low-priority* cluster is not, and creates happily whatever the quota.
So "the cluster created" means nothing for low-priority and everything for
dedicated — do not generalise either way.

### Cost per 1B run on A100 (D = 94.4B tokens, 6.68e20 FLOPs, 40 % MFU)

AmlCompute exposes only `dedicated` and `low_priority`; the Spot meter is
unreachable, which is what makes dedicated H100 so expensive here.

| Cluster | GPUs/node | wall-clock | $/h | **$/run** |
|---|---|---|---|---|
| `ND96amsr_A100_v4` low-pri (canadaeast) | 8 + IB | 7.7 d | 7.87 | **$1,461** |
| `NC96ads_A100_v4` low-pri (canadacentral) | 4 | 15.5 d | 8.82 | **$3,278** |
| `NC96ads_A100_v4` dedicated | 4 | 15.5 d | 17.63 | $6,553 |
| `ND96isr_H100_v5` dedicated (canadacentral) | 8 + IB | 2.4 d | 122.40 | $7,172 |

**A100 low-priority is ~5× cheaper per run than dedicated H100** — the only
H100 tier reachable through AML. If low-priority capacity ever appears,
canadaeast ND96amsr is the cheapest way to train these rungs on Azure, at
3.2× the wall-clock of H100.

## Low-priority / Spot was never actually requested (2026-08-26) — superseded, see above

All **23 quota-change payloads across the 20 open tickets are `Type:
Dedicated`** — verified from each ticket's `quotaTicketDetails`. No Spot or
low-priority request has ever been filed, so the "Spot/LP vCPUs" column in
_Quota requested_ below never reached Azure. This matters because the
low-priority counters tell a different story from the dedicated ones:

| Counter (any candidate region) | Value |
|---|---|
| `standardNDv5H100Family` dedicated | 0 |
| `standardNDv5H100Family` **low-priority** | **-1 (no per-family cap)** |
| `TotalLowPriorityCores` (regional) | **0 / 300 — already granted** |

300 regional low-priority vCPUs = **3 × ND96isr = 24 H100**, apparently
available now in any region where the SKU is not allow-list blocked
(italynorth, norwayeast, uksouth, canadacentral). Quota is not capacity,
so this must be confirmed by actually creating a low-priority cluster —
but it needs no new ticket. Note `az vm list-skus` reports
`LowPriorityCapable=False` for the NC H100 SKUs (Spain/Switzerland economy
pool) and `True` for every ND SKU, so the low-priority path runs through the
**ND nodes, not the NC ones** the economy plan above is built on.

## Deadline schedule (finish Aug 31, hard stop Sep 4)

1,722 H100-days of work; runs are independent AML jobs that parallelize freely
across cluster nodes (jobs queue when all nodes are busy). ~15% eviction
overhead assumed on Spot/low-priority.

| Dates | What happens |
|---|---|
| Aug 13–14 | Quota tickets filed ✔. Build the datasets on CSCS with `build_data_mixtures.py` (sources live there), upload to both workspaces (`azcopy`, ~2.5 TB). |
| Aug 15–17 | Quota lands; `setup_azure.sh` per workspace; smoke test; verify data mounts. |
| ~Aug 17 → Aug 30–Sep 1 | **~14 days of training on ~140 concurrent H100s**: 16× ND96isr Spot (UK; all 1B/1.7B — longest run 7.2 d) + 6× NC80adis low-pri (Spain; everything ≤600M). |
| Sep 1–4 | Slack for eviction overruns; thrashing runs move to dedicated nodes (headroom covers it). |

Contingency: if UK grants only ~8 ND nodes, the big rungs take ~2× longer
(→ ~Sep 8–12 for all three levels); then finish 2 levels by Aug 31 and either
relax the date for level 3 or temporarily rent dedicated ND capacity
(still inside the grant). If Spot capacity is thin, fall back to the UK
low-priority meter (fixed price, ~8% dearer) or briefly to dedicated.

Fleet ↔ calendar (per level ≈ 574 H100-days): 12 d needs 165 H100 · 14 d → 141
· 16 d → 124 · 18 d → 110.

## Quota requested (2026-08-14)

| Region | Family | Dedicated vCPUs | Spot/LP vCPUs | Buys |
|---|---|---|---|---|
| UK South | Standard NDSH100v5 | 96–192 | 1,536 | 16 Spot ND96isr (128 H100) + dedicated fallback |
| Spain Central | Standard NCadsH100v5 | 160 | 480 | 6 low-pri NC80adis (12 H100) + 2 dedicated |
| Sweden Central (optional) | Standard NCADSA100v4 | — | 192 | A100 low-pri overflow |

The families were absent from the subscription's usage counters in every
region (Appendix A.5/A.6), so absent-family requests go through Help + Support
→ "Service and subscription limits (quotas)". The subscription's offer is
**EnterpriseAgreement_2014-09-01** (Appendix A.7) — Spot and low-priority are
supported. Azure ML's separate low-priority quota may have a non-zero default:
check Studio → Quota once a workspace exists.

## Method & assumptions

- Budgets: D(N) = 100 × N_non-emb per size (5× Chinchilla), stored in
  `hyperparams_deep.json`'s per-size `predictivity` block (formula in the
  file's `predictivity_schedule` note).
- FLOPs: 6 · (N_non-emb + d·V) · D. Attention overhead folded into the MFU
  band (40% ± 15% relative).
- Effective throughput: H100 0.40 PFLOP/s, A100 0.125 PFLOP/s.
- Prices: exact meters from the Retail Prices API (Appendix A.9); Spot meters
  drift with the market, low-priority meters are fixed.
- Extras on top of any scenario: blob storage ~2.4 TB (~$100/mo), data-prep
  CPU time on CSCS (free allocation), evals/conversions ≈ 1–3% of training.

---

# CSCS training time (Clariden GH200)

Wall-clock to train the predictivity grid on CSCS — the cluster the sweep
actually runs on — distinct from the Azure budget above. Nodes are 4× GH200
each; global batch 504 × 4096. **Time/run is one complete model** (one
size × language-setting × seed, scratch → full D = 100·N budget).

| Size  | Tokens (D=100·N) | Iters  | Nodes | ms/iter deep · shallow | Time/run | 12h segments | Runs | Node-hours |
| ----- | ---------------: | -----: | ----: | ---------------------: | -------: | :----------: | ---: | ---------: |
| 90M   |            9.3 B |  4,500 |     3 |    **1,248 · 1,154**   |   ~1.6 h |      1       |    7 |         33 |
| 175M  |           17.6 B |  8,540 |     6 |      **844 · 810**     |   ~2.0 h |      1       |   15 |        180 |
| 350M  |           34.4 B | 16,660 |    14 |      **604 · 567**     |   ~2.8 h |      1       |    7 |        274 |
| 600M  |           59.5 B | 28,800 |    21 |      **548 · 539**     |   ~4.4 h |      1       |   15 |      1,381 |
| 1B    |           94.4 B | 45,720 |    21 |                  715\* |   ~9.1 h |      1       |    7 |      1,336 |
| 1.7B  |          167.2 B | 81,000 |    21 |                1,200\* |  ~27.0 h |      3       |    5 |      2,835 |
| **Σ** |                  |        |       |                        |          |              |   56 |    ~6,040  |

- **Runs execute in parallel** (each cell on its own node allocation), so the
  serial run-hours are not calendar time — **node-hours** is the compute cost.
  1B + 1.7B are ~69% of it (the ×3 seeds sit at 175M and 600M, not 1B).
- **90M–600M are measured on this sweep**, in bold: medians over **500 k+
  logged iterations** from the 2026-08-21…27 runs, i.e. entirely after the
  training data moved to iopsstor. Sampled mid-run (first 20 and last 10
  iterations dropped, so neither cold start nor the end-of-run
  async-checkpoint flush is included). p10…p90 sits within **1–3%** of the
  median at every one of those rungs.
- **Iteration cost does not depend on L.** Across all seven language settings
  at a fixed size the medians vary by ≤4% (350M deep: 589–614 ms). Tokens per
  iteration and sequence length are identical at every L — only the *content*
  of the batch differs. Training time is therefore a function of (size, arch)
  alone; the language setting costs **eval** time, not training time.
- **\* 1B and 1.7B are NOT measured on this sweep** and are left at the older
  36-sweep figure (1B) and extrapolation (1.7B) rather than given a fabricated
  precision. They are ~90% of the node-hours, so this is the single largest
  budget uncertainty — one calibration run at each would close it.
- Measure ms/iter as a **median with a tight p10/p90 band**. A wide spread means
  the run was I/O-bound, not compute-bound, and the number is not a cost
  estimate — that is how the capstor dataloader stall hid for a day
  (`src/pretrain/CLAUDE.md` #8). Training must read from iopsstor.
- deep and shallow cost **nearly** the same per iteration at equal size, with
  shallow consistently 2–8% faster (90M 1,154 vs 1,248; 600M 539 vs 548) — as
  expected once both ladders pin ffw = 4 and gqa = 4.
- `launch_trainings.py::ITER_MS` previously held values fitted during the
  capstor period, inflated by dataloader stalls: deep 175M read 2,176 against
  844 now, shallow 90M 4,072 against 1,154. The power-of-2 GEMM aliasing
  hypothesis attached to them was refuted by a standalone benchmark (flat
  553–633 TFLOP/s at every ladder shape).
- 1.7B exceeds the 12 h queue wall → ~3 resume segments/run, handled by the
  standard `--use-checkpoint-opt_param-scheduler` path (there is no separate
  resume script: re-running `launch_trainings.py` is the resume).
- Steady-state compute only; excludes cold-start, save-iter overhead, and queue
  wait. 56 runs: the grid gained the 1.7B@L2 cell and ×3 seeds at L2 on the
  175M and 600M columns.

## Checkpointing, conversion and eval cost

Each run writes **20 checkpoints — 40 at the 1B and 60 at the 1.7B rung**
(2026-08-23; the reference rungs get denser sampling, and 40/60 are multiples
of 20 so every size stays on the shared *k*/20 grid). Checkpoint *k* is at
*k*/*n* of training at every size, and the 1×C operating point is always
checkpoint *n*/5 — 4, 8 or 12.

| Stage | Volume | Unit cost | Node-hours |
| ----- | -----: | --------: | ---------: |
| Convert (Megatron → HF, every checkpoint) | 1,460 ckpts | ~3 min | ~73 |
| Eval (every 2nd checkpoint + final, `auto` group; the planned +1 FLOPs milestone per run is **not implemented yet**) | 730 due today (786 with milestones), **610 submittable** | 30–720 min requested | **~2,200–2,550** |

The densification costs **+20%** on the current 56-run grid (1,220 → 1,460
checkpoints, 610 → 730 every-2nd evals) and lands entirely on 1B/1.7B, the
two most expensive rungs; the FLOPs milestones would add one more eval per
run on top once implemented (see the training plan's "The compute axis" —
no `milestone_iters` helper exists yet). Recomputed 2026-09-02
directly from `auto_evals_cscs.eval_minutes()` over the deep grid — 90M–600M
are measured, so the range is only about the two unmeasured rungs: the low
end evaluates 1B/1.7B at 600M's measured 0.85 min/task, the high end at the
conservative 2.0/2.8. Both include `SAFETY`, the 15-min overhead and the
15-min rounding, so they are what the watcher *requests*, not what it burns.

**120 of the 730 due jobs (16%) are refused at submission** — they exceed
the 11:59 queue cap (1B at L30/L50/L100 and 1.7B at L30/L100 — there is no
1.7B×L50 cell) and are excluded from the
node-hours above. They need `NUM_SPLITS`/`SPLIT_INDEX` before they can run
at all, so that column is a backlog, not a saving. The 600M re-estimate
(1.6 → 0.85 min/task measured) took 600M back under the cap.

Eval is **~26–29% of a level's compute** (2,200–2,550 against the ~6,290
training node-hours above), not the ~2% originally stated here and not the
~12–19% that held before the checkpoint grid was densified. The old ~142
node-hours applied L2's ~14 min to every cell, but cost scales with the task
count and the high-L cells dominate: the `auto` group expands to one task per
benchmark per language the cell trains on, from **15 tasks at L1 to 463 at
L100** (2026-08-21 wiring; it was 9 to 290 before). The range above is the
honest spread — its low end assumes 600M+ evaluate like 350M, its high end
uses the conservative per-task numbers `eval_walltime()` actually requests for
them.

Elapsed time is very close to linear in the task count. Fitted on 69 completed
eval jobs (2026-08-21…27), median elapsed per (size, n_tasks):

| Size | 9 tasks | 18 tasks | 60 tasks | overhead | per task |
| ---- | ------: | -------: | -------: | -------: | -------: |
| 90M  | 7.9 min | 13.9 min |     —    | 1.95 min | 0.667 min |
| 175M | 8.3 min | 13.9 min |     —    | 2.70 min | 0.622 min |
| 350M | 9.1 min | 15.3 min | 47.4 min | 2.34 min | 0.751 min |

The 350M fit, taken on 9 → 60 tasks, predicts **47.4 min at 60 tasks against
47.4 measured**, so extending it across language settings is sound. Extending
it across *sizes* is not: 600M and above have no eval run yet and keep
conservative estimates in `MIN_PER_TASK`.

- `eval_walltime()` previously assumed a fixed **60 min** overhead against the
  ~2.5 min measured, and put 90M at 0.6 min/task — *below* the 0.667 measured.
  Both are now fitted; overhead is 15 min for cold-start headroom.
- At the unmeasured sizes the L100 request exceeds the 11:59 queue cap
  outright. A walltime kill writes **nothing** under `BATCH_TASKS=1`, so a
  clamped request would be resubmitted and killed forever; `submit_eval` now
  refuses those instead (600M/1B/1.7B at L100 and 1.7B at L50). They need the
  eval split across jobs — `evaluate.sbatch` already has
  `NUM_SPLITS`/`SPLIT_INDEX` — before they can run at all.
- Eval is also **latency-bound, not throughput-bound**: jobs are 1 node, and
  `scripts/debug_drain.sh` moves pending work off the busy `normal` queue into
  `debug` (1:30 cap, 1 running + 1 queued). Only *conversions* and *BPB* jobs
  are moved regardless of length, because both resume per checkpoint; an eval
  is moved only if it already fits 1:30, i.e. L1/L2/L8.
- Conversion gates evaluation — a cell cannot be evaluated until its HF
  snapshot exists — so it is on the critical path despite being cheap.

---

# Appendix A — verbatim terminal evidence (María's machine, 2026-08-13)

## A.1 Azure ML quota families visible in Switzerland North (portal screenshots, transcribed)

Standard ESv3 · EDv5 · EDv4 · EDSv5 · EDSv4 · EBDSv5 · EASv4 · EADSv5 · DDv5 ·
DDv4 · DDSv5 · DAv4 · DADSv5 · Av2 · FSv2 · Ev3 · Dv3 · Dv2 · DSv3 · DSv2 ·
DDSv4 · D · EAv4 · DASv4 — "Family Cluster Dedicated vCPUs" each; **all
CPU-only, no NC/ND/NV family present**. Stated quotas: ESv3 1000 cores;
EDv5…Av2 350; FSv2…D 300; EAv4 and DASv4 175.

## A.2 `az vm list-skus --size Standard_NC --all -o table` (A100/H100 rows; T4 / legacy v3 / RTXPRO rows omitted — none bear on the analysis)

```
ResourceType     Locations           Name                        Zones    Restrictions
---------------  ------------------  --------------------------  -------  --------------------------------------------
virtualMachines  australiaeast       Standard_NC24ads_A100_v4    2        NotAvailableForSubscription, type: Location
virtualMachines  brazilsouth         Standard_NC24ads_A100_v4    1        None
virtualMachines  CanadaCentral       Standard_NC24ads_A100_v4    3        None
virtualMachines  CentralIndia        Standard_NC24ads_A100_v4    3        NotAvailableForSubscription, type: Location
virtualMachines  centralus           Standard_NC24ads_A100_v4    1,3      None
virtualMachines  eastus              Standard_NC24ads_A100_v4    2        ['NotAvailableForSubscription: Location+Zones']
virtualMachines  eastus2             Standard_NC24ads_A100_v4    1,3      None
virtualMachines  FranceCentral       Standard_NC24ads_A100_v4    1,3      None
virtualMachines  GermanyWestCentral  Standard_NC24ads_A100_v4    2,3      None
virtualMachines  ItalyNorth          Standard_NC24ads_A100_v4    1,2      None
virtualMachines  japaneast           Standard_NC24ads_A100_v4    1        None
virtualMachines  KoreaCentral        Standard_NC24ads_A100_v4    2        NotAvailableForSubscription, type: Location
virtualMachines  northeurope         Standard_NC24ads_A100_v4    1,2      NotAvailableForSubscription, type: Location
virtualMachines  PolandCentral       Standard_NC24ads_A100_v4    1,3      None
virtualMachines  southcentralus      Standard_NC24ads_A100_v4    1,3      None
virtualMachines  southeastasia       Standard_NC24ads_A100_v4    1,3      None
virtualMachines  SwedenCentral       Standard_NC24ads_A100_v4    2,3      None
virtualMachines  SwitzerlandNorth    Standard_NC24ads_A100_v4    3        None
virtualMachines  uksouth             Standard_NC24ads_A100_v4    1,2      None
virtualMachines  westeurope          Standard_NC24ads_A100_v4    2,3      ['NotAvailableForSubscription: Location+Zone1']
virtualMachines  westus              Standard_NC24ads_A100_v4             NotAvailableForSubscription, type: Location
virtualMachines  westus2             Standard_NC24ads_A100_v4    1,2,3    ['NotAvailableForSubscription: Location+Zones']
virtualMachines  WestUS3             Standard_NC24ads_A100_v4    2,3      None
(NC48ads_A100_v4: same pattern as NC24ads per region)
(NC96ads_A100_v4: same pattern as NC24ads per region)
virtualMachines  australiaeast       Standard_NC40ads_H100_v5    2,3      None
virtualMachines  CanadaCentral       Standard_NC40ads_H100_v5    2        None
virtualMachines  CentralIndia        Standard_NC40ads_H100_v5    1,2      None
virtualMachines  centralus           Standard_NC40ads_H100_v5    2        ['NotAvailableForSubscription: Location+Zones']
virtualMachines  eastus              Standard_NC40ads_H100_v5    2,3      ['NotAvailableForSubscription: Location+Zones']
virtualMachines  eastus2             Standard_NC40ads_H100_v5    1,2,3    ['NotAvailableForSubscription: Location+Zones']
virtualMachines  GermanyWestCentral  Standard_NC40ads_H100_v5    2,3      ['NotAvailableForSubscription: Location+Zones']
virtualMachines  IndonesiaCentral    Standard_NC40ads_H100_v5    1,2      None
virtualMachines  japaneast           Standard_NC40ads_H100_v5    1,3      None
virtualMachines  KoreaCentral        Standard_NC40ads_H100_v5    3        None
virtualMachines  MalaysiaWest        Standard_NC40ads_H100_v5    1,2      None
virtualMachines  northeurope         Standard_NC40ads_H100_v5    1        ['NotAvailableForSubscription: Location+Zones']
virtualMachines  southcentralus      Standard_NC40ads_H100_v5    2,3      ['NotAvailableForSubscription: Location+Zones']
virtualMachines  southeastasia       Standard_NC40ads_H100_v5    2,3      ['NotAvailableForSubscription: Location+Zones']
virtualMachines  SpainCentral        Standard_NC40ads_H100_v5    1,2      None
virtualMachines  SwedenCentral       Standard_NC40ads_H100_v5    2,3      ['NotAvailableForSubscription: Location+Zones']
virtualMachines  SwitzerlandNorth    Standard_NC40ads_H100_v5    3        None
virtualMachines  uksouth             Standard_NC40ads_H100_v5    1,2,3    ['NotAvailableForSubscription: Location+Zones']
virtualMachines  westeurope          Standard_NC40ads_H100_v5    2        ['NotAvailableForSubscription: Location+Zones']
virtualMachines  westus              Standard_NC40ads_H100_v5             NotAvailableForSubscription, type: Location
virtualMachines  westus2             Standard_NC40ads_H100_v5    2,3      ['NotAvailableForSubscription: Location+Zones']
virtualMachines  WestUS3             Standard_NC40ads_H100_v5    1,2,3    ['NotAvailableForSubscription: Location+Zones']
(NC80adis_H100_v5: same pattern as NC40ads per region — Spain Central & Switzerland North unrestricted)
```

## A.3 `az vm list-skus -l swedencentral --size Standard_NC / Standard_ND -o table`

```
ResourceType     Locations      Name                      Zones    Restrictions
---------------  -------------  ------------------------  -------  --------------
virtualMachines  SwedenCentral  Standard_NC16as_T4_v3     2,3      None
virtualMachines  SwedenCentral  Standard_NC24ads_A100_v4  2,3      None
virtualMachines  SwedenCentral  Standard_NC48ads_A100_v4  2,3      None
virtualMachines  SwedenCentral  Standard_NC4as_T4_v3      2,3      None
virtualMachines  SwedenCentral  Standard_NC64as_T4_v3     2,3      None
virtualMachines  SwedenCentral  Standard_NC8as_T4_v3      2,3      None
virtualMachines  SwedenCentral  Standard_NC96ads_A100_v4  2,3      None

virtualMachines  SwedenCentral  Standard_ND96amsr_A100_v4  3       None
```

## A.4 `az vm list-skus --size Standard_ND --all -o table` (H100/H200/A100/MI300X rows)

```
virtualMachines  CanadaCentral       Standard_ND96isr_H100_v5    2      None
virtualMachines  ItalyNorth          Standard_ND96isr_H100_v5    3      None
virtualMachines  NorwayEast          Standard_ND96isr_H100_v5    1,3    None
virtualMachines  SoutheastUS         Standard_ND96isr_H100_v5           None
virtualMachines  SwedenCentral       Standard_ND96isr_H100_v5    1      ['NotAvailableForSubscription: Location+Zones']
virtualMachines  uksouth             Standard_ND96isr_H100_v5    3      None
virtualMachines  westeurope          Standard_ND96isr_H100_v5    2      ['NotAvailableForSubscription: Location+Zones']
(australiaeast, centralus, eastus, eastus2, japaneast, KoreaCentral, PolandCentral,
 SouthAfricaNorth, southcentralus, UAENorth, westus2 … all NotAvailableForSubscription)

virtualMachines  PolandCentral       Standard_ND96isr_H200_v5    3      None
virtualMachines  australiasoutheast  Standard_ND96isr_H200_v5           None
virtualMachines  ukwest              Standard_ND96isr_H200_v5           None
(FranceCentral, ItalyNorth, SpainCentral, SwitzerlandNorth, uksouth … NotAvailableForSubscription)

virtualMachines  SwedenCentral       Standard_ND96amsr_A100_v4   3      None
virtualMachines  CanadaEast          Standard_ND96amsr_A100_v4          None
(eastus2, ItalyNorth, southcentralus, uksouth, westeurope, westus2 … NotAvailableForSubscription)

virtualMachines  FranceCentral       Standard_ND96isr_MI300X_v5  1      None
virtualMachines  westus              Standard_ND96isr_MI300X_v5         None
(GB200/GB300 v6: NotAvailableForSubscription everywhere relevant)
```

## A.5 GPU quota check — empty in every candidate region

```
mariagrandury@Marias-MacBook-Air-7 ~ % for r in spaincentral uksouth norwayeast italynorth polandcentral swedencentral francecentral switzerlandnorth; do
  echo "== $r"
  az vm list-usage -l $r -o table | grep -Ei "A100|H100|H200"
done
== spaincentral
== uksouth
== norwayeast
== italynorth
== polandcentral
== swedencentral
== francecentral
== switzerlandnorth
```

## A.6 Families truly absent (not a formatting fluke)

```
mariagrandury@Marias-MacBook-Air-7 ~ % az vm list-usage -l spaincentral -o json | grep -i "NCAds\|NDS\|A100\|H100" || echo "really absent"
really absent
```

## A.7 Offer type — Spot/low-priority supported

```
mariagrandury@Marias-MacBook-Air-7 ~ % az rest --method get \
  --url "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)?api-version=2022-12-01" \
  --query subscriptionPolicies.quotaId -o tsv
EnterpriseAgreement_2014-09-01
```

## A.8 No Spot/low-priority usage counters yet

```
mariagrandury@Marias-MacBook-Air-7 ~ % az vm list-usage -l spaincentral -o table | grep -i "spot\|low-priority"
(no output)
```

## A.9 Retail Prices API — the exact meters (USD/h, Linux)

```
spaincentral       Standard_NC40ads_H100_v5         NC40adsH100v5                                 9.0740 USD/h
spaincentral       Standard_NC40ads_H100_v5         NC40adsH100v5 Spot                            2.7975 USD/h
spaincentral       Standard_NC40ads_H100_v5         NC40adsH100v5 Spot                            2.2376 USD/h
spaincentral       Standard_NC40ads_H100_v5         NC40adsH100v5 Low Priority                    1.8150 USD/h
spaincentral       Standard_NC80adis_H100_v5        NC80adisH100v5 Low Priority                   3.6300 USD/h
spaincentral       Standard_NC80adis_H100_v5        NC80adisH100v5                               18.1480 USD/h
spaincentral       Standard_NC80adis_H100_v5        NC80adisH100v5 Spot                           5.5950 USD/h
spaincentral       Standard_NC80adis_H100_v5        NC80adisH100v5 Spot                           4.4753 USD/h
uksouth            Standard_ND96isr_H100_v5         ND96isrH100v5 Spot                           22.7119 USD/h
uksouth            Standard_ND96isr_H100_v5         ND96isrH100v5 Low Priority                   24.5800 USD/h
uksouth            Standard_ND96isr_H100_v5         ND96isrH100v5                               122.9000 USD/h
norwayeast         Standard_ND96isr_H100_v5         — no meters returned
italynorth         Standard_ND96isr_H100_v5         — no meters returned
polandcentral      Standard_ND96isr_H200_v5         ND96isrH200v5 Spot                          130.5920 USD/h
polandcentral      Standard_ND96isr_H200_v5         ND96isrH200v5                               130.5920 USD/h
swedencentral      Standard_NC96ads_A100_v4         NC96ads_A100_v4                              19.1000 USD/h
swedencentral      Standard_NC96ads_A100_v4         NC96ads_A100_v4 Spot                          7.3229 USD/h
swedencentral      Standard_NC96ads_A100_v4         NC96ads_A100_v4 Spot                          5.8580 USD/h
swedencentral      Standard_NC96ads_A100_v4         NC96ads_A100_v4 Low Priority                  3.8200 USD/h
switzerlandnorth   Standard_NC80adis_H100_v5        NC80adisH100v5                               19.9630 USD/h
switzerlandnorth   Standard_NC80adis_H100_v5        NC80adisH100v5 Spot                           6.7734 USD/h
switzerlandnorth   Standard_NC80adis_H100_v5        NC80adisH100v5 Spot                           5.4180 USD/h
switzerlandnorth   Standard_NC80adis_H100_v5        NC80adisH100v5 Low Priority                   3.9930 USD/h
```

Where two Spot meters exist for one SKU, billing follows the current market
price (use the higher for planning). Query used:

```bash
python3 - <<'EOF'
import json, urllib.request, urllib.parse
PAIRS = [("spaincentral","Standard_NC40ads_H100_v5"), ("spaincentral","Standard_NC80adis_H100_v5"),
         ("uksouth","Standard_ND96isr_H100_v5"), ("norwayeast","Standard_ND96isr_H100_v5"),
         ("italynorth","Standard_ND96isr_H100_v5"), ("polandcentral","Standard_ND96isr_H200_v5"),
         ("swedencentral","Standard_NC96ads_A100_v4"), ("switzerlandnorth","Standard_NC80adis_H100_v5")]
for region, sku in PAIRS:
    filt = f"armRegionName eq '{region}' and armSkuName eq '{sku}' and priceType eq 'Consumption'"
    url = "https://prices.azure.com/api/retail/prices?" + urllib.parse.urlencode({"$filter": filt})
    items = json.load(urllib.request.urlopen(url, timeout=30)).get("Items", [])
    for i in items:
        if "Windows" not in i["productName"]:
            print(f"{region:18s} {sku:32s} {i['meterName']:<42s} {i['retailPrice']:>9.4f} USD/h")
EOF
```

# Appendix B — decision log

- **Sweden Central rejected** despite being the cheapest EU tier on paper: the
  subscription cannot deploy any H100 SKU there (A.2/A.4); only A100s remain.
- **Spain Central chosen** as economy region: the only cheap-H100 (NC v5)
  region in Europe for this subscription, and its low-priority meter
  ($1.82/GPU-h) even undercuts its own Spot price.
- **UK South chosen** for the 8×H100 ND nodes (Norway/Italy list the SKU but
  return no price meters; Poland's H200 "Spot" equals PAYG). *Reversed
  2026-08-26 → Canada Central: the Spot/low-priority meter this choice rested
  on is unreachable for any H100 SKU on this subscription, and Canada Central
  is both allow-list clear and marginally cheaper dedicated ($122.40 vs
  $122.90/node-h).*
- **EA offer confirmed** → the entire plan runs on discounted meters; dedicated
  is only the eviction-thrash fallback.
- **200-language setting dropped, ×3 seeds moved to L=100** (2026-08-13);
  the validation set follows the trained languages (99 + English), not the
  old 199 list.
- **Ladder endpoints corrected** (2026-08-14): the sweep's earlier config file
  mistakenly used 75M/1.4B; fixed to 90M (L15×d768, 92.9M non-emb) and 1.7B
  (L30×d2304, 1.672B non-emb) per the plan.
- **Reviewed hyperparams made the source of truth** (2026-08-14): the
  unreviewed `hyperparams_predictivity.json` (another session's output) and its
  generator were deleted. The sweep now reads the two reviewed files —
  `hyperparams_deep.json` (deep baseline) and `hyperparams_shallow.json`
  (shallow depth-intervention variant, retargeted to the same six sizes) —
  selected by `--arch` in both launchers; D = 100·N schedules are stored in
  each config's `predictivity` block (added 2026-08-14, generators emit). Peak
  LRs come from the 6ND law evaluated at each run's OWN budget
  (C = 6·N·100N — commit 8be0eac, replacing a fixed-100B-token fit): 90M
  1.428e-3, 175M 1.217e-3, 350M 1.029e-3, 600M 8.98e-4, 1B 8.00e-4, 1.7B
  6.93e-4 (deep; shallow within 2%). Whether this law is too hot at the
  small end is the open question in `90M-rung-anomaly.md`. FLOPs and cost
  totals are unchanged (same N, same D); a shallow level differs by ±5% FLOPs
  at most (its N deviates ≤5.2% from the deep targets).
- Cluster-side caveat (resolved with the deletion): the unreviewed file's
  `nodes`×`micro_batch_size` pairs mostly didn't divide GBS 504 on the
  4-GPU-per-node cluster. Every pair in `hyperparams_deep.json` now does
  (audited 2026-08-14); Azure was never affected (train.sh auto-shrinks MBS).
