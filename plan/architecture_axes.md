# The model axes: activation (built), optimizer (blocked)

Decided 2026-10-03. Two new design axes were asked for — the optimizer and the
activation function — each as its own ladder, 90M–1.7B, deep, scheme A, T=1, at
L ∈ {8, 15, 30} (L1 until 2026-10-05, see below). The activation half is
registered and launchable. The optimizer
half cannot be launched from this checkout; see the blocker below.

## Why three language settings, and why those three

`MIN_PAIRS = 3` (rule 5, `analysis/RULES.md`). A new ladder at one setting
pairs with its baseline twin at that setting and yields exactly **one**
mono-axis pair, so it would appear in the scaling figures and contribute
nothing to any decision-accuracy table — the trap `plan/l1_third_family.md`
documents for L = 1 and `plan/3b_models.md` for the 3B 2×2.

Three settings give three pairs, the bare minimum. They were chosen to span the
axis the paper is about rather than to cluster:

| setting | why |
| --- | --- |
| **L = 8** | the modal multilingual setting: both schemes, both depths, and the trained 3B cells. |
| **L = 15** | between the two, both schemes; its list contains L8's and is contained in L30's. |
| **L = 30** | near the top of the language range, both schemes, a 3B cell. |

**L = 1 was dropped on 2026-10-05.** It was the first choice, as the
monolingual anchor, but an activation decision is read per task over the pairs
that train the task's language, and L1 trains English only: every non-English
task had at most two activation pairs, under `MIN_PAIRS`, so the axis would
have been readable on English alone (at 1B, 108 of the 110 tasks with three
pairs). The lists nest (L8 ⊂ L15 ⊂ L30), so at L ∈ {8, 15, 30} every L8
language has three pairs. The 90M L1 swiglu run had already trained; it stays
on disk, off the grid and out of `models.json`.

The alternative with the same cost — three variants at one L, pairing against
`arch` and `list` instead — answers "is the effect robust to depth and list
choice?" rather than "does it survive more languages?". The second is this
paper's question.

**Limitation to state, not discover.** Three pairs is the minimum, not a
comfortable number: a per-task DA over three pairs can only take 0, ⅓, ⅔, 1.
The pooled-over-tasks ratio the figures draw is fine; per-task verdicts on this
axis will not be. A fourth setting buys one more pair and little else.

## Cost

Measured from the completed runs, not extrapolated (2026-10-02):

| | per 6-rung ladder |
| --- | ---: |
| training | **921 node-h** (1,036 at the launcher's `ITER_MS`, which carries headroom) |
| eval + BPB + conversion | ~370 node-h for an `ALL_LANGUAGES_RUNS` cell, ~200 otherwise |

**~1,120–1,290 node-h per ladder**; the activation axis at three settings is
**~3,400 node-h**. The two top rungs are 84 % of the training cost (1.7B 546,
1B 226), and 1.7B cannot be dropped — it is the reference DA-size is measured
against. No new data is needed: all three settings' mixtures are built and
staged.

## The activation ladder (built)

`swiglu` is a ladder — an entry in `HYPERPARAMS` and in `LADDERS` (arch deep,
activation swiglu, optimizer ademamix) — so it reuses the whole launcher: the
ladder token of the cell name carries it (`lm-350M-L8-swiglu-seed1904`),
`seeds_for` keeps it single-seed, and `ladders_for` is what restricts it to the
three settings (scheme A's `ladders_by_L`, not its `ladders`).

**It is deep's shape at a matched parameter count.** Megatron's `--swiglu` MLP
holds three weight matrices per layer against XIELU's two, so at an unchanged
FFN width the ladder would carry ~33 % more non-embedding parameters and the
rung labels would stop meaning the same thing across ladders. Solving
`3 × ffn_swiglu = 2 × ffn_deep` gives `ffn = 8/3 × hidden`; the attention term
cancels. `hyperparams/find_hyperparams_swiglu.py` derives the file — it is not
a search, every other shape parameter is copied from `hyperparams_deep.json`:

| rung | ffn deep | ffn swiglu | N deep | N swiglu | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 90M | 3072 | 2048 | 92,897,280 | 92,897,280 | 0.000 % |
| 175M | 4096 | 2752 | 176,160,768 | 177,209,344 | +0.595 % |
| 350M | 5120 | 3392 | 344,064,000 | 342,425,600 | −0.476 % |
| 600M | 6144 | 4096 | 594,542,592 | 594,542,592 | 0.000 % |
| 1B | 7168 | 4800 | 944,111,616 | 947,322,880 | +0.340 % |
| 1.7B | 9216 | 6144 | 1,672,151,040 | 1,672,151,040 | 0.000 % |

Rounding to a multiple of 64 keeps the GEMMs tensor-core friendly; the residual
is well inside the −5.2 %..+3.7 % the **shallow** ladder already spans against
deep. Everything downstream follows each cell's own N, as it does for shallow:
the budget D = 100 × N, the LR from the 6ND law at that budget, and the
checkpoint grid. The analysis keys `NON_EMB` on the rung label, so both ladders
plot at the same nominal x.

The micro-batch is **copied from deep**, not re-derived: deep's values are
hand-tuned memory caps (`suggest_mbs` proposes 24 at 90M against the 7 that
fits), and the gated MLP's two ffn-wide activations at 2/3 the width come to
about the same footprint.

`ITER_MS["swiglu"]` borrows deep's values — the same stand-in the shallow 1B
uses. FLOPs per token are matched by construction, so it is the right starting
guess. **Re-measure from a clean run and replace them.**

## A ladder is not one axis

The ladder token (`deep`, `shallow`, `swiglu`) names a trained model
configuration, not a level of one axis. A `(deep, shallow)` pair is an ARCH
(depth) decision; a `(deep, swiglu)` pair is an ACTIVATION one; `(shallow,
swiglu)` moves both and is neither. Pooling them would report a depth decision
three times over and call it an architecture effect — the same conflation that
makes a collapsed `list` axis wrong (rule 15 in
`src/signal-and-noise/analysis/RULES.md`).

`launch_trainings.LADDERS` is the registry: per ladder, its `arch`
(deep|shallow), `activation` (xielu|swiglu) and `optimizer` (ademamix) levels.
The launcher takes the levels (`--arch`, `--activation`, `--optimizer`) and
resolves the ladder with `ladder_of()`, refusing a combination no ladder
trains. `ladder_report.py` publishes the levels as meta columns beside
`ladder`, the analysis loader carries all four, and
`analysis/utils.design_axes` reads the levels from the frame, so `arch`,
`activation` and `optimizer` are each their own axis (a cached CSV from before
the split, `arch` = the token, goes through one compatibility branch in the
loader). Verified on the trained grid when the axes were first split
(2026-10-03): every existing mono-axis count is unchanged (L 39, arch 10,
list 6, T 4, lang2 3, en 1 — 63 in all; since 2026-10-05 list, lang2 and en
are one `scheme` axis, plan/decision_accuracy.md §9).

A ladder added to `HYPERPARAMS` without a `LADDERS` entry raises rather than
being silently pooled into an axis.

## The optimizer axis is blocked

`--optimizer` in this Megatron accepts **`adam | sgd | ademamix`** only
(`megatron/training/arguments.py:1538`), and there is no Muon implementation
anywhere in the checkout. The mechanism is ready — `LADDERS` carries an
`optimizer` level, the launcher's `--optimizer` selects it, and `cell_env`
emits `OPTIMIZER` whenever a ladder differs from the baseline — but nothing can
consume it yet. Muon would be an optimizer level (a new ladder at arch deep,
activation xielu, optimizer muon), never an architecture. Three ways forward:

1. **Implement Muon in the fork.** Newton–Schulz orthogonalisation on the 2D
   parameters, an AdamW path for embeddings / norms / the head, and an
   interaction with `--use-distributed-optimizer` to settle. Real work, and it
   would want its own validation before 3,400 node-h ride on it.
2. **Use AdamW instead** (`--optimizer adam`, supported today). Arguably the
   better paper contrast: AdEMAMix against the baseline everyone knows, and
   this project already has a live question about AdEMAMix's slow-EMA timescale
   (`plan/90M-rung-anomaly.md`, the β₃ = 0.9999 endpoint against a 4,500-iter
   90M run).
3. **Defer**, and ship the activation axis alone.

Nothing is registered for it, so the grid is unchanged until the choice is made.

## Launching

Data is built and all 18 cells are launched (90M L8/L30 finished on
2026-10-05); the launcher is idempotent, so the commands below skip finished
or queued cells and resume the rest rather than starting them fresh.

```bash
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain
python3.11 launch_trainings.py cscs --activation swiglu --dry-run   # 18 cells
python3.11 launch_trainings.py cscs --activation swiglu             # then for real
```

18 cells = 6 rungs × 3 settings. The watcher picks them up without a flag: it
walks `predictivity_cells()` over every ladder.

**Check on the first run** that the model really is gated — `--swiglu` in the
rank-0 argument dump, and `ffn_hidden_size` at the value in the table above.
`megatron_args.sh` defaults `ACTIVATION` to `xielu`, so a cell launched without
the variable trains the baseline silently.
