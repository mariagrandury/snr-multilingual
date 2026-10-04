# INCLUDE, Gemini-rewritten (`rfgm`) — item-level comparison

The first complete `rfgm` set: all 43 INCLUDE tasks, rewritten by
`gemini-3.8-flash` through `rewrite_items_gemini.py online` (2026-09-21/22).
This compares the **items**, not model scores — no `rfgm_*` evals exist yet.
Per-task numbers in `rfgm_include_items.csv`.

| | |
|---|---:|
| tasks | 43 |
| items rewritten | 21,929 |
| items rejected | 158 (0.72 %) |
| source items | 22,087 |
| items flagged `self_ref` | 265 (1.2 %) |

Every source item is accounted for: 21,929 + 158 = 22,087.

## The answer-shape leak

The rewriter is never told which option is correct, but it can often work it
out, and a model that writes the true statement more fully than the false
ones leaves a cue a small model can follow without knowing anything. What
matters is not the absolute rate — the originals already carry the artefact —
but whether the rewrite **added** to it.

| | gold is single longest | mean tokens, gold | mean tokens, distractors |
|---|---:|---:|---:|
| original | 23.24 % | 13.43 | 12.78 |
| rewritten | 23.34 % | 13.94 | 13.28 |

Chance is 25 %. Both sides sit below it, the rewrite moves the rate by
**+0.10 pp**, and the gold−distractor gap is unchanged (0.65 → 0.66 tokens):
the rewrite lengthens gold and distractors equally. No length tell was
introduced. This reproduces the 230-item pilot result (23.4 % vs 23.3 %) at
ninety times the scale.

## Rejections

| reason | n |
|---|---:|
| duplicate source options | 68 |
| script changed | 46 |
| a choice appears in the stem | 25 |
| a choice changed script | 7 |
| unparseable JSON | 7 |
| duplicate choices | 5 |

`duplicate source options` is the largest and is a property of the source,
not the rewrite: four distinct continuations cannot come out of fewer than
four distinct options, so those items are refused before the call.

The remaining `script changed` rejects are concentrated in Telugu, whose
source options are often bare option labels (`"a, b"`, `"b, c, d"`) — the
rewrite renders them in Telugu and the check, correctly, sees a script
change. These items are unusable in any formulation: the harness scores
`stem + " " + choice` one choice at a time, so an option naming other options
never appears beside what it refers to.

**A per-language reject gradient was found and fixed here.** The first pass
rejected 318 items for `a choice changed script`, 155 of them Japanese
(30.9 % of that task) against 0–2 % elsewhere: `script_of` took the majority
Unicode script per choice, and Japanese mixes kanji and kana *within* a
sentence, so the majority flipped for reasons unrelated to drift. Folding
kana/katakana/hangul/han into one class, and applying the per-choice check
only where the source wrote the option in the item's own script, recovered
250 of 257 Japanese items and cut that reason from 318 to 7. Uneven item loss
across languages is the one bias a per-language comparison cannot absorb, so
this mattered more than the raw count suggests.

## `self_ref`

265 items carry an option that points at other options (`"a, b"`, "all of the
above"). They are **flagged, not dropped**: the originals and the `rf_` twins
keep them, and dropping them here alone would score the three sets on
different item sets. `--drop-self-ref` drops them once that decision is made.
