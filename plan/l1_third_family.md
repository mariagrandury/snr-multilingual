# A third data family at L = 1

*2026-09-21, revised 2026-09-22. Decision doc: what a second English mixture
at L = 1 should differ by, so that the L = 1 rung enters the per-L
decision-accuracy tables at all. Numbers measured from the build artifacts on
capstor/iopsstor, the source corpora under
`/capstor/store/cscs/swissai/infra01/datasets/`, and
`analysis/rq00_gate_and_curves/pretraining/predictivity/above_random_mask.csv`.*

> **Revision (09-22).** The first draft recommended FineWeb-Edu. That was
> wrong: the English build in use is already edu-filtered, so a second
> edu-filtered corpus varies the corpus while holding the interesting knob
> fixed. The axis is the **edu filter**, and the recommendation is now
> `dclm_processed`.*

## The question

L = 1 is 100 % English, so three of the four data axes are undefined there: the
language lists (A vs B), the temperature (AT3) and the second language (ZH/ES)
all need at least two languages. Only **depth** survives, which leaves two
families — `L1-deep` and `L1-shallow` — and `2·1/2 = 1` pair.

Rule 5 (`analysis/RULES.md`) needs `MIN_PAIRS = 3`, and because a pair count is
triangular over `k` families (0, 1, 3, 6, …, never 2), three pairs means
**three families**. So L = 1 is dropped from every per-L DA table today:
`da_by_L_per_task_multi_axes.csv` carries L ∈ {2, 8, 15, 30, 50} and no L = 1 row, and the
panel is labelled `L1 (no pairs yet)`.

One more family takes L = 1 from 1 pair to 3. This is exactly what ZH did for
L = 2 (`launch_trainings.py:238-244`). The question is what that third family
should differ by.

## What the English data actually is

| finding | evidence |
| --- | --- |
| There is **one** English build for the whole sweep: `english_dclm`, **184.0B tokens in 137.7M documents**, a single 736 GB file. | `english_dclm.bin` = 736,000,002,892 B ÷ 4 (int32); `.idx` = (2,754,732,422 − 42) ÷ 20 |
| It is **symlinked** into every scheme directory, on the capstor master and the iopsstor stage alike. Every trained cell reads one physical file. | `launch_builds.sh:106-117`, `stage_to_iopsstor.sh:50-61` |
| At L = 1 the blend is `1.0 english_dclm`; at L ≥ 2 it is `0.50 english_dclm + 0.50 fineweb_L<L>`. **Same pool, drawn deeper — there is no "first half" and "second half".** | `launch_trainings.py:492-502` |
| Megatron shuffles all 137.7M documents with the *training* seed, so an L = 1 1.7B draws 167.2B tokens = 0.91 epochs ≈ a random 91 % of the pool, overlapping ~91 % of every other cell's English. | `gpt_dataset.py:443-447, 611-617` |
| The build is a **deterministic prefix walk** in sorted filename order until a token budget is hit — no shuffle, no sampling, and no seed/offset/file-range argument in either build script. | `create_data_mixture.py:414-420, 742-780`; CLIs at `:848-921` and `build_data_mixtures.py:307-345` |
| It consumed **files 0–334 of 3,530**. About **3,195 files / ~1,480B tokens of DCLM have never been read** — roughly 8.8× a second full build. | doc-count match against cumulative parquet rows |
| **Nothing checks the English epoch count.** `fineweb_source` short-circuits at L = 1, so `undersized_build` never runs on an English build; an undersized one would silently repeat. | `launch_trainings.py:580-581`, `:513-561` |
| The builder never repeats: a source short of its target yields a *smaller* build plus a warning. So the realized size, not the target, must cover the largest run. | `build_data_mixtures.py:37-40`, `create_data_mixture.py:830-834` |

### The axis the third family should vary: the **edu filter**

The English build is `dclm-edu-filterrobots_fine` — DCLM put through an
educational-quality classifier. So the natural contrast is not "another
edu-filtered corpus" (which is what an earlier draft of this document
proposed, wrongly): it is **the same web text without the edu filter**.

Measured on capstor 2026-09-22 with the project's own estimator
(`create_data_mixture.estimate_tokens_per_file_byte`, Apertus-70B-2509), so
these are the tokenizer's tokens, not a byte heuristic:

| corpus | edu filter | layout | files | size | tok/byte | est. tokens | vs the 184 B build |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dclm-edu-filterrobots_fine` **(in use)** | **yes** | flat | 3,530 | 3.8 TB | 0.509 | ~1,935 B | — |
| **`dclm_processed`** | **no** | **flat** | 5,834 | 6.3 TB | 0.523 | **~3,282 B** | **18×** |
| **plain FineWeb** (`HuggingFaceFW/fineweb/data`) | **no** | nested, 104 crawls | 25,868 | 55.6 TB | 0.350 | **~19,418 B** | **106×** |
| `dclm_raw` | no (**and no robots filter**) | nested, 10 shards | 27,938 | 6.3 TB | 0.522 | ~3,299 B | 18× |
| `HuggingFaceFW/fineweb-edu` | yes | nested | 2,784 | 6.5 TB | 0.354 | ~2,290 B | 12× |
| `Nemotron-CC-v2.1_High-Quality` | quality, not edu | nested | 20 | 0.05 TB | 0.557 | ~26 B | **0.14× — too small** |

**`dclm_processed` is the same pipeline with the edu classifier removed, and
the metadata proves it** rather than the name suggesting it. Its per-document
metadata keys are *exactly* the current corpus's minus the two edu fields:

| corpus | metadata keys |
| --- | --- |
| `dclm-edu-filterrobots_fine` | `edu_int_score`, `edu_score`, `fasttext_score`, `file_path`, `language`, `language_score`, `pii_count`, `url` |
| `dclm_processed` | `fasttext_score`, `file_path`, `language`, `language_score`, `pii_count`, `url` |

Same robots handling, same PII counting, same language ID, same fastText
quality score. One knob.

**Only the `text` column is ever read** (`columns=["text"]` at
`create_data_mixture.py:360, 645, 757`), so the differing column sets of the
other candidates are irrelevant — the only structural issue is flat vs nested
discovery, and `dclm_processed` is **flat**, exactly like the corpus in use.

### Is any of this already tokenized?

Yes, but not usefully. Two pre-tokenized sets exist in Megatron `MMIDIDX`
format under `datasets/tokenized/`, both with our tokenizer:

| tokenized set | tokens | verdict |
| --- | ---: | --- |
| `swissai-dclm-edu-filterrobots_fine-merge` | ~1,658 B (40 `.bin`) | it **is** the corpus already in use |
| `Nemotron-CC-v2.1-preprocessed/Apertus-70B-2509` | ~34 B (75 `.bin`) | far short of the 167.2 B a 1.7B draws |

Nothing is pre-tokenized for `dclm_processed`, plain FineWeb or `dclm_raw`, so
**every real option needs its own build**. And a pre-tokenized set would not
drop straight in anyway: `data_blend()` emits one prefix per mixture at L = 1,
while these ship 40–75 shards, so using one would mean either a merge pass or
a multi-prefix weighted blend — a change to the launcher, not a free lunch.

### What L = 1 can actually measure

A cell is evaluated on 13 benchmark tasks at L = 1. Through the above-random
gate: **6 clear chance at 175M** (arc_easy, hellaswag, multiblimp_eng, xnli_en,
xstorycloze_en, xwinograd_en), **8 from 600M** (+ arc_challenge, paws_en), **9
at 1.7B** (+ global_piqa). `belebele_eng_Latn`, `global_mmlu_full_en` and
`truthfulqa-multi_mc1_en` never clear it; `lambada_openai_mt_en` has no chance
level and always passes. So the item set is small but it is the strongest
English benchmarks, plus English BPB and the loss.


## Options

All four hold English constant in *quantity* (184 B target, ~167.2 B drawn by
the 1.7B) and vary what kind of English it is.

| # | the third family differs by | corpus | tokens available |
| --- | --- | --- | ---: |
| 1 | **the edu filter alone** | `dclm_processed` | ~3,282 B |
| 2 | **corpus and filter** | plain FineWeb | ~19,418 B |
| 3 | edu filter **and** robots filter | `dclm_raw` | ~3,299 B |
| 4 | nothing — leave L = 1 out of the per-L tables | — | — |

| | pros | cons |
| --- | --- | --- |
| **1. `dclm_processed`** | The cleanest knob available anywhere in this sweep: one classifier on or off, with the metadata evidence above that nothing else moved. It is the data decision practitioners actually make, and the one the paper is about — "is the edu-filter choice resolvable on a 175M proxy?" Flat layout, identical schema, so **no discovery change at all** — only a per-scheme corpus path. 18× the tokens needed. | The two corpora overlap heavily (the edu set is a subset of the same crawl), so this is a filter contrast, not a corpus contrast — a reader wanting "DCLM vs something else" will not find it here. |
| **2. plain FineWeb** | The largest pool by far (106× what the build needs) and the canonical unfiltered-web baseline, so the contrast is legible to anyone outside the project. Likely the biggest effect of the four, which is what a DA measurement most needs. | Two things change at once — corpus *and* filter — so a rank flip cannot be attributed to either. Needs `discover_dclm_files()` to recurse (104 crawl dirs), and a crawl-range decision nobody has made (FineWeb spans 2013–2024; the edu build does not). |
| **3. `dclm_raw`** | Same size as option 1, and would additionally measure what the robots filter costs. | Two knobs again, and the second one is **robots exclusion** — training on text whose publishers opted out, in a model the project publishes. Not worth it for one DA pair. |
| **4. nothing** | Free. L = 1 is structurally thin because it has no language axis, and saying so is honest. | The L = 1 rung stays blank in every per-L table, and the ladder's cheapest, cleanest rung contributes nothing to the headline DA question. |

## Recommendation

**Option 1 — `dclm_processed`, one new scheme, deep only, seed 1904,
175M through 1.7B.**

It is the only option that changes exactly one thing, and the thing it changes
is a decision every practitioner makes. It needs no change to file discovery,
no crawl-range judgement call, and no robots-filter compromise, and it clears
the token requirement by 18×. That makes it both the cheapest to implement and
the easiest to defend in the paper.

**Take plain FineWeb (option 2) instead if** the question is meant to be "which
corpus", not "which filter" — for instance if a reviewer would find
DCLM-vs-DCLM too narrow. The cost is the same; the interpretation is weaker
but the effect is probably larger.

**The new family must train all the way to 1.7B.** `by_L` counts a pair only
when both members are planned at the reference size, so capping it at 1B the
way ES is capped would buy zero pairs.

## Cost

Training, deep, seed 1904, per scheme (`iters × ITER_MS × nodes`). The 90M is
in the grid as it is for every scheme, and dropped from the analysis by rule
10 — so six registered cells, five of which the tables read:

| size | node-h |
| --- | ---: |
| 90M | 6 |
| 175M | 14 |
| 350M | 49 |
| 600M | 111 |
| 1B | 251 |
| 1.7B | 605 |
| **total** | **1,036** |

Plus roughly **110 node-h** of evals and BPB for the cells (measured per-cell
averages from `compute_cost.py`), and one 184B English build — tens of
node-hours on a single node, self-chaining, comparable to the original.

**~1,150 node-h per scheme**, so **~2,300 node-h** for both if both train. The
builds themselves are ~16 h and ~20–24 h on one node each; nothing but the
training decision is expensive.

## What had to change

*All of this is implemented (2026-09-22) — see "How the two builds are
generated" below for what was actually done. The list is kept as the record of
what had to move, and of the two silent-failure risks it turned on.*

- **`DATA_SCHEMES`** (`launch_trainings.py:216`): a new entry with
  `langs={1}`, `arches=("deep",)`, `seeds="single"`, `label`/`subdir`, and a
  new key naming its English corpus (the registry has `sets` and `temp` for the
  multilingual half and **no key describing the English half**).
- **`build_english`** (`build_data_mixtures.py:200-213`) is scheme-blind: fixed
  prefix, fixed `--fineweb_pct 0 --dclm_pct 100` recipe.
  `english_target_tokens()` takes no scheme either.
- **`DCLM_DIR`** (`create_data_mixture.py:117-120`) must become per-scheme
  rather than a module constant. **`discover_dclm_files` (`:414-420`) needs no
  change for `dclm_processed`** — it is flat, like the corpus in use. It would
  have to recurse only if plain FineWeb (option 2) were chosen instead.
- **`launch_builds.sh:98`** skips `L == 1`, so a scheme whose only setting is
  L = 1 gets no build job at all.
- ⚠️ **`variant_dir()`** (`launch_builds.sh:106-117`) symlinks scheme A's
  `english_dclm` into every scheme subdir. **This must be suppressed for the new
  scheme**, or its cells train on A's English and are silently identical to A —
  no error, no warning, five wasted runs. Naming the new build `english_dclm`
  inside the scheme's own subdir keeps `data_blend` unchanged.
- ⚠️ **No English epoch guard exists.** The new build must realize ≥ 167.2B
  tokens or the 1.7B repeats silently. Check the realized size, not the target.
  At 0.523 tokens/byte the 184 B target needs ~352 GB of `dclm_processed`, about
  330 of its 5,834 files — the same shape as the build in use (335 of 3,530),
  so the prefix walk behaves identically.
- ⚠️ **Analysis**: the pool picks the cells up on its own (`predictivity_all`
  has no scheme filter), but **the design axes did not**. `DESIGN_AXES` had no
  column for the English corpus, so all three deep L = 1 families collapsed
  onto one row: the three pairs that actually vary the corpus differed on
  *nothing* and were filed as multi-axis, while the mono-axis set filled with
  three `(x-deep, shallow)` pairs — the depth decision reported three times as
  three design pairs, which is worse than the honest blank L = 1 gives today.
  Fixed 2026-09-22 by adding an `en` axis (`utils.py`, plus its `AXIS_LABEL`
  entry, or `scale_convergence` drops the axis silently). Verified additive:
  every pre-existing scheme maps to the same level, and over the 38 real
  families the multi/mono/seed pair sets are unchanged (276/62/29 both ways).
  Still owed before these cells train: `TRANSFORMATIONS`
  (`rq05/transformations.py:48-57`) and `INTERVENTIONS` (`rq05/analyze.py:79-85`).
- **Headline pool, separately decided**: `pools.predictivity` hardcodes
  `"scheme": ["A", "B"]`, and its own description says it holds only the planned
  levels "so its signal is not widened by a different intervention". Adding the
  new scheme there is a deliberate choice, not a consequence of this one.
- **Name parsing** (`pretrain_progress.NAME_RE`, `ladder_report.SCHEME_OF`) is
  generated from the registry and needs no edit, provided the label is
  non-empty and unique.
- **Plot style**: `analysis/style.py` `SCHEME_DASH` needs a dash for the new
  label, or its curves are drawn identically to scheme A (the gap BT3 had until
  2026-09-22).
- **No pre-tokenized shortcut**: see above — nothing is tokenized for any of
  the non-edu candidates, and the sharded sets that do exist would need a merge
  or a multi-prefix blend.

## How the two builds are generated

*Implemented 2026-09-22. Both mixtures are built; the training decision is
deferred, so the twelve cells are registered but unlaunched.*

Both non-edu corpora get a build, in this order — `DCLMP` first because it
needs no change to file discovery, `FWEB` second.

### What each scheme names

`DATA_SCHEMES` (`launch_trainings.py`) gained two entries and one new key,
`english` — the first key in the registry that describes the **English** half
rather than the FineWeb-2 half, because at L = 1 there is no FineWeb-2 half to
vary:

| scheme | label | subdir | `english` | `english_max_year` |
| --- | --- | --- | --- | --- |
| `DCLMP` | `-dclmP` | `DCLMP/` | `.../datasets/dclm_processed/output` | — |
| `FWEB` | `-fweb` | `FWEB/` | `.../HuggingFaceFW/fineweb/data` | 2022 |

Both are `langs={1}`, `arches=("deep",)`, `seeds="single"`, no `max_size` — so
each runs 90M through the 1.7B reference, the rung `by_L` needs for a pair.
`temp`/`sets` are inert at L = 1. Cell names are e.g.
`lm-1.7B-L1-dclmP-deep-seed1904`, and because the build is written as
`english_dclm` **inside the scheme's own subdir**, `data_blend()` is unchanged.

### Why FineWeb reads one file per crawl, and stops at 2022

The builder is a deterministic prefix walk over a sorted file list, stopping at
its token target. FineWeb is 104 directories, one per Common Crawl snapshot,
and sorted crawl names are chronological — so a plain sort would have spent the
entire 184 B budget inside `CC-MAIN-2013-20` (205 files, 441 GB). That is a
2013 web snapshot, not FineWeb.

`discover_dclm_files()` therefore keeps its flat behaviour verbatim for a flat
corpus (this is what every trained cell's English used, and it is asserted to
return the same 3,530 files in the same order) and, only for a nested one,
reads **one file per crawl in rotation**. `english_max_year=2022` drops the 15
crawls after 2022, leaving DCLM's own Common Crawl window so that recency is
not a second difference. Measured: **89 crawls, 21,531 files, 28.9 TB,
~10,105 B tokens**, and the 184 B build reads **319 files spanning all 89
crawls** — about 3.6 files each.

### Keeping English BPB honest

English BPB is measured on 3,561 held-out DCLM documents, and training skips
them by **row position** in the corpus they were carved from. A different
English corpus holds the same documents in different files, so the positional
skip would carve out the wrong rows and the new families would train on the
documents scheme A holds out — a contamination asymmetry on exactly the axis
this comparison rests on.

Both corpora carry the Common Crawl `<urn:uuid:...>` WARC record id (verified
on both), so the exclusion is done by id instead:
`build_data_mixtures.py` derives `validation.dclm.ids.json` once from the
manifest's own `dclm` entry (its `first_file` and `val_doc_count`) and passes
it as `--exclude_ids`. Scheme A's command is untouched — byte-identical to the
one the 184 B build on capstor was made with.

### Two ways this could have destroyed the existing English build

1. **`variant_dir()` symlinks scheme A's `english_dclm.*` into every scheme
   subdir**, and `ln -sfn` *replaces a regular file*. Before a build that is
   silent (the cells read A's English and are identical to A, with no error);
   after it, the link would replace a finished 736 GB file. The registry now
   drives the suppression — `launch_builds.sh` carries an owns-English field
   out of `DATA_SCHEMES`, so a later unqualified `./launch_builds.sh` is safe —
   and `variant_dir` additionally refuses to turn any real file into a link.
2. **There is no English epoch guard**: `fineweb_source()` short-circuits at
   L = 1, so `undersized_build` never runs on an English build and a short one
   would repeat silently. The realized size must be read off disk (below).

Two smaller guards came with them: the build records its corpus in
`<prefix>.plan.json` and in an `english_dclm.source` sidecar, and a resume
against a different `--dclm_dir` is refused rather than appending a second
corpus to the same `.bin`.

### Running them

```bash
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data
./launch_builds.sh --dry-run          # adds exactly build-en-dclmp and build-en-fweb
BUILD_PARTITION=preemptable ./launch_builds.sh
```

Both are one-node, self-chaining (singleton successor queued before the work
starts, 25-attempt budget) and idempotent. At the rate the BT3 L30 build
measured — 88.5 B tokens in 7 h 48 m, ~11.3 B tokens/h — `DCLMP` is **~16 h**
and `FWEB` **~20–24 h** (1.5x the bytes per token at 0.350 tok/byte). Each adds
736 GB on capstor and again on the iopsstor stage.

Before trusting either build:

```bash
stat -c%s <data>/DCLMP/english_dclm.bin   # / 4 must be >= 167.2e9 (the 1.7B's draw)
```

plus: `english_dclm.plan.json` names the right corpus (and, for `FWEB`, a file
list spanning ~89 crawls and none after `CC-MAIN-2022-49`); the build log
reports a non-zero excluded-by-id count; `english_dclm.source` exists; and the
staged iopsstor copy is a real file, not a symlink into scheme A's.
