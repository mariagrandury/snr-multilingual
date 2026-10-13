---
title: "RQ8 · Subsets"
---

# RQ8 · Can a subset beat the full benchmark?

**The question.** A multilingual benchmark averages many languages, and some
of them only add noise to the others. Does a subset of a benchmark's
languages (or of its subjects, or of its items) separate the training choices
more sharply than the full set, beyond what any random subset of the same
size would do?

**How we measure it.** For every benchmark and model size (seed-1904 runs of
every cell, 90M–1.7B, languages above chance at that size, the accuracy
benchmarks and their bBPB twins), we rank the languages by their own SNR, add
them one by one, and keep the prefix with the highest SNR. Because that
prefix is chosen on the same numbers it is scored on, it always looks at
least as good as the full set. So we also draw 100 random subsets of the same
size, and a chosen subset only counts when it beats their 95th percentile.

## Explore

The first view reads the earlier sweep's replication on public models
(270M–70B): per benchmark, the full set (grey) against its best subset
(blue).

<div class="viz" data-viz="subsets"></div>

## Cost against reliability

Pick a benchmark and a size, then slide the number of languages you would
evaluate: the blue line is the SNR of the best languages first, the dashed
line a random order, and the orange diamond what random subsets of the best
size reach.

<div class="viz" data-viz="subsetCurve"></div>

## What we find

<!-- highlight: rq08_subset_selection -->

- **Without the null the lever looks universal.** The best prefix beats the
  full set in 221 of the 254 swept (benchmark, size) cells, by a median of
  +0.80 SNR.
- **Against random subsets, a win is testable in few cells, and there the
  chosen subset usually wins.** A cell is testable when the best prefix beats
  the full set, holds two or more languages and has more than 20 random
  subsets of its size. A chosen subset beats the 95th percentile of 100
  random ones in 83 of the 104 testable cells, 77 of 98 for language subsets.

- **Item subsets win SNR on a handful of items but read the 1.7B model
  worse than random items.** The best item prefix beats its null in 97–99 %
  of the tasks, on a median of 4–8 items; held out, its decision accuracy is
  0.31–0.37 against 0.52–0.56 for the full set and 0.35–0.39 for random items.
- **The items the 1.7B model solves are the exception.** Chosen on half of
  the runs and scored on the other half, they raise decision accuracy by
  +0.018 to +0.039 at 90M–600M and by nothing at 1B (see also
  [RQ12](above-chance-items.md), the in-sample version).

!!! success "Takeaway"
    A chosen subset of languages can raise a benchmark's SNR and cut its
    cost, but the comparison with a random subset of the same size is possible
    in few cells, and choosing single items by their SNR backfires. Keep
    the full benchmark unless a subset beats the random-subset null on runs
    you did not choose it on.

[Full analysis →](../signal-noise/subset_selection.md)
