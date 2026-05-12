# Seed-split generalization: `seeds_28_1797` → `seeds_1904`

## Headline metrics

|  | DA-size | DA-ckpt |
|---|---:|---:|
| Exact-variant agreement (lang-level) | 7% (1/14) | 7% (1/14) |
| **Family-level agreement** (lang-level) | 14% (2/14) | 14% (2/14) |
| Pearson r between splits (over all variant cells) | +0.272 (n = 264) | +0.694 (n = 264) |
| **Spearman ρ on global variant ranking** | +0.843 | +0.903 |
| Retention of train-best (r_test / r_test_best, mean across langs) | 53% (n = 12) | 73% (n = 12) |

**Family** groups together algebraically near-equivalent variants (e.g. the dispersion cluster: `dispersion`/`mpd`/`range`/`quartile_deviation`/`rms_deviation`/`aad`). At n_mixes=3, members of a family correlate at r ≥ 0.999 so exact-variant equality is overly strict.

**Retention** is how much of the test-split's best r the train-picked variant captures on the test split. 100% = train-pick is also test-best; lower numbers mean we lose predictive correlation by picking on the train split.

## DA-size — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `discrepancy` (discrepancy) | +0.560 | +0.132 | `mad` (robust) | +0.400 |  |  |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `iqr` (rel_spread) | +0.592 | +0.217 | `dispersion_shifted` (discrepancy) | +0.299 |  |  |
| es | `dispersion_shifted` (discrepancy) | +0.537 | +0.461 | `dispersion_shifted` (discrepancy) | +0.461 | ✅ | ✅ |
| eu | `discrepancy` (discrepancy) | +0.294 | +0.074 | `dist_std` (dispersion) | +0.571 |  |  |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `star_discrepancy_shifted` (discrepancy) | +0.552 | +0.436 | `dist_std` (dispersion) | +0.524 |  |  |
| ja | `star_discrepancy_shifted` (discrepancy) | +0.096 | -0.373 | `mad` (robust) | +0.277 |  |  |
| ru | `iqr` (rel_spread) | +0.607 | +0.665 | `dispersion` (dispersion) | +0.735 |  |  |
| sw | `quartile_deviation` (dispersion) | +0.492 | +0.289 | `discrepancy` (discrepancy) | +0.397 |  |  |
| th | `tukey` (depth) | +0.528 | +0.575 | `projection` (depth) | +0.708 |  | ✅ |
| tr | `rel_mpsd` (rel_spread) | +0.821 | +0.320 | `mad` (robust) | +0.472 |  |  |
| vi | `dist_std` (dispersion) | +0.687 | +0.083 | `discrepancy` (discrepancy) | +0.510 |  |  |
| zh | `dispersion_shifted` (discrepancy) | +0.258 | +0.013 | `rel_std` (rel_spread) | +0.237 |  |  |

## DA-ckpt — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `discrepancy` (discrepancy) | +0.489 | +0.291 | `mpd` (dispersion) | +0.374 |  |  |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `iqr` (rel_spread) | +0.575 | +0.339 | `mad` (robust) | +0.498 |  |  |
| es | `aad` (dispersion) | +0.428 | +0.306 | `dispersion_shifted` (discrepancy) | +0.472 |  |  |
| eu | `mad` (robust) | +0.321 | +0.185 | `rel_dispersion` (rel_spread) | +0.222 |  |  |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `discrepancy` (discrepancy) | +0.466 | +0.330 | `dispersion_shifted` (discrepancy) | +0.377 |  | ✅ |
| ja | `quartile_deviation` (dispersion) | +0.007 | +0.241 | `dispersion_shifted` (discrepancy) | +0.373 |  |  |
| ru | `rel_mpd` (rel_spread) | +0.519 | +0.438 | `aad` (dispersion) | +0.460 |  |  |
| sw | `mpsd` (dispersion) | +0.153 | +0.017 | `tukey` (depth) | +0.258 |  |  |
| th | `mad` (robust) | +0.477 | +0.304 | `dist_std` (dispersion) | +0.441 |  |  |
| tr | `quartile_deviation` (dispersion) | +0.476 | +0.385 | `quartile_deviation` (dispersion) | +0.385 | ✅ | ✅ |
| vi | `quartile_deviation` (dispersion) | +0.368 | +0.262 | `gini` (discrepancy) | +0.448 |  |  |
| zh | `tukey` (depth) | +0.089 | +0.146 | `star_discrepancy` (discrepancy) | +0.150 |  |  |
