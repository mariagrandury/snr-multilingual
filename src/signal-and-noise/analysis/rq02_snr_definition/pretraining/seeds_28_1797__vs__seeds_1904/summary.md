# Seed-split generalization: `seeds_28_1797` → `seeds_1904`

## Headline metrics

|  | DA-size | DA-ckpt |
|---|---:|---:|
| Exact-variant agreement (lang-level) | 7% (1/14) | 7% (1/14) |
| **Family-level agreement** (lang-level) | 21% (3/14) | 29% (4/14) |
| Pearson r between splits (over all variant cells) | +0.478 (n = 242) | +0.600 (n = 242) |
| **Spearman ρ on global variant ranking** | -0.069 | +0.813 |
| Retention of train-best (r_test / r_test_best, mean across langs) | 61% (n = 11) | 79% (n = 11) |

**Family** groups together algebraically near-equivalent variants (e.g. the dispersion cluster: `dispersion`/`mpd`/`range`/`quartile_deviation`/`rms_deviation`/`aad`). At n_mixes=3, members of a family correlate at r ≥ 0.999 so exact-variant equality is overly strict.

**Retention** is how much of the test-split's best r the train-picked variant captures on the test split. 100% = train-pick is also test-best; lower numbers mean we lose predictive correlation by picking on the train split.

## DA-size — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `rel_dispersion` (rel_spread) | +0.519 | +0.980 | `iqr` (rel_spread) | +0.980 |  | ✅ |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `rel_mpd` (rel_spread) | +0.571 | +0.318 | `rel_star_discrepancy` (discrepancy) | +0.485 |  |  |
| es | `tukey` (depth) | +0.704 | +0.467 | `dist_std` (dispersion) | +0.671 |  |  |
| eu | `rel_mpsd` (rel_spread) | +0.166 | -0.380 | `mad` (robust) | +0.764 |  |  |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `mad` (robust) | +0.686 | +0.836 | `discrepancy` (discrepancy) | +0.961 |  |  |
| ja | `tukey` (depth) | +0.432 | +0.167 | `mad` (robust) | +0.252 |  |  |
| ru | `gini` (discrepancy) | +0.685 | +0.458 | `mad` (robust) | +0.754 |  |  |
| sw | `` () | +nan | +nan | `` () | +nan |  |  |
| th | `tukey` (depth) | +0.218 | +0.317 | `projection` (depth) | +0.899 |  | ✅ |
| tr | `discrepancy` (discrepancy) | +0.688 | +0.430 | `mad` (robust) | +0.570 |  |  |
| vi | `dispersion_shifted` (discrepancy) | +0.583 | +0.639 | `dispersion_shifted` (discrepancy) | +0.639 | ✅ | ✅ |
| zh | `rel_star_discrepancy` (discrepancy) | +0.375 | +0.066 | `projection` (depth) | +0.393 |  |  |

## DA-ckpt — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `rel_star_discrepancy` (discrepancy) | +0.048 | +0.501 | `star_discrepancy` (discrepancy) | +0.508 |  | ✅ |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `iqr` (rel_spread) | +0.580 | +0.362 | `mad` (robust) | +0.479 |  |  |
| es | `iqr` (rel_spread) | +0.472 | +0.460 | `rel_star_discrepancy` (discrepancy) | +0.563 |  |  |
| eu | `quartile_deviation` (dispersion) | +0.570 | +0.186 | `mpsd` (dispersion) | +0.246 |  | ✅ |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `gini` (discrepancy) | +0.707 | +0.333 | `dist_std` (dispersion) | +0.448 |  |  |
| ja | `star_discrepancy_shifted` (discrepancy) | +0.178 | +0.422 | `tukey` (depth) | +0.564 |  |  |
| ru | `iqr` (rel_spread) | +0.615 | +0.594 | `rel_dispersion` (rel_spread) | +0.594 |  | ✅ |
| sw | `` () | +nan | +nan | `` () | +nan |  |  |
| th | `mpd` (dispersion) | +0.470 | +0.273 | `projection` (depth) | +0.308 |  |  |
| tr | `mpsd` (dispersion) | +0.517 | +0.456 | `iqr` (rel_spread) | +0.525 |  |  |
| vi | `dispersion_shifted` (discrepancy) | +0.701 | +0.523 | `dispersion_shifted` (discrepancy) | +0.523 | ✅ | ✅ |
| zh | `dist_std` (dispersion) | +0.354 | +0.042 | `mad` (robust) | +0.347 |  |  |
