# Seed-split generalization: `seeds_28_1797` → `seeds_1904`

## Headline metrics

|  | DA-size | DA-ckpt |
|---|---:|---:|
| Exact-variant agreement (lang-level) | 0% (0/14) | 7% (1/14) |
| **Family-level agreement** (lang-level) | 14% (2/14) | 36% (5/14) |
| Pearson r between splits (over all variant cells) | +0.569 (n = 264) | +0.725 (n = 264) |
| **Spearman ρ on global variant ranking** | +0.797 | +0.925 |
| Retention of train-best (r_test / r_test_best, mean across langs) | 62% (n = 12) | 78% (n = 12) |

**Family** groups together algebraically near-equivalent variants (e.g. the dispersion cluster: `dispersion`/`mpd`/`range`/`quartile_deviation`/`rms_deviation`/`aad`). At n_mixes=3, members of a family correlate at r ≥ 0.999 so exact-variant equality is overly strict.

**Retention** is how much of the test-split's best r the train-picked variant captures on the test split. 100% = train-pick is also test-best; lower numbers mean we lose predictive correlation by picking on the train split.

## DA-size — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `discrepancy` (discrepancy) | +0.512 | +0.154 | `mad` (robust) | +0.345 |  |  |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `rel_mpd` (rel_spread) | +0.582 | +0.324 | `dispersion_shifted` (discrepancy) | +0.412 |  |  |
| es | `dispersion_shifted` (discrepancy) | +0.505 | +0.507 | `aad` (dispersion) | +0.536 |  |  |
| eu | `iqr` (rel_spread) | +0.330 | +0.418 | `rel_std` (rel_spread) | +0.421 |  | ✅ |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `gini` (discrepancy) | +0.581 | +0.462 | `discrepancy` (discrepancy) | +0.519 |  | ✅ |
| ja | `discrepancy` (discrepancy) | +0.099 | -0.353 | `mad` (robust) | +0.225 |  |  |
| ru | `iqr` (rel_spread) | +0.618 | +0.569 | `quartile_deviation` (dispersion) | +0.676 |  |  |
| sw | `quartile_deviation` (dispersion) | +0.275 | +0.286 | `rel_std` (rel_spread) | +0.375 |  |  |
| th | `tukey` (depth) | +0.487 | +0.222 | `rel_std` (rel_spread) | +0.418 |  |  |
| tr | `rel_mpsd` (rel_spread) | +0.671 | +0.401 | `mpsd` (dispersion) | +0.418 |  |  |
| vi | `quartile_deviation` (dispersion) | +0.512 | +0.015 | `discrepancy` (discrepancy) | +0.557 |  |  |
| zh | `dispersion_shifted` (discrepancy) | +0.075 | +0.031 | `rel_std` (rel_spread) | +0.117 |  |  |

## DA-ckpt — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `dispersion_shifted` (discrepancy) | +0.533 | +0.300 | `dispersion` (dispersion) | +0.387 |  |  |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `quartile_deviation` (dispersion) | +0.539 | +0.425 | `mad` (robust) | +0.511 |  |  |
| es | `quartile_deviation` (dispersion) | +0.433 | +0.306 | `dispersion_shifted` (discrepancy) | +0.472 |  |  |
| eu | `iqr` (rel_spread) | +0.275 | +0.222 | `rel_dispersion` (rel_spread) | +0.222 |  | ✅ |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `discrepancy` (discrepancy) | +0.490 | +0.330 | `dispersion_shifted` (discrepancy) | +0.377 |  | ✅ |
| ja | `quartile_deviation` (dispersion) | +0.141 | +0.241 | `dispersion_shifted` (discrepancy) | +0.373 |  |  |
| ru | `rel_dispersion` (rel_spread) | +0.492 | +0.438 | `aad` (dispersion) | +0.460 |  |  |
| sw | `mpsd` (dispersion) | +0.176 | +0.017 | `tukey` (depth) | +0.258 |  |  |
| th | `mad` (robust) | +0.446 | +0.304 | `dist_std` (dispersion) | +0.441 |  |  |
| tr | `quartile_deviation` (dispersion) | +0.508 | +0.385 | `quartile_deviation` (dispersion) | +0.385 | ✅ | ✅ |
| vi | `dispersion_shifted` (discrepancy) | +0.411 | +0.432 | `gini` (discrepancy) | +0.448 |  | ✅ |
| zh | `star_discrepancy` (discrepancy) | +0.129 | +0.152 | `rel_star_discrepancy` (discrepancy) | +0.161 |  | ✅ |
