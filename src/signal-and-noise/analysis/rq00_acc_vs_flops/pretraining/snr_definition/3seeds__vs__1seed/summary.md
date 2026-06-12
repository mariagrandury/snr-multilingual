# Seed-split generalization: `3seeds` → `1seed`

## Headline metrics

|  | DA-size | DA-ckpt |
|---|---:|---:|
| Exact-variant agreement (lang-level) | 7% (1/14) | 0% (0/14) |
| **Family-level agreement** (lang-level) | 43% (6/14) | 50% (7/14) |
| Pearson r between splits (over all variant cells) | +0.714 (n = 264) | +0.830 (n = 264) |
| **Spearman ρ on global variant ranking** | +0.746 | +0.940 |
| Retention of train-best (r_test / r_test_best, mean across langs) | 72% (n = 12) | 87% (n = 12) |

**Family** groups together algebraically near-equivalent variants (e.g. the dispersion cluster: `dispersion`/`mpd`/`range`/`quartile_deviation`/`rms_deviation`/`aad`). At n_mixes=3, members of a family correlate at r ≥ 0.999 so exact-variant equality is overly strict.

**Retention** is how much of the test-split's best r the train-picked variant captures on the test split. 100% = train-pick is also test-best; lower numbers mean we lose predictive correlation by picking on the train split.

## DA-size — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `discrepancy` (discrepancy) | +0.492 | +0.154 | `mad` (robust) | +0.345 |  |  |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `rel_mpd` (rel_spread) | +0.633 | +0.324 | `dispersion_shifted` (discrepancy) | +0.412 |  |  |
| es | `quartile_deviation` (dispersion) | +0.633 | +0.531 | `aad` (dispersion) | +0.536 |  | ✅ |
| eu | `iqr` (rel_spread) | +0.415 | +0.418 | `rel_std` (rel_spread) | +0.421 |  | ✅ |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `star_discrepancy_shifted` (discrepancy) | +0.749 | +0.396 | `discrepancy` (discrepancy) | +0.519 |  | ✅ |
| ja | `rel_mpsd` (rel_spread) | +0.277 | +0.005 | `mad` (robust) | +0.225 |  |  |
| ru | `quartile_deviation` (dispersion) | +0.718 | +0.676 | `quartile_deviation` (dispersion) | +0.676 | ✅ | ✅ |
| sw | `mad` (robust) | +0.288 | +0.152 | `rel_std` (rel_spread) | +0.375 |  |  |
| th | `iqr` (rel_spread) | +0.446 | +0.416 | `rel_std` (rel_spread) | +0.418 |  | ✅ |
| tr | `rel_mpsd` (rel_spread) | +0.662 | +0.401 | `mpsd` (dispersion) | +0.418 |  |  |
| vi | `dispersion_shifted` (discrepancy) | +0.577 | +0.508 | `discrepancy` (discrepancy) | +0.557 |  | ✅ |
| zh | `star_discrepancy_shifted` (discrepancy) | +0.127 | +0.048 | `rel_std` (rel_spread) | +0.117 |  |  |

## DA-ckpt — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `quartile_deviation` (dispersion) | +0.564 | +0.387 | `dispersion` (dispersion) | +0.387 |  | ✅ |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `quartile_deviation` (dispersion) | +0.649 | +0.425 | `mad` (robust) | +0.511 |  |  |
| es | `star_discrepancy_shifted` (discrepancy) | +0.492 | +0.415 | `dispersion_shifted` (discrepancy) | +0.472 |  | ✅ |
| eu | `iqr` (rel_spread) | +0.326 | +0.222 | `rel_dispersion` (rel_spread) | +0.222 |  | ✅ |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `discrepancy` (discrepancy) | +0.580 | +0.330 | `dispersion_shifted` (discrepancy) | +0.377 |  | ✅ |
| ja | `quartile_deviation` (dispersion) | +0.266 | +0.241 | `dispersion_shifted` (discrepancy) | +0.373 |  |  |
| ru | `rel_mpd` (rel_spread) | +0.613 | +0.438 | `aad` (dispersion) | +0.460 |  |  |
| sw | `dispersion` (dispersion) | +0.277 | +0.125 | `tukey` (depth) | +0.258 |  |  |
| th | `rel_dispersion` (rel_spread) | +0.527 | +0.380 | `dist_std` (dispersion) | +0.441 |  |  |
| tr | `dist_std` (dispersion) | +0.635 | +0.374 | `quartile_deviation` (dispersion) | +0.385 |  | ✅ |
| vi | `dispersion_shifted` (discrepancy) | +0.583 | +0.432 | `gini` (discrepancy) | +0.448 |  | ✅ |
| zh | `star_discrepancy` (discrepancy) | +0.147 | +0.152 | `rel_star_discrepancy` (discrepancy) | +0.161 |  | ✅ |
