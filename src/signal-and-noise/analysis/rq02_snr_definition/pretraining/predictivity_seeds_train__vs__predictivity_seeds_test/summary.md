# Seed-split generalization: `predictivity_seeds_train` → `predictivity_seeds_test`

## Headline metrics

|  | DA-size | DA-ckpt |
|---|---:|---:|
| Exact-variant agreement (lang-level) | 2% (3/122) | 7% (8/122) |
| **Family-level agreement** (lang-level) | 3% (4/122) | 15% (18/122) |
| Pearson r between splits (over all variant cells) | -0.433 (n = 115) | +0.053 (n = 1128) |
| **Spearman ρ on global variant ranking** | +0.006 | -0.212 |
| Retention of train-best (r_test / r_test_best, mean across langs) | 68% (n = 7) | 39% (n = 45) |

**Family** groups together algebraically near-equivalent variants (e.g. the dispersion cluster: `dispersion`/`mpd`/`range`/`quartile_deviation`/`rms_deviation`/`aad`). At n_mixes=3, members of a family correlate at r ≥ 0.999 so exact-variant equality is overly strict.

**Retention** is how much of the test-split's best r the train-picked variant captures on the test split. 100% = train-pick is also test-best; lower numbers mean we lose predictive correlation by picking on the train split.

## DA-size — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| af | `` () | +nan | +nan | `` () | +nan |  |  |
| am | `` () | +nan | +nan | `` () | +nan |  |  |
| ar | `gini` (discrepancy) | +0.966 | -0.708 | `projection` (depth) | +1.000 |  |  |
| as | `` () | +nan | +nan | `` () | +nan |  |  |
| az | `` () | +nan | +nan | `` () | +nan |  |  |
| be | `` () | +nan | +nan | `` () | +nan |  |  |
| bew | `` () | +nan | +nan | `` () | +nan |  |  |
| bg | `` () | +nan | +nan | `` () | +nan |  |  |
| bn | `` () | +nan | +nan | `` () | +nan |  |  |
| bo | `` () | +nan | +nan | `` () | +nan |  |  |
| bs | `` () | +nan | +nan | `` () | +nan |  |  |
| ca | `` () | +nan | +nan | `` () | +nan |  |  |
| ceb | `` () | +nan | +nan | `` () | +nan |  |  |
| ckb | `` () | +nan | +nan | `` () | +nan |  |  |
| cs | `` () | +nan | +nan | `` () | +nan |  |  |
| cy | `` () | +nan | +nan | `` () | +nan |  |  |
| da | `` () | +nan | +nan | `` () | +nan |  |  |
| de | `rel_star_discrepancy` (discrepancy) | -0.789 | +0.998 | `projection` (depth) | +1.000 |  |  |
| dv | `` () | +nan | +nan | `` () | +nan |  |  |
| el | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `tukey` (depth) | +0.815 | +0.592 | `tukey` (depth) | +0.592 | ✅ | ✅ |
| eo | `` () | +nan | +nan | `` () | +nan |  |  |
| es | `rel_std` (rel_spread) | +1.000 | +0.259 | `rel_star_discrepancy` (discrepancy) | +0.902 |  |  |
| et | `` () | +nan | +nan | `` () | +nan |  |  |
| eu | `` () | +nan | +nan | `` () | +nan |  |  |
| fa | `` () | +nan | +nan | `` () | +nan |  |  |
| fi | `` () | +nan | +nan | `` () | +nan |  |  |
| fo | `` () | +nan | +nan | `` () | +nan |  |  |
| fr | `rel_star_discrepancy` (discrepancy) | +0.895 | +0.919 | `rel_star_discrepancy` (discrepancy) | +0.919 | ✅ | ✅ |
| ga | `` () | +nan | +nan | `` () | +nan |  |  |
| gd | `` () | +nan | +nan | `` () | +nan |  |  |
| gl | `` () | +nan | +nan | `` () | +nan |  |  |
| gmh | `` () | +nan | +nan | `` () | +nan |  |  |
| grc | `` () | +nan | +nan | `` () | +nan |  |  |
| gu | `` () | +nan | +nan | `` () | +nan |  |  |
| ha | `` () | +nan | +nan | `` () | +nan |  |  |
| haw | `` () | +nan | +nan | `` () | +nan |  |  |
| hbo | `` () | +nan | +nan | `` () | +nan |  |  |
| he | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `` () | +nan | +nan | `` () | +nan |  |  |
| hif | `` () | +nan | +nan | `` () | +nan |  |  |
| hr | `` () | +nan | +nan | `` () | +nan |  |  |
| ht | `` () | +nan | +nan | `` () | +nan |  |  |
| hu | `` () | +nan | +nan | `` () | +nan |  |  |
| hy | `` () | +nan | +nan | `` () | +nan |  |  |
| hyw | `` () | +nan | +nan | `` () | +nan |  |  |
| id | `` () | +nan | +nan | `` () | +nan |  |  |
| ig | `` () | +nan | +nan | `` () | +nan |  |  |
| is | `` () | +nan | +nan | `` () | +nan |  |  |
| it | `gini` (discrepancy) | +0.954 | +0.464 | `rel_star_discrepancy` (discrepancy) | +1.000 |  | ✅ |
| ja | `` () | +nan | +nan | `` () | +nan |  |  |
| jv | `` () | +nan | +nan | `` () | +nan |  |  |
| ka | `` () | +nan | +nan | `` () | +nan |  |  |
| kk | `` () | +nan | +nan | `` () | +nan |  |  |
| km | `` () | +nan | +nan | `` () | +nan |  |  |
| kmr | `` () | +nan | +nan | `` () | +nan |  |  |
| kn | `` () | +nan | +nan | `` () | +nan |  |  |
| ko | `` () | +nan | +nan | `` () | +nan |  |  |
| ky | `` () | +nan | +nan | `` () | +nan |  |  |
| la | `` () | +nan | +nan | `` () | +nan |  |  |
| lb | `` () | +nan | +nan | `` () | +nan |  |  |
| lo | `` () | +nan | +nan | `` () | +nan |  |  |
| lt | `` () | +nan | +nan | `` () | +nan |  |  |
| lv | `` () | +nan | +nan | `` () | +nan |  |  |
| mg | `` () | +nan | +nan | `` () | +nan |  |  |
| mi | `` () | +nan | +nan | `` () | +nan |  |  |
| mk | `` () | +nan | +nan | `` () | +nan |  |  |
| ml | `` () | +nan | +nan | `` () | +nan |  |  |
| mn | `` () | +nan | +nan | `` () | +nan |  |  |
| mr | `` () | +nan | +nan | `` () | +nan |  |  |
| ms | `` () | +nan | +nan | `` () | +nan |  |  |
| mt | `` () | +nan | +nan | `` () | +nan |  |  |
| multi | `` () | +nan | +nan | `` () | +nan |  |  |
| my | `` () | +nan | +nan | `` () | +nan |  |  |
| nds | `` () | +nan | +nan | `` () | +nan |  |  |
| ne | `` () | +nan | +nan | `` () | +nan |  |  |
| nl | `` () | +nan | +nan | `` () | +nan |  |  |
| nn | `` () | +nan | +nan | `` () | +nan |  |  |
| no | `` () | +nan | +nan | `` () | +nan |  |  |
| nrm | `` () | +nan | +nan | `` () | +nan |  |  |
| ny | `` () | +nan | +nan | `` () | +nan |  |  |
| or | `` () | +nan | +nan | `` () | +nan |  |  |
| pa | `` () | +nan | +nan | `` () | +nan |  |  |
| pl | `` () | +nan | +nan | `` () | +nan |  |  |
| ps | `` () | +nan | +nan | `` () | +nan |  |  |
| pt | `` () | +nan | +nan | `` () | +nan |  |  |
| ro | `` () | +nan | +nan | `` () | +nan |  |  |
| ru | `` () | +nan | +nan | `` () | +nan |  |  |
| rw | `` () | +nan | +nan | `` () | +nan |  |  |
| sa | `` () | +nan | +nan | `` () | +nan |  |  |
| sah | `` () | +nan | +nan | `` () | +nan |  |  |
| sd | `` () | +nan | +nan | `` () | +nan |  |  |
| se | `` () | +nan | +nan | `` () | +nan |  |  |
| si | `` () | +nan | +nan | `` () | +nan |  |  |
| sk | `` () | +nan | +nan | `` () | +nan |  |  |
| sl | `` () | +nan | +nan | `` () | +nan |  |  |
| sn | `` () | +nan | +nan | `` () | +nan |  |  |
| so | `` () | +nan | +nan | `` () | +nan |  |  |
| sq | `` () | +nan | +nan | `` () | +nan |  |  |
| sr | `` () | +nan | +nan | `` () | +nan |  |  |
| st | `` () | +nan | +nan | `` () | +nan |  |  |
| su | `` () | +nan | +nan | `` () | +nan |  |  |
| sv | `` () | +nan | +nan | `` () | +nan |  |  |
| sw | `` () | +nan | +nan | `` () | +nan |  |  |
| ta | `` () | +nan | +nan | `` () | +nan |  |  |
| te | `` () | +nan | +nan | `` () | +nan |  |  |
| tg | `` () | +nan | +nan | `` () | +nan |  |  |
| th | `` () | +nan | +nan | `` () | +nan |  |  |
| ti | `` () | +nan | +nan | `` () | +nan |  |  |
| tl | `` () | +nan | +nan | `` () | +nan |  |  |
| tr | `` () | +nan | +nan | `rel_mpsd` (rel_spread) | +0.988 |  |  |
| tt | `` () | +nan | +nan | `` () | +nan |  |  |
| ug | `` () | +nan | +nan | `` () | +nan |  |  |
| uk | `rel_mpsd` (rel_spread) | +1.000 | +0.744 | `rel_mpsd` (rel_spread) | +0.744 | ✅ | ✅ |
| ur | `mpsd` (dispersion) | +1.000 | +nan | `` () | +nan |  |  |
| uz | `` () | +nan | +nan | `` () | +nan |  |  |
| vi | `` () | +nan | +nan | `` () | +nan |  |  |
| xh | `` () | +nan | +nan | `` () | +nan |  |  |
| yo | `` () | +nan | +nan | `` () | +nan |  |  |
| yue | `` () | +nan | +nan | `` () | +nan |  |  |
| zh | `` () | +nan | +nan | `` () | +nan |  |  |
| zu | `` () | +nan | +nan | `` () | +nan |  |  |

## DA-ckpt — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| af | `` () | +nan | +nan | `mad` (robust) | -0.408 |  |  |
| am | `` () | +nan | +nan | `star_discrepancy` (discrepancy) | +0.183 |  |  |
| ar | `star_discrepancy` (discrepancy) | +0.255 | +0.147 | `projection` (depth) | +0.594 |  |  |
| as | `` () | +nan | +nan | `rel_mpsd` (rel_spread) | +0.513 |  |  |
| az | `dispersion` (dispersion) | -0.356 | -0.500 | `aad` (dispersion) | -0.500 |  | ✅ |
| be | `` () | +nan | +nan | `tukey` (depth) | +0.239 |  |  |
| bew | `` () | +nan | +nan | `` () | +nan |  |  |
| bg | `projection` (depth) | +0.636 | -0.466 | `discrepancy` (discrepancy) | +0.577 |  |  |
| bn | `discrepancy` (discrepancy) | +0.366 | +0.891 | `discrepancy` (discrepancy) | +0.891 | ✅ | ✅ |
| bo | `` () | +nan | +nan | `aad` (dispersion) | +0.209 |  |  |
| bs | `projection` (depth) | +0.485 | -0.444 | `rel_mpsd` (rel_spread) | -0.444 |  |  |
| ca | `rel_star_discrepancy` (discrepancy) | +0.360 | +0.021 | `discrepancy` (discrepancy) | +0.647 |  | ✅ |
| ceb | `` () | +nan | +nan | `` () | +nan |  |  |
| ckb | `` () | +nan | +nan | `star_discrepancy_shifted` (discrepancy) | +0.420 |  |  |
| cs | `projection` (depth) | +0.424 | -0.485 | `discrepancy` (discrepancy) | +0.898 |  |  |
| cy | `` () | +nan | +nan | `projection` (depth) | +0.110 |  |  |
| da | `star_discrepancy` (discrepancy) | +0.446 | +0.143 | `star_discrepancy` (discrepancy) | +0.143 | ✅ | ✅ |
| de | `projection` (depth) | +0.352 | +0.153 | `mad` (robust) | +0.175 |  |  |
| dv | `` () | +nan | +nan | `` () | +nan |  |  |
| el | `projection` (depth) | +0.484 | -0.427 | `iqr` (rel_spread) | +0.760 |  |  |
| en | `rel_mpsd` (rel_spread) | +0.486 | +0.438 | `iqr` (rel_spread) | +0.441 |  | ✅ |
| eo | `` () | +nan | +nan | `` () | +nan |  |  |
| es | `mad` (robust) | +0.325 | -0.314 | `rel_star_discrepancy` (discrepancy) | +0.286 |  |  |
| et | `mad` (robust) | +0.346 | -0.435 | `discrepancy` (discrepancy) | +0.378 |  |  |
| eu | `` () | +nan | +nan | `star_discrepancy` (discrepancy) | +0.497 |  |  |
| fa | `projection` (depth) | +0.236 | -0.328 | `rel_mpsd` (rel_spread) | +0.297 |  |  |
| fi | `rel_mpsd` (rel_spread) | +0.489 | -0.498 | `discrepancy` (discrepancy) | +0.577 |  |  |
| fo | `` () | +nan | +nan | `` () | +nan |  |  |
| fr | `rel_mpd` (rel_spread) | +0.311 | -0.234 | `projection` (depth) | +0.463 |  |  |
| ga | `` () | +nan | +nan | `star_discrepancy` (discrepancy) | -0.127 |  |  |
| gd | `` () | +nan | +nan | `` () | +nan |  |  |
| gl | `` () | +nan | +nan | `rel_mpsd` (rel_spread) | +0.288 |  |  |
| gmh | `` () | +nan | +nan | `` () | +nan |  |  |
| grc | `` () | +nan | +nan | `` () | +nan |  |  |
| gu | `` () | +nan | +nan | `dispersion_shifted` (discrepancy) | +0.199 |  |  |
| ha | `` () | +nan | +nan | `` () | +nan |  |  |
| haw | `` () | +nan | +nan | `` () | +nan |  |  |
| hbo | `` () | +nan | +nan | `` () | +nan |  |  |
| he | `discrepancy` (discrepancy) | +0.500 | +0.898 | `discrepancy` (discrepancy) | +0.898 | ✅ | ✅ |
| hi | `mpsd` (dispersion) | +0.346 | -0.277 | `projection` (depth) | +0.785 |  |  |
| hif | `` () | +nan | +nan | `` () | +nan |  |  |
| hr | `gini` (discrepancy) | -0.243 | -0.513 | `rms_deviation` (dispersion) | -0.513 |  |  |
| ht | `` () | +nan | +nan | `` () | +nan |  |  |
| hu | `projection` (depth) | +0.704 | +0.026 | `tukey` (depth) | +0.489 |  | ✅ |
| hy | `` () | +nan | +nan | `mad` (robust) | +0.220 |  |  |
| hyw | `` () | +nan | +nan | `` () | +nan |  |  |
| id | `projection` (depth) | +0.521 | +0.627 | `projection` (depth) | +0.627 | ✅ | ✅ |
| ig | `` () | +nan | +nan | `` () | +nan |  |  |
| is | `` () | +nan | +nan | `rel_mpsd` (rel_spread) | +0.588 |  |  |
| it | `projection` (depth) | +0.259 | -0.048 | `discrepancy` (discrepancy) | +0.275 |  |  |
| ja | `rel_mpsd` (rel_spread) | +0.958 | -0.054 | `mpsd` (dispersion) | -0.051 |  |  |
| jv | `` () | +nan | +nan | `` () | +nan |  |  |
| ka | `projection` (depth) | +0.743 | -0.550 | `discrepancy` (discrepancy) | +0.965 |  |  |
| kk | `rel_mpsd` (rel_spread) | +0.301 | +0.571 | `projection` (depth) | +0.583 |  |  |
| km | `` () | +nan | +nan | `aad` (dispersion) | +0.143 |  |  |
| kmr | `` () | +nan | +nan | `star_discrepancy` (discrepancy) | +0.761 |  |  |
| kn | `` () | +nan | +nan | `star_discrepancy_shifted` (discrepancy) | +0.374 |  |  |
| ko | `mpsd` (dispersion) | +0.225 | +0.475 | `dist_std` (dispersion) | +0.475 |  | ✅ |
| ky | `` () | +nan | +nan | `mpsd` (dispersion) | +0.803 |  |  |
| la | `` () | +nan | +nan | `projection` (depth) | +0.364 |  |  |
| lb | `` () | +nan | +nan | `` () | +nan |  |  |
| lo | `` () | +nan | +nan | `dispersion` (dispersion) | +0.277 |  |  |
| lt | `projection` (depth) | +0.507 | -0.377 | `discrepancy` (discrepancy) | +0.775 |  |  |
| lv | `aad` (dispersion) | -0.277 | -0.232 | `aad` (dispersion) | -0.232 | ✅ | ✅ |
| mg | `` () | +nan | +nan | `aad` (dispersion) | +0.524 |  |  |
| mi | `` () | +nan | +nan | `` () | +nan |  |  |
| mk | `` () | +nan | +nan | `projection` (depth) | +0.880 |  |  |
| ml | `gini` (discrepancy) | -0.067 | -0.405 | `star_discrepancy_shifted` (discrepancy) | +0.405 |  | ✅ |
| mn | `` () | +nan | +nan | `aad` (dispersion) | +0.436 |  |  |
| mr | `iqr` (rel_spread) | +0.279 | +0.472 | `dist_std` (dispersion) | +0.551 |  |  |
| ms | `iqr` (rel_spread) | -0.402 | -0.258 | `dist_std` (dispersion) | -0.258 |  |  |
| mt | `` () | +nan | +nan | `dispersion_shifted` (discrepancy) | +0.556 |  |  |
| multi | `mad` (robust) | +0.081 | -0.084 | `iqr` (rel_spread) | +0.164 |  |  |
| my | `` () | +nan | +nan | `dist_std` (dispersion) | +0.140 |  |  |
| nds | `` () | +nan | +nan | `` () | +nan |  |  |
| ne | `dispersion` (dispersion) | -0.302 | +0.289 | `mpd` (dispersion) | +0.289 |  | ✅ |
| nl | `projection` (depth) | +0.763 | -0.291 | `discrepancy` (discrepancy) | +0.577 |  |  |
| nn | `` () | +nan | +nan | `rel_dispersion` (rel_spread) | -0.475 |  |  |
| no | `star_discrepancy` (discrepancy) | -0.436 | -0.475 | `gini` (discrepancy) | -0.475 |  | ✅ |
| nrm | `` () | +nan | +nan | `` () | +nan |  |  |
| ny | `` () | +nan | +nan | `` () | +nan |  |  |
| or | `` () | +nan | +nan | `gini` (discrepancy) | -0.218 |  |  |
| pa | `` () | +nan | +nan | `rel_star_discrepancy` (discrepancy) | -0.140 |  |  |
| pl | `rel_mpsd` (rel_spread) | +0.672 | -0.469 | `discrepancy` (discrepancy) | +0.577 |  |  |
| ps | `` () | +nan | +nan | `dispersion_shifted` (discrepancy) | +0.712 |  |  |
| pt | `projection` (depth) | +0.130 | -0.296 | `discrepancy` (discrepancy) | +0.597 |  |  |
| ro | `rel_mpd` (rel_spread) | +0.301 | -0.535 | `discrepancy` (discrepancy) | +1.000 |  |  |
| ru | `quartile_deviation` (dispersion) | +0.503 | +0.289 | `gini` (discrepancy) | +0.316 |  |  |
| rw | `` () | +nan | +nan | `` () | +nan |  |  |
| sa | `` () | +nan | +nan | `` () | +nan |  |  |
| sah | `` () | +nan | +nan | `` () | +nan |  |  |
| sd | `` () | +nan | +nan | `dispersion` (dispersion) | +0.447 |  |  |
| se | `` () | +nan | +nan | `` () | +nan |  |  |
| si | `` () | +nan | +nan | `mpsd` (dispersion) | +0.189 |  |  |
| sk | `rel_star_discrepancy` (discrepancy) | +0.442 | -0.572 | `discrepancy` (discrepancy) | +0.378 |  | ✅ |
| sl | `projection` (depth) | +0.451 | -0.307 | `discrepancy` (discrepancy) | +0.378 |  |  |
| sn | `` () | +nan | +nan | `` () | +nan |  |  |
| so | `` () | +nan | +nan | `mpsd` (dispersion) | +0.000 |  |  |
| sq | `dispersion` (dispersion) | -0.258 | -0.000 | `star_discrepancy_shifted` (discrepancy) | +0.000 |  |  |
| sr | `projection` (depth) | +0.201 | +0.310 | `tukey` (depth) | +0.327 |  | ✅ |
| st | `` () | +nan | +nan | `` () | +nan |  |  |
| su | `` () | +nan | +nan | `` () | +nan |  |  |
| sv | `rel_mpsd` (rel_spread) | +0.361 | +0.489 | `dist_std` (dispersion) | +0.546 |  |  |
| sw | `` () | +nan | +nan | `aad` (dispersion) | -0.107 |  |  |
| ta | `projection` (depth) | +0.717 | -0.243 | `discrepancy` (discrepancy) | +0.729 |  |  |
| te | `` () | +nan | +nan | `mpd` (dispersion) | +0.470 |  |  |
| tg | `` () | +nan | +nan | `rel_star_discrepancy` (discrepancy) | -0.081 |  |  |
| th | `rel_mpsd` (rel_spread) | +0.414 | -0.528 | `discrepancy` (discrepancy) | +0.500 |  |  |
| ti | `` () | +nan | +nan | `` () | +nan |  |  |
| tl | `` () | +nan | +nan | `quartile_deviation` (dispersion) | -0.217 |  |  |
| tr | `discrepancy` (discrepancy) | +0.408 | +0.345 | `discrepancy` (discrepancy) | +0.345 | ✅ | ✅ |
| tt | `` () | +nan | +nan | `` () | +nan |  |  |
| ug | `` () | +nan | +nan | `rel_mpsd` (rel_spread) | +0.483 |  |  |
| uk | `discrepancy` (discrepancy) | +0.566 | +0.351 | `discrepancy` (discrepancy) | +0.351 | ✅ | ✅ |
| ur | `projection` (depth) | +0.147 | +0.410 | `dist_std` (dispersion) | +0.873 |  |  |
| uz | `projection` (depth) | +0.378 | +0.197 | `iqr` (rel_spread) | +0.361 |  |  |
| vi | `iqr` (rel_spread) | +0.478 | +0.304 | `mpsd` (dispersion) | +0.411 |  |  |
| xh | `` () | +nan | +nan | `` () | +nan |  |  |
| yo | `` () | +nan | +nan | `` () | +nan |  |  |
| yue | `` () | +nan | +nan | `` () | +nan |  |  |
| zh | `projection` (depth) | +0.706 | +0.739 | `projection` (depth) | +0.739 | ✅ | ✅ |
| zu | `` () | +nan | +nan | `` () | +nan |  |  |
