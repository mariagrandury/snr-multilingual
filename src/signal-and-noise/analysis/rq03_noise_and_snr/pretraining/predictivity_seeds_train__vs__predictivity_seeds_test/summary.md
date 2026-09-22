# Seed-split generalization: `predictivity_seeds_train` → `predictivity_seeds_test`

DA-size here is 175M → 600M (scaling pair); DA-ckpt is the within-size early → final ranking. A language's r needs ≥ 3 distinct tasks with a value (rule 8).

## Headline metrics

|  | DA-size | DA-ckpt |
|---|---:|---:|
| Exact-variant agreement (lang-level) | 0% (0/1) | 0% (0/1) |
| **Family-level agreement** (lang-level) | 100% (1/1) | 100% (1/1) |
| Pearson r between splits (over all variant cells) | -0.677 (n = 22) | +0.881 (n = 22) |
| **Spearman ρ on global variant ranking** | -0.451 | +0.790 |
| Retention of train-best (r_test / r_test_best, mean across langs) | 0% (n = 1) | 99% (n = 1) |

**Family** groups together algebraically near-equivalent variants (e.g. the dispersion cluster: `dispersion`/`mpd`/`range`/`quartile_deviation`/`rms_deviation`/`aad`). At n_mixes=3, members of a family correlate at r ≥ 0.999 so exact-variant equality is overly strict.

**Retention** is how much of the test-split's best r the train-picked variant captures on the test split. 100% = train-pick is also test-best; lower numbers mean we lose predictive correlation by picking on the train split.

## DA-size — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `` () | +nan | +nan | `` () | +nan |  |  |
| az | `` () | +nan | +nan | `` () | +nan |  |  |
| bg | `` () | +nan | +nan | `` () | +nan |  |  |
| bn | `` () | +nan | +nan | `` () | +nan |  |  |
| bs | `` () | +nan | +nan | `` () | +nan |  |  |
| ca | `` () | +nan | +nan | `` () | +nan |  |  |
| cs | `` () | +nan | +nan | `` () | +nan |  |  |
| da | `` () | +nan | +nan | `` () | +nan |  |  |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| el | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `star_discrepancy_shifted` (discrepancy) | +0.148 | -0.299 | `discrepancy` (discrepancy) | +0.129 |  | ✅ |
| es | `` () | +nan | +nan | `` () | +nan |  |  |
| et | `` () | +nan | +nan | `` () | +nan |  |  |
| fa | `` () | +nan | +nan | `` () | +nan |  |  |
| fi | `` () | +nan | +nan | `` () | +nan |  |  |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| he | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `` () | +nan | +nan | `` () | +nan |  |  |
| hr | `` () | +nan | +nan | `` () | +nan |  |  |
| hu | `` () | +nan | +nan | `` () | +nan |  |  |
| id | `` () | +nan | +nan | `` () | +nan |  |  |
| it | `` () | +nan | +nan | `` () | +nan |  |  |
| ja | `` () | +nan | +nan | `` () | +nan |  |  |
| ka | `` () | +nan | +nan | `` () | +nan |  |  |
| kk | `` () | +nan | +nan | `` () | +nan |  |  |
| ko | `` () | +nan | +nan | `` () | +nan |  |  |
| lt | `` () | +nan | +nan | `` () | +nan |  |  |
| lv | `` () | +nan | +nan | `` () | +nan |  |  |
| ml | `` () | +nan | +nan | `` () | +nan |  |  |
| mr | `` () | +nan | +nan | `` () | +nan |  |  |
| ms | `` () | +nan | +nan | `` () | +nan |  |  |
| ne | `` () | +nan | +nan | `` () | +nan |  |  |
| nl | `` () | +nan | +nan | `` () | +nan |  |  |
| no | `` () | +nan | +nan | `` () | +nan |  |  |
| pl | `` () | +nan | +nan | `` () | +nan |  |  |
| pt | `` () | +nan | +nan | `` () | +nan |  |  |
| ro | `` () | +nan | +nan | `` () | +nan |  |  |
| ru | `dispersion_shifted` (discrepancy) | +0.896 | +nan | `` () | +nan |  |  |
| sk | `` () | +nan | +nan | `` () | +nan |  |  |
| sl | `` () | +nan | +nan | `` () | +nan |  |  |
| sq | `` () | +nan | +nan | `` () | +nan |  |  |
| sr | `` () | +nan | +nan | `` () | +nan |  |  |
| sv | `` () | +nan | +nan | `` () | +nan |  |  |
| ta | `` () | +nan | +nan | `` () | +nan |  |  |
| th | `` () | +nan | +nan | `` () | +nan |  |  |
| tr | `` () | +nan | +nan | `` () | +nan |  |  |
| uk | `` () | +nan | +nan | `` () | +nan |  |  |
| ur | `` () | +nan | +nan | `` () | +nan |  |  |
| vi | `` () | +nan | +nan | `` () | +nan |  |  |
| zh | `` () | +nan | +nan | `` () | +nan |  |  |

## DA-ckpt — per language

| lang | train-best (family) | r (train) | r (test) | test-best (family) | r (test) | same variant | same family |
|---|---|---:|---:|---|---:|:---:|:---:|
| ar | `` () | +nan | +nan | `` () | +nan |  |  |
| az | `` () | +nan | +nan | `` () | +nan |  |  |
| bg | `` () | +nan | +nan | `` () | +nan |  |  |
| bn | `` () | +nan | +nan | `` () | +nan |  |  |
| bs | `` () | +nan | +nan | `` () | +nan |  |  |
| ca | `` () | +nan | +nan | `` () | +nan |  |  |
| cs | `` () | +nan | +nan | `` () | +nan |  |  |
| da | `` () | +nan | +nan | `` () | +nan |  |  |
| de | `` () | +nan | +nan | `` () | +nan |  |  |
| el | `` () | +nan | +nan | `` () | +nan |  |  |
| en | `rel_mpsd` (rel_spread) | +0.384 | +0.270 | `rel_dispersion` (rel_spread) | +0.272 |  | ✅ |
| es | `` () | +nan | +nan | `` () | +nan |  |  |
| et | `` () | +nan | +nan | `` () | +nan |  |  |
| fa | `` () | +nan | +nan | `` () | +nan |  |  |
| fi | `` () | +nan | +nan | `` () | +nan |  |  |
| fr | `` () | +nan | +nan | `` () | +nan |  |  |
| he | `` () | +nan | +nan | `` () | +nan |  |  |
| hi | `` () | +nan | +nan | `` () | +nan |  |  |
| hr | `` () | +nan | +nan | `` () | +nan |  |  |
| hu | `` () | +nan | +nan | `` () | +nan |  |  |
| id | `` () | +nan | +nan | `` () | +nan |  |  |
| it | `` () | +nan | +nan | `` () | +nan |  |  |
| ja | `` () | +nan | +nan | `` () | +nan |  |  |
| ka | `` () | +nan | +nan | `` () | +nan |  |  |
| kk | `` () | +nan | +nan | `` () | +nan |  |  |
| ko | `` () | +nan | +nan | `` () | +nan |  |  |
| lt | `` () | +nan | +nan | `` () | +nan |  |  |
| lv | `` () | +nan | +nan | `` () | +nan |  |  |
| ml | `` () | +nan | +nan | `` () | +nan |  |  |
| mr | `` () | +nan | +nan | `` () | +nan |  |  |
| ms | `` () | +nan | +nan | `` () | +nan |  |  |
| ne | `` () | +nan | +nan | `` () | +nan |  |  |
| nl | `` () | +nan | +nan | `` () | +nan |  |  |
| no | `` () | +nan | +nan | `` () | +nan |  |  |
| pl | `` () | +nan | +nan | `` () | +nan |  |  |
| pt | `` () | +nan | +nan | `` () | +nan |  |  |
| ro | `` () | +nan | +nan | `` () | +nan |  |  |
| ru | `dist_std` (dispersion) | +0.413 | +nan | `` () | +nan |  |  |
| sk | `` () | +nan | +nan | `` () | +nan |  |  |
| sl | `` () | +nan | +nan | `` () | +nan |  |  |
| sq | `` () | +nan | +nan | `` () | +nan |  |  |
| sr | `` () | +nan | +nan | `` () | +nan |  |  |
| sv | `` () | +nan | +nan | `` () | +nan |  |  |
| ta | `` () | +nan | +nan | `` () | +nan |  |  |
| th | `` () | +nan | +nan | `` () | +nan |  |  |
| tr | `` () | +nan | +nan | `` () | +nan |  |  |
| uk | `` () | +nan | +nan | `` () | +nan |  |  |
| ur | `` () | +nan | +nan | `` () | +nan |  |  |
| vi | `` () | +nan | +nan | `` () | +nan |  |  |
| zh | `` () | +nan | +nan | `` () | +nan |  |  |
