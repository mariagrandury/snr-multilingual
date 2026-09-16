# RQ8 — How early can we make the right decision? (paper RQ2)

## Research question

> For the two planned design decisions — model depth (deep vs shallow) and
> the language lists (scheme A vs B) — how small a proxy, and how early in
> that proxy's run, still reads the decision the way the reference size does
> at its final checkpoint? The paper's RQ2
> ([`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)).

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_all) -->
## Highlighted result

- **depth (deep vs shallow), per-language bits per byte** — final-checkpoint agreement by proxy: 175M 0.00, 350M 0.51; no proxy reaches 0.75.
- **depth (deep vs shallow), benchmark tasks** — final-checkpoint agreement by proxy: 175M 0.49, 350M 0.48, 600M 0.51; no proxy reaches 0.75.
- **language lists (A vs B), per-language bits per byte** — final-checkpoint agreement by proxy: 175M 0.40, 350M 1.00, 600M 0.94, 1B 0.65; smallest proxy at ≥ 0.75: **350M**, which reaches it at 20 % of its run.
- **language lists (A vs B), benchmark tasks** — final-checkpoint agreement by proxy: 175M 0.46, 350M 0.51, 600M 0.45, 1B 0.38; no proxy reaches 0.75.
<!-- END auto:highlight -->

## Experimental setup

Everything comes from rq06's decision table
([`rq06_proxy_predictivity/`](../rq06_proxy_predictivity/)): with two levels,
decision accuracy is the share of population items on which the proxy
prefers the level the reference prefers. The reference is the largest size
trained at both levels, at its final checkpoint; the proxy is every smaller
size, read at the checkpoint nearest 20, 40, 60, 80 and 100 % of its own
run — one grid answers both halves of the question, how small and how
early. Two populations: the per-language BPB of the languages both levels
train, and the benchmark tasks both levels were evaluated on. A cell needs
at least three items.

## Methodology

The per-(intervention, L, population, proxy size, fraction) rows of the two
planned decisions are averaged over L, so that every language setting counts
once. `cells` in the table is the number of settings behind a mean; the
reference each setting resolved against is carried in `refs`.

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from rq06's `predictivity_all` decision table. Regenerate with `python analysis/rq08_early_decision/analyze.py --pool predictivity_all`.

**depth (deep vs shallow) — per-language bits per byte** (rows: proxy size; columns: fraction of the proxy's run; mean over L of the per-L agreement):

| proxy | 20 % | 40 % | 60 % | 80 % | 100 % |
|---|---|---|---|---|---|
| 175M | 0.50 | 0.01 | 0.00 | 0.00 | 0.00 |
| 350M | 0.74 | 0.83 | 0.55 | 0.69 | 0.51 |
| 600M | 1.00 | 1.00 | 1.00 | 0.99 |  |

**depth (deep vs shallow) — benchmark tasks** (rows: proxy size; columns: fraction of the proxy's run; mean over L of the per-L agreement):

| proxy | 20 % | 40 % | 60 % | 80 % | 100 % |
|---|---|---|---|---|---|
| 175M | 0.39 | 0.45 | 0.47 | 0.49 | 0.49 |
| 350M | 0.50 | 0.49 | 0.48 | 0.49 | 0.48 |
| 600M | 0.46 | 0.45 | 0.53 | 0.51 | 0.51 |
| 1.7B | 0.58 | 0.46 | 0.55 | 0.51 |  |

**language lists (A vs B) — per-language bits per byte** (rows: proxy size; columns: fraction of the proxy's run; mean over L of the per-L agreement):

| proxy | 20 % | 40 % | 60 % | 80 % | 100 % |
|---|---|---|---|---|---|
| 175M | 0.86 | 0.68 | 0.40 | 0.40 | 0.40 |
| 350M | 0.88 | 0.94 | 1.00 | 0.97 | 1.00 |
| 600M | 0.94 | 0.94 | 0.87 | 0.88 | 0.94 |
| 1B | 0.87 | 0.75 | 0.75 | 0.58 | 0.65 |
| 1.7B | 1.00 | 1.00 | 1.00 | 0.98 |  |

**language lists (A vs B) — benchmark tasks** (rows: proxy size; columns: fraction of the proxy's run; mean over L of the per-L agreement):

| proxy | 20 % | 40 % | 60 % | 80 % | 100 % |
|---|---|---|---|---|---|
| 175M | 0.46 | 0.50 | 0.47 | 0.45 | 0.46 |
| 350M | 0.48 | 0.48 | 0.46 | 0.49 | 0.51 |
| 600M | 0.50 | 0.49 | 0.49 | 0.49 | 0.45 |
| 1B | 0.46 | 0.44 | 0.53 | 0.53 | 0.38 |
| 1.7B | 0.43 | 0.49 | 0.54 | 0.57 |  |

![Early and small](pretraining/predictivity_all/rq2_early_small.png)
<!-- END auto:results -->

## Files

- `pretraining/<pool>/rq2_decisions.csv` — the rq06 rows of the two planned decisions.
- `…/rq2_early_small.csv` — the mean-over-L agreement per (decision, population, proxy, fraction).
- `…/rq2_early_small.png/.pdf` — the paper figure; `facts.json` the numbers it quotes.
