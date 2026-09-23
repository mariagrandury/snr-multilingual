Scripts to generate the multilingual data mixtures. Which data schemes exist,
which settings each one builds and at what sampling temperature all come from
`DATA_SCHEMES` in
`/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/launch_trainings.py`
— nothing here restates them.

- `generate_language_sets.py` — writes `language_sets_scheme{A,B,ZH,ES}.json`
  from `fineweb2-language-distribution.csv` and `configs/tasks.json` (benchmark
  availability); `--check` verifies they are current. Never edit the JSONs by hand.
  AT3 has no file of its own: it is scheme A's lists sampled at T=3.
- `build_data_mixtures.py` — drives `create_data_mixture.py` over one scheme
  (`--scheme {A,AT3,B,ZH,ES,DCLMP,FWEB}` and its own `--output_dir`): the
  shared validation set, the English build, and one FineWeb-2 build per setting
  that scheme defines. A scheme with an `english` corpus in the registry
  (DCLMP, FWEB — the edu-filter axis at L=1) builds its own English instead of
  sharing scheme A's, and records which corpus in `<prefix>.source`. The temperature and every token target are derived from the
  grid (92 B where a 1.7B trains, 52 B where the largest rung is 1B), so there
  is no `--temperature` flag. A finished build records its language list and
  temperature in `<prefix>.languages` and is refused if either has since
  changed — give the new scheme its own `--output_dir` instead of overwriting.
- `create_data_mixture.py` — tokenization + Megatron .bin/.idx writer.
- `build_status.sh` — one line per build: DONE / building / STALLED / pending,
  with the token count as it grows and, for a STALLED chain, the one `sbatch`
  that resumes it. For an English build it also checks the realized size
  against the largest run's draw, which nothing else does (`undersized_build`
  never sees an English build — `fineweb_source` short-circuits at L=1).
  Complements `data_progress.py`, which answers the other question: which
  languages and how many tokens, as a heatmap.
- `refetch_fineweb.sh` — repairs the shared FineWeb mirror in place. Its 2019+
  crawls are an interrupted download (2013–2018 100% intact, 2019–2022 only
  15–38%), which stalls the FWEB English build with `Parquet magic bytes not
  found in footer` and burns its whole chain budget in ~60 s a link.
  `./refetch_fineweb.sh --dry-run` lists what it would fetch; `sbatch
  refetch_fineweb.sh` does it. It only ever overwrites a file that fails to
  open as parquet, re-checks that immediately before an atomic replace, and
  copies the replaced file's mode and ACL onto its replacement — without that
  last step `os.replace` would hand the mirror this user's private staging
  permissions and lock every other `infra01` user out of the repaired files.
