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
  (`--scheme {A,AT3,B,ZH,ES}` and its own `--output_dir`): the shared
  validation set, the English build, and one FineWeb-2 build per setting that
  scheme defines. The temperature and every token target are derived from the
  grid (92 B where a 1.7B trains, 52 B where the largest rung is 1B), so there
  is no `--temperature` flag. A finished build records its language list and
  temperature in `<prefix>.languages` and is refused if either has since
  changed — give the new scheme its own `--output_dir` instead of overwriting.
- `create_data_mixture.py` — tokenization + Megatron .bin/.idx writer.
