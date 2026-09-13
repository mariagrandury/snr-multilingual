Scripts to generate the multilingual data mixtures.

- `generate_language_sets.py` — writes `language_sets_scheme{A,B}.json` from
  `fineweb2-language-distribution.csv` and `configs/tasks.json` (benchmark
  availability); `--check` verifies they are current. Never edit the JSONs by hand.
- `build_data_mixtures.py` — drives `create_data_mixture.py` over the sweep
  (validation set, English build, one FineWeb-2 build per language setting);
  a finished build records its language list in `<prefix>.languages` and is
  refused if the list has since changed.
- `create_data_mixture.py` — tokenization + Megatron .bin/.idx writer.
