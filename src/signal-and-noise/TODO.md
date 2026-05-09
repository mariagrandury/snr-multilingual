Fix bugs:

- mgsm_direct: all 157 rows have primary_score=NaN because the
  parquet was generated with primary_metric=acc but the actual
  metric is exact_match. Fix belongs in swissai-evals-post-train's
  parquet generator.
- Relaxed inner-join in snr_for_subset, English truthfulqa_mc1
  silently excluded as a singleton family, and per-sample SNR
  operating on binary acc — all by design but worth a louder
  callout for readers.

- mmlu vs global mmlu in AllenAI comparison:
  \_load_apertus_with_alias may merge methodologically distinct tasks
  results/allenai_comparison/analyze.py:73-89

  global*mmlu_full_en*<subject> is Cohere Full (MT + post-edit), while AllenAI's mmlu*<subject> is the original MMLU. The dedup keeps whichever has more non-NaN columns, which will reliably be the global_mmlu row. The cross-corpus correlation then partially measures English-vs-Cohere-Full content drift, not purely SNR transfer. The docstring acknowledges the alias but underplays that the content differs slightly. Not a code bug, but a methodological footgun. Fix: rename the row to mmlu_via_global_mmlu_full_en*<subject> and keep both, or surface the discrepancy in the agreement.md output.
