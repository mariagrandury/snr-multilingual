
## Handoff: changing the validation languages

Worth knowing first: the validation set is already out of date. It covers 8 languages the current L100 list dropped (bew div epo gmh hif ltz nrm tat) and lacks 8 it added (fao hat ibo jav kin sot xho zul). The next --stage validation run rebuilds it automatically, even before your L100 change.

Tasks

Change FW_L100, run generate_language_sets.py, then git diff: only FW_L100 may change. The script rewrites every JSON, and scheme B and AT3 reuse scheme A's L100 list.
Add any new codes to configs/languages.json (fineweb_iso2), and check each new language exists in the filtered FineWeb directory.
Save a copy of the current validation.manifest.json.
Wait until no data build is running or queued. Three build-*-92b jobs are running now. The manifest is shared by every scheme through symlinks and rewritten in place.
Rebuild in the data root: build_data_mixtures.py --scheme A --output_dir <root> --stage validation.
Check every language that stays: first_file, val_doc_count, tokens and bytes must be identical to the old manifest. If any differ, already-trained cells may have trained on the new validation rows.
Check tokenizer round-trips on the new languages (see 4).
Rebuild the L100 training mix. The one on disk (21 Aug) uses the old list, and no L100 cell has trained yet.
Copy the new validation .bin/.idx files and the manifest to /iopsstor/.../data by hand; stage_to_iopsstor.sh skips validation files.
Decide how existing checkpoints get BPB for the new languages: re-score everything, or add a "score missing languages and merge" mode. Then change the "already done" check to "all manifest languages present".

Remember

Existing checkpoints never get re-scored. Any non-empty bpb.json counts as done, so the 2,184 existing files would never gain the new languages.
One missing file breaks the whole job. If the manifest lists a language whose .bin isn't staged, the scoring job crashes and writes no bpb.json at all.
macro_bpb changes meaning with the language set. Compare old and new only on shared languages.
Small languages get less validation text. The 30% cap can leave a low-resource language below 1M tokens, so its BPB is noisier.
Decide on purpose whether to keep a few never-trained languages as zero-shot controls.

## The byte-count self-check

BPB divides by UTF-8 bytes, and score_bpb.py computes those bytes by decoding the tokens back to text. The guard at score_bpb.py:199 compares that decoded count with the manifest, but only when a whole language is scored. Production scores a 1M-token prefix of roughly 5M, so the guard never runs and nothing else checks the bytes.

Why it matters. If decoding stops being lossless for some script, that language's BPB is silently scaled by a wrong factor. Causes could be a transformers or tokenizers upgrade in the eval container, Unicode normalisation, space clean-up, or a different tokenizer in a converted checkpoint. Comparing models within that language would still work, but absolute values and cross-language comparisons would be wrong. My 20-script test passed, but it ran in the snr environment (transformers 5.5.3), not in the eval container (eval-vllm-26-02-17.sqsh).

Fix. Once per scoring job, decode each language's full validation text and compare with the manifest. That costs about 1 s per language, roughly 100 s against 450–2,100 s per checkpoint, and the job should stop on any mismatch. When rebuilding validation (task 3), also store byte counts at document boundaries so each 1M-token prefix can be checked exactly.
