#!/usr/bin/env python3
"""Rewrite the three letter-format families with Gemini into a statement stem
plus four short continuations — the `rfgm_*` twins (Tier 2 of
src/signal-and-noise/analysis/rq00_task_reformulation/README.md).

Tier 1 (`rf_*`) only dropped the letters. Here the item itself changes: the
question becomes one declarative sentence that stops where the answer goes,
the options become short parallel continuations, in the item's own language.
The rewritten sets are read by the harness through `dataset_path: json`
YAMLs (make_rf_tasks.py --set rfgm), and are never pushed publicly: they
carry the gold labels.

One JSONL per task lands in DATA_DIR (`<task>.jsonl`, the row schema below);
the Batch API bookkeeping lives next to it (`_requests/`, `_jobs.json`,
`_rejects/`). Every mode is idempotent and resumable — re-run after any
interruption, nothing is redone that is already on disk:

    pilot   [--n 10] [--tasks a,b]   synchronous calls on a few items of the REVIEW_TASKS, printed side by side
    build   [--family F] [--tasks a,b] [--dry-run]   the batch request files + the cost estimate
    submit  [--max-jobs N] [--retry]   upload + create one batch job per task
    status                       refresh and print every job's state
    fetch                        download SUCCEEDED jobs, validate, write <task>.jsonl
    report  [--show]             counts, reject rates, lengths, spent tokens; sample items of the REVIEW_TASKS

Runs on the login node (outbound internet, the snr env, the offline datasets
cache), against either Gemini backend — the SDK picks one from the
environment and so does this script:

  Vertex AI (what we use): Application Default Credentials, no API key.
      GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION.
      Its batch jobs read and write Cloud Storage — an uploaded file is not a
      valid source — so `submit` puts each task's requests in
      gs://<bucket>/rfgm/requests/ and `fetch` reads the job's destination
      back. The bucket is $RFGM_GCS_BUCKET or <project>-msnr-rfgm, created on
      first use, and the same ADC credential signs the storage calls.
  Gemini Developer API: GEMINI_API_KEY, the Files API, no bucket.

Row schema of <task>.jsonl — `text`, `choices`, `gold` are what the YAML reads:

    {"id": 17, "text": "<passage>\\n<stem>", "choices": [4 strings], "gold": 2,
     "stem": …, "context": "<passage or ''>", "subject": …, "orig_question": …,
     "orig_choices": […], "lang": "deu_Latn"}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_rf_tasks import HARNESS, TASKS_JSON, TEMPLATES, source_config  # noqa: E402

DATA_DIR = Path("/capstor/store/cscs/swissai/infra01/msnr/msnr-harness/rf-data/rfgm")
PILOT_DIR = Path(__file__).resolve().parents[3] / "src" / "signal-and-noise" / "analysis" / "rq00_task_reformulation"
MODEL = "gemini-3.8-flash"
# Vertex AI (ADC) rather than an API key; its batch jobs are Cloud Storage in
# and out. MODEL has to run on GOOGLE_CLOUD_LOCATION=global: the whole Gemini
# 3.x family is served only from the global endpoint on this project, and a
# regional job answers 404 "The PublisherModel does not exist" (verified
# 2026-09-20 against us-central1, europe-west4 and us-east5; 2.5 is regional).
# Batch prediction does accept global, and a US multi-region bucket serves it.
VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")
GCS_API = "https://storage.googleapis.com/storage/v1/b"
GCS_UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b"
BUCKET_LOCATION = os.environ.get("RFGM_GCS_LOCATION", "US")
PRICE_IN, PRICE_OUT = 0.375, 1.875     # $/Mtok, Batch API, through 2026-12-31 (doubles in 2027)
TOKENS_PER_BYTE = 1 / 3.5               # the README's estimate (±30 %, CJK/Indic heavier)

SYSTEM = """You rewrite multiple-choice test items so that small language models can be
evaluated by scoring candidate text continuations.

Use exactly the SAME LANGUAGE and script as the input. Never translate the
item or switch to English.

Rewrite the question as ONE declarative sentence with a missing final
constituent. Return the part before that constituent as the "stem". Rewrite
each of the four "choices" as a short continuation that completes the stem.

Requirements:
- The stem and every continuation choice must combine into a grammatical,
  natural-sounding statement in the input language.
- Preserve the meaning of the question and all four choices.
- Preserve the order of the choices and which choice is correct. Each
  completed statement must retain the truth value implied by its original
  choice.
- The stem must not reveal or hint at the correct answer.
- Do not use question wording such as the input-language equivalent of
  "which of the following".
- Make the four continuations syntactically parallel and similar in length.
  Keep them concise, preferably 1-8 words, unless additional words are
  necessary to preserve meaning or grammaticality.
- You may make minimal grammatical changes to an option (such as changing
  capitalization, inflection, agreement, or function words) but must not
  change its meaning or add factual content.
- For negatively framed questions, such as questions asking which option is
  NOT true, preserve the negation explicitly in the stem using natural
  wording in the input language.
- The stem must end exactly at the shared completion boundary, with no
  trailing whitespace or terminal punctuation.
- The stem and the continuation are joined with exactly one space, which you
  do not control. Write each continuation as it should read after that
  space, with the capitalization the input language requires.
- Include terminal punctuation in each continuation when the completed
  statement requires it.
- Do not include option letters or numbers in the continuations.
- Never add an intensifier, comparative or qualifier an option did not have.
- Reproduce any quoted passage, list or scenario in the question verbatim at
  the start of the stem; never summarise it away.
- The stem must carry the whole proposition but its last constituent; never
  just a framing phrase, and never ending at a comma.
- If the blank is mid-sentence, restructure so the missing part comes last.
- The stem must not end with a word that agrees in gender, number or case
  with only some options; move it into each continuation.
- Do not add quotation marks or brackets the original lacked, and keep the
  original numeral system.
{family_note}

Return only valid JSON, with no Markdown fence, explanation, or additional
keys:
{{"stem":"...","choices":["...","...","...","..."]}}"""
NOTES = {
    "belebele": "A passage is provided as context. The completed statements must be answerable\n"
                "from that passage alone. Do not summarize, rewrite, or quote the passage\n"
                "in the stem or choices.",
    "global_mmlu_full": "A subject label is provided as metadata. Mention the subject in the stem only\n"
                        "when needed for clarity; do not add subject information that makes the answer\n"
                        "easier or changes the item's meaning.",
    "include_base_44": "A subject label may be provided as metadata. The item may depend on knowledge\n"
                       "specific to a region, such as local driving rules, laws, or history.\n"
                       "Preserve the original terms.",
}
SCHEMA = {"type": "OBJECT",
          "properties": {"stem": {"type": "STRING"},
                         "choices": {"type": "ARRAY", "items": {"type": "STRING"},
                                     "minItems": 4, "maxItems": 4}},
          "required": ["stem", "choices"]}

# The human reviews (pilot, report --show): Spanish, Hindi, Turkish, Farsi,
# Greek, Arabic, Basque, Chinese in every family that has them — scripts and
# resource levels the script check cannot tell apart.
REVIEW_TASKS = [f"belebele_{c}" for c in ("spa_Latn", "hin_Deva", "tur_Latn", "pes_Arab", "ell_Grek",
                                          "arb_Arab", "eus_Latn", "zho_Hans")] \
    + [f"global_mmlu_full_{c}" for c in ("es", "hi", "tr", "fa", "el", "ar", "zh")] \
    + [f"include_base_44_{c}" for c in ("spanish", "hindi", "turkish", "persian", "greek", "arabic",
                                        "basque", "chinese")]

# family -> (option fields, gold index from the row, passage field, subject field)
FIELDS = {
    "belebele": (("mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"),
                 lambda r: int(r["correct_answer_num"]) - 1, "flores_passage", None),
    "global_mmlu_full": (("option_a", "option_b", "option_c", "option_d"),
                         lambda r: "ABCD".index(r["answer"]), None, "subject"),
    "include_base_44": (("option_a", "option_b", "option_c", "option_d"),
                        lambda r: int(r["answer"]), None, "subject"),
}


def originals() -> list[tuple[str, str]]:
    """(task, family) of every original task the rf twins were built for."""
    tasks = json.loads(TASKS_JSON.read_text())["tasks"]
    return sorted((n, e["benchmark"]) for n, e in tasks.items()
                  if e["benchmark"] in TEMPLATES and "pretraining" in e["stages"]
                  and e["language"] not in ("multi", "??"))


# An option that points at other options ("b and c are correct", "all of the
# above") cannot survive the rewrite: the harness scores stem + " " + choice
# one choice at a time, so the antecedent is never on screen. 6 of the 230
# pilot items carry one. They are flagged rather than dropped, because the
# originals and the rf_ twins keep them and dropping them here alone would
# score the three sets on different item sets; `--drop-self-ref` drops them
# once that decision is made. The word list covers the eight review
# languages and English only — the letter pattern is language-agnostic and
# does the rest.
LETTER_REF = re.compile(r"[(\[]\s*[a-dA-D]\s*[)\]]|(?<![\w)])[a-dA-D]\s*\)")
# an option that IS a list of option labels: "a, b", "b, c, d", "II", "IV"
_TOK = r"(?:[a-dA-D]|[ivxIVX]{1,4})"
LETTER_LIST = re.compile(rf"^{_TOK}(?:\s*(?:,|/|&|and|y|und|e)\s*{_TOK})*$")
ABOVE_REF = ("all of the above", "none of the above", "both of the above", "all of these",
             "todas las anteriores", "ninguna de las anteriores", "ambos", "ambas",
             "उपरोक्त सभी", "इनमें से कोई नहीं", "yukarıdakilerin hepsi", "hiçbiri",
             "همه موارد", "هیچ‌کدام", "όλα τα παραπάνω", "κανένα από τα παραπάνω",
             "جميع ما ذكر", "كل ما سبق", "لا شيء مما سبق", "aurreko guztiak", "zuzenak dira",
             "以上都是", "以上都不是", "以上皆是")


def self_ref(options: list[str]) -> bool:
    return any(LETTER_REF.search(o) or LETTER_LIST.match(o.strip().rstrip("."))
               or any(w in o.casefold() for w in ABOVE_REF) for o in options)


def items(task: str, fam: str) -> list[dict]:
    """The task's items from the offline datasets cache, rows with a missing
    option dropped (the rf twins' process_docs rule)."""
    import datasets
    opts, gold, passage, subject = FIELDS[fam]
    cfg = source_config(task, fam, HARNESS)
    ds = datasets.load_dataset(cfg["dataset_path"], cfg.get("dataset_name"), split=cfg["test_split"])
    out = []
    for i, r in enumerate(ds):
        vals = [r[k] for k in ("question",) + opts]
        if not all(isinstance(v, str) and v.strip() for v in vals):
            continue
        options = [r[k].strip() for k in opts]
        out.append({"id": i, "question": r["question"].strip(), "options": options,
                    "gold": gold(r), "passage": (r[passage] or "").strip() if passage else "",
                    "subject": (r[subject] or "").replace("_", " ") if subject else "",
                    "lang": task[len(fam) + 1:], "self_ref": self_ref(options),
                    "dup_opts": len({o.casefold() for o in options}) < 4})
    return out


def user_content(it: dict) -> str:
    lines = [f"Language: {it['lang']}"]
    if it["subject"]:
        lines.append(f"Subject: {it['subject']}")
    if it["passage"]:
        lines.append(f"Passage: {it['passage']}")
    lines += [f"Question: {it['question']}", "Options:"]
    lines += [f"{i + 1}. {o}" for i, o in enumerate(it["options"])]
    return "\n".join(lines)


def request(task: str, fam: str, it: dict) -> dict:
    """One Batch API line: REST JSON, the shape the SDK writes for inline
    requests. The Gemini API returns the `key` with the answer; Vertex ignores
    every field outside `request` and echoes the request instead, so there the
    line carries no key and `fetch` matches on the prompt text."""
    body = {"contents": [{"role": "user", "parts": [{"text": user_content(it)}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM.format(family_note=NOTES[fam])}]},
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseSchema": SCHEMA,
                                 "thinkingConfig": {"thinkingLevel": "LOW"}}}
    return {"request": body} if VERTEX else {"key": f"{task}:{it['id']}", "request": body}


def tokens(text: str) -> float:
    return len(text.encode()) * TOKENS_PER_BYTE


def scaffold(fam: str) -> float:
    """The instruction and the response schema: identical on every request of
    a family and billed on every one of them. Measure it rather than assume
    it — at ~700 tokens against a ~350-token Global-MMLU item it is the
    larger half of the input bill, and the 150 this used to assume put the
    estimate out by a third. (No cache discount to expect: implicit caching
    needs a 2 048-token shared prefix on Vertex and this is a quarter of
    that.)"""
    # bytes/3.5 reads ~30 % high on English prose: count_tokens says 550 for
    # this prompt where this returns ~719 (measured 2026-09-20).
    return tokens(SYSTEM.format(family_note=NOTES[fam])) + tokens(json.dumps(SCHEMA))


def estimate(its: list[dict], fam: str) -> tuple[float, float]:
    """(input, output) tokens of a task: the item plus the scaffold in, the
    question and options (not the passage) out."""
    s = scaffold(fam)
    t_in = sum(tokens(user_content(it)) + s for it in its)
    t_out = sum(tokens(it["question"]) + sum(map(tokens, it["options"])) + 30 for it in its)
    return t_in, t_out


# ---------------------------------------------------------------- validation
# Japanese mixes kanji and kana and Korean mixes hangul and hanja WITHIN a
# sentence, so the majority script of a clause flips between them for reasons
# that have nothing to do with language drift. Folding them into one class is
# what makes a per-choice check usable: without it the check rejected 155 of
# 501 Japanese INCLUDE items (30.9 %) against 0-2 % everywhere else, which is
# exactly the per-language item loss this comparison cannot absorb.
SCRIPT_FOLD = {"HIRAGANA": "CJK", "KATAKANA": "CJK", "HANGUL": "CJK", "IDEOGRAPHIC": "CJK"}


def script_of(text: str) -> str:
    """Majority Unicode script of the letters in `text` (LATIN, CYRILLIC, CJK,
    ARABIC, …) — a cheap 'still in the right language' check."""
    c = Counter(SCRIPT_FOLD.get(s, s) for s in
                (unicodedata.name(ch, "?").split()[0] for ch in text if ch.isalpha()))
    return c.most_common(1)[0][0] if c else ""


STEM_TAIL = ":\uff1a;-\u2013\u2014\u2026"


def normalize(out: dict) -> dict:
    """Trim the joiner punctuation off the stem rather than rejecting the
    rewrite for it. Five of the six pilot rejects were a trailing colon, and
    they were not spread evenly — Basque 3/10 and Persian 2/20 against 0 of
    180 in the other six languages, because both languages front a
    cataphoric "the following is:". Rejecting them would delete more items
    in some languages than others, which is the one bias a per-language
    comparison cannot absorb."""
    s = out.get("stem")
    if not isinstance(s, str):
        return out
    prev = None
    while prev != s:
        prev = s
        s = s.rstrip()
        if s.endswith("..."):
            s = s[:-3]
        s = s.rstrip(STEM_TAIL)
    return {**out, "stem": s}


def validate(it: dict, out: dict) -> str | None:
    """The reason an answer is rejected, or None. Call `normalize` first."""
    stem, choices = out.get("stem"), out.get("choices")
    if not isinstance(stem, str) or not stem.strip():
        return "empty stem"
    if stem != stem.rstrip() or stem.rstrip().endswith(":"):
        return "stem ends with whitespace or a colon"
    if not isinstance(choices, list) or len(choices) != 4:
        return "not four choices"
    if not all(isinstance(c, str) and c.strip() for c in choices):
        return "empty choice"
    if len({c.strip().casefold() for c in choices}) < 4:
        return "duplicate choices"
    low = stem.casefold()
    if any(re.search(rf"(?<!\w){re.escape(c.strip().casefold())}(?!\w)", low) for c in choices):
        return "a choice appears in the stem"
    if script_of(stem + " ".join(choices)) != script_of(it["question"] + " ".join(it["options"])):
        return "script changed"
    # per choice as well: the majority over the whole item was blind to
    # include_base_44_greek#404, whose four Greek numerals came back in Latin.
    # Only for options the source wrote in the item's own script: an option
    # that was already off-script ("a, b", a Latin formula, a bare numeral)
    # is not evidence of drift, and comparing it produced most of the false
    # rejects in the first INCLUDE run.
    ref = script_of(it["question"] + " ".join(it["options"]))
    for c, o in zip(choices, it["options"]):
        so = script_of(o)
        if so != ref:
            continue
        sc = script_of(c)
        if sc and sc != so:
            return "a choice changed script"
    return None


def row(it: dict, out: dict) -> dict:
    stem, choices = out["stem"].strip(), [c.strip() for c in out["choices"]]
    return {"id": it["id"], "text": (it["passage"] + "\n" if it["passage"] else "") + stem,
            "choices": choices, "gold": it["gold"], "stem": stem, "context": it["passage"],
            "subject": it["subject"], "orig_question": it["question"], "orig_choices": it["options"],
            "lang": it["lang"], "self_ref": it.get("self_ref", False)}


# ---------------------------------------------------------------- state
def load_state(d: Path) -> dict:
    p = d / "_jobs.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(d: Path, state: dict) -> None:
    (d / "_jobs.json").write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")


def client():
    from google import genai
    if not VERTEX and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        # Without GOOGLE_GENAI_USE_VERTEXAI the SDK takes the API-key route and
        # dies several frames down with "No API key was provided", which reads
        # like a credentials problem rather than a missing export. Two launches
        # were lost to it; say which variable is missing.
        sys.exit("GOOGLE_GENAI_USE_VERTEXAI is not set, so this would call the Gemini API with an API "
                 "key rather than Vertex with ADC. Export it with the other two:\n"
                 "  export GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=silin-482809 "
                 "GOOGLE_CLOUD_LOCATION=global")
    if VERTEX:
        # With no project/location the SDK reads a GEMINI_API_KEY left in the
        # shell and calls Vertex in express mode with it, instead of ADC — the
        # same invalid-key error, one layer down. With both set it ignores the
        # key (_api_client: "implicit project/location takes precedence").
        if not (os.environ.get("GOOGLE_CLOUD_PROJECT") and os.environ.get("GOOGLE_CLOUD_LOCATION")):
            sys.exit("Vertex needs GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION; without them the SDK "
                     "falls back to a GEMINI_API_KEY left in the shell rather than to ADC.")
        if os.environ["GOOGLE_CLOUD_LOCATION"] != "global":
            sys.exit(f"GOOGLE_CLOUD_LOCATION={os.environ['GOOGLE_CLOUD_LOCATION']!r}: {MODEL} is served only "
                     "from the global endpoint, and a regional job fails with 404 'The PublisherModel does "
                     "not exist'. Set GOOGLE_CLOUD_LOCATION=global.")
    c = genai.Client()
    if c.vertexai != VERTEX:
        sys.exit(f"the SDK resolved {'Vertex' if c.vertexai else 'the Gemini API'} but this script was "
                 f"configured for {'Vertex' if VERTEX else 'the Gemini API'}: the two write different "
                 "request files. Set GOOGLE_GENAI_USE_VERTEXAI (with GOOGLE_CLOUD_PROJECT and "
                 "GOOGLE_CLOUD_LOCATION) or GEMINI_API_KEY consistently, and rebuild.")
    return c


# ---------------------------------------------------------------- Cloud Storage (Vertex only)
def gcs(bucket: str | None = None):
    """(session, bucket) for the batch objects, the bucket created on first
    use. Signed with the ADC credential the Vertex client already uses, so
    there is nothing more to configure and no gcloud at run time."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    creds, adc_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
    bucket = bucket or os.environ.get("RFGM_GCS_BUCKET") or f"{project}-msnr-rfgm"
    s = AuthorizedSession(creds)
    if s.get(f"{GCS_API}/{bucket}").status_code == 404:
        r = s.post(GCS_API, params={"project": project},
                   json={"name": bucket, "location": BUCKET_LOCATION})
        if r.status_code == 403:
            sys.exit(f"gs://{bucket} does not exist and this account cannot create it:\n  "
                     + r.json().get("error", {}).get("message", r.text)[:300]
                     + f"\nVertex batch reads and writes Cloud Storage, so a bucket is required. Ask an "
                     f"admin on project {project} either to grant roles/storage.admin (or "
                     f"roles/storage.bucketCreator + roles/storage.objectAdmin), or to create "
                     f"gs://{bucket} in {BUCKET_LOCATION} and grant you object access. Point the script at "
                     "an existing bucket with RFGM_GCS_BUCKET.")
        r.raise_for_status()
        print(f"created gs://{bucket} in {BUCKET_LOCATION}")
    return s, bucket


def gcs_put(s, bucket: str, obj: str, path: Path) -> str:
    r = s.post(f"{GCS_UPLOAD}/{bucket}/o", params={"uploadType": "media", "name": obj},
               data=path.read_bytes(), headers={"Content-Type": "application/x-ndjson"})
    r.raise_for_status()
    return f"gs://{bucket}/{obj}"


def gcs_read(s, uri: str) -> bytes:
    """Every .jsonl object under a gs:// prefix, concatenated — a batch job's
    destination is a directory whose file names the service chooses."""
    bucket, prefix = uri[len("gs://"):].split("/", 1)
    out, token = b"", None
    while True:
        r = s.get(f"{GCS_API}/{bucket}/o", params={"prefix": prefix.rstrip("/") + "/", "pageToken": token})
        r.raise_for_status()
        page = r.json()
        for o in page.get("items", []):
            if o["name"].endswith(".jsonl"):
                d = s.get(f"{GCS_API}/{bucket}/o/{quote(o['name'], safe='')}", params={"alt": "media"})
                d.raise_for_status()
                out += d.content
        token = page.get("nextPageToken")
        if not token:
            return out


def cell_tasks(L: int, scheme: str) -> set[str]:
    """Original task names in the languages an L-cell trains on.

    The ladder only ever scores a task on models that trained its language
    (analysis rule 2, utils.trained_only), so a twin outside this set is
    paid for and never read: L50 keeps 124 of the 185 tasks, and only 59 of
    belebele's 105. The rq00 gate's own task counts (n=59 / 29 / 36) are
    this set."""
    pretrain = str(Path(__file__).resolve().parents[2] / "pretrain")
    if pretrain not in sys.path:
        sys.path.insert(0, pretrain)
    from launch_trainings import cell_languages          # noqa: E402
    from utils.configs import tasks_for_benchmarks       # noqa: E402
    return set(tasks_for_benchmarks(list(TEMPLATES), cell_languages(L, scheme), path=TASKS_JSON))


def selected(args) -> list[tuple[str, str]]:
    want = set(args.tasks.split(",")) if args.tasks else None
    keep = cell_tasks(args.L, args.scheme) if getattr(args, "L", None) else None
    out = [(t, f) for t, f in originals()
           if (not args.family or f == args.family) and (want is None or t in want)
           and (keep is None or t in keep)]
    if keep is not None:
        print(f"L{args.L} scheme {args.scheme}: {len(out)} tasks selected")
    return out


def retrying(fn, tries: int = 7, base: float = 5.0):
    """Gemini 3.x online serving runs on Vertex's dynamic shared quota, which
    answers 429 RESOURCE_EXHAUSTED under any sustained sequential use and is
    not something a quota increase fixes; the SDK's own retry does not cover
    it. Back off and keep the run alive — the pilot is minutes of traffic, and
    the 636k-item run goes through batch, which has its own queue."""
    from google.genai import errors
    for i in range(tries):
        try:
            return fn()
        except errors.ClientError as e:
            if getattr(e, "code", None) != 429 or i == tries - 1:
                raise
            wait = base * 2 ** i + random.random()
            print(f"  429 (shared quota); retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------- modes
def pilot(args) -> None:
    from google.genai import types
    c = client()
    tasks = selected(args) if args.tasks or args.family else [(t, f) for t, f in originals() if t in REVIEW_TASKS]
    for task, fam in tasks:
        out_path = (args.data_dir / f"pilot_{task}.jsonl" if args.data_dir != DATA_DIR
                    else PILOT_DIR / f"pilot_{task}.jsonl")
        if out_path.exists() and not args.force:
            print(f"{task}: already written, skipping (--force to redo)")
            continue
        its = items(task, fam)
        random.Random(0).shuffle(its)
        rows, rejects = [], []
        for it in its[:args.n]:
            r = retrying(lambda: c.models.generate_content(
                model=MODEL, contents=user_content(it),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM.format(family_note=NOTES[fam]),
                    response_mime_type="application/json", response_schema=SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_level="LOW"))))
            out = normalize(json.loads(r.text))
            why = validate(it, out)
            print(f"\n[{task} #{it['id']}] {'REJECT: ' + why if why else 'ok'}")
            print(f"  Q: {it['question']}\n     {it['options']}  gold={it['gold']}")
            print(f"  S: {out.get('stem')}\n     {out.get('choices')}")
            (rejects if why else rows).append({**row(it, out), "reject": why} if why else row(it, out))
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows + rejects))
        print(f"\n{task}: {len(rows)} accepted, {len(rejects)} rejected -> {out_path}")


def online(args) -> None:
    """Rewrite through the online endpoint instead of Batch: no Cloud
    Storage, so it runs with nothing but ADC, at double the batch price.

    Resumable per item — every answer is appended and flushed as it lands,
    and a re-run skips the ids already in the output or the reject file, so
    a killed login-node process loses at most the requests in flight.
    Concurrency is capped because Gemini 3.x online runs on Vertex's dynamic
    shared quota: 12 threads sustain ~2.9 items/s, while 24 trips 429s and
    collapses throughput to 0.3 (measured 2026-09-20)."""
    from google.genai import types
    c = client()
    for task, fam in selected(args):
        out_p = args.data_dir / f"{task}.jsonl"
        rej_p = args.data_dir / "_rejects" / f"{task}.jsonl"
        rej_p.parent.mkdir(parents=True, exist_ok=True)
        done = {r["id"] for q in (out_p, rej_p) if q.exists()
                for r in map(json.loads, q.open(encoding="utf-8"))}
        if args.retry_rejects and rej_p.exists():
            # Re-run everything a previous pass rejected, and clear the file so
            # this pass's failures are the record. An item lost to a crash in
            # between is in neither file and is simply picked up next run.
            back = [json.loads(l) for l in rej_p.open(encoding="utf-8")]
            done -= {r["id"] for r in back}
            rej_p.write_text("")
            print(f"{task}: retrying {len(back)} previously rejected items")
        its = [it for it in items(task, fam)
               if it["id"] not in done and not (args.drop_self_ref and it["self_ref"])]
        if args.limit:
            its = its[:args.limit]
        if not its:
            print(f"{task}: nothing to do ({len(done)} items already written)")
            continue
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM.format(family_note=NOTES[fam]),
            response_mime_type="application/json", response_schema=SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"))

        def one(it: dict) -> tuple[dict | None, int, int]:
            # four distinct choices cannot come out of fewer than four
            # distinct options: reject before paying for the call
            if it["dup_opts"]:
                return {**it, "reject": "duplicate source options"}, 0, 0
            try:
                r = retrying(lambda: c.models.generate_content(
                    model=MODEL, contents=user_content(it), config=cfg))
            except Exception:                           # noqa: BLE001 - one bad item must not end the task
                # None, not a reject row: an item that outlived the backoff
                # failed on the API, not on its content, and writing it to
                # the reject file would put its id in `done` and drop it for
                # good. Left unwritten, the next run simply picks it up.
                return None, 0, 0
            u = r.usage_metadata
            t_in = u.prompt_token_count or 0
            t_out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
            try:
                out = normalize(json.loads(r.text))
            except (json.JSONDecodeError, TypeError):
                return {**it, "reject": "unparseable JSON"}, t_in, t_out
            why = validate(it, out)
            # keep what the model produced on a reject too: without it a
            # rejected item cannot be diagnosed or re-judged without paying
            # for the call again.
            rec = ({**it, "reject": why, "stem": out.get("stem"), "choices": out.get("choices")}
                   if why else row(it, out))
            return rec, t_in, t_out

        usage, n_ok, n_rej, n_api, t0 = Counter(), 0, 0, 0, time.time()
        with out_p.open("a", encoding="utf-8") as fo, rej_p.open("a", encoding="utf-8") as fr, \
                ThreadPoolExecutor(args.concurrency) as ex:
            for rec, t_in, t_out in ex.map(one, its):   # main thread writes: no lock needed
                usage.update(t_in=t_in, t_out=t_out)
                if rec is None:
                    n_api += 1
                    continue
                f = fr if "reject" in rec else fo
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_rej += "reject" in rec
                n_ok += "reject" not in rec
                if (n_ok + n_rej) % 200 == 0:
                    rate = (n_ok + n_rej + n_api) / (time.time() - t0)
                    left = (len(its) - n_ok - n_rej - n_api) / rate / 3600
                    print(f"  {task}: {n_ok + n_rej}/{len(its)}  {rate:.1f}/s  ~{left:.1f} h left  "
                          f"${2 * (usage['t_in'] * PRICE_IN + usage['t_out'] * PRICE_OUT) / 1e6:.2f}", flush=True)
        cost = 2 * (usage["t_in"] * PRICE_IN + usage["t_out"] * PRICE_OUT) / 1e6   # online = 2x batch
        print(f"{task}: {n_ok} rows, {n_rej} rejected, {(time.time() - t0) / 60:.0f} min, ${cost:.2f}"
              + (f" — {n_api} left unwritten after API failures, re-run to retry them" if n_api else ""))


def build(args) -> None:
    req = args.data_dir / "_requests"
    tot = Counter()
    per_fam: dict[str, Counter] = {}
    for task, fam in selected(args):
        its = items(task, fam)
        t_in, t_out = estimate(its, fam)
        per_fam.setdefault(fam, Counter()).update(items=len(its), t_in=t_in, t_out=t_out, tasks=1)
        if not args.dry_run:
            req.mkdir(parents=True, exist_ok=True)
            p = req / f"{task}.jsonl"
            if not p.exists():
                p.write_text("".join(json.dumps(request(task, fam, it), ensure_ascii=False) + "\n" for it in its))
    print(f"{'family':<18} {'tasks':>5} {'items':>8} {'in Mtok':>8} {'out Mtok':>9} {'batch $':>8}")
    for fam, c in per_fam.items():
        cost = (c["t_in"] * PRICE_IN + c["t_out"] * PRICE_OUT) / 1e6
        tot.update(c)
        print(f"{fam:<18} {c['tasks']:>5} {c['items']:>8} {c['t_in'] / 1e6:>8.1f} {c['t_out'] / 1e6:>9.1f} {cost:>8.0f}")
    cost = (tot["t_in"] * PRICE_IN + tot["t_out"] * PRICE_OUT) / 1e6
    print(f"{'total':<18} {tot['tasks']:>5} {tot['items']:>8} {tot['t_in'] / 1e6:>8.1f} {tot['t_out'] / 1e6:>9.1f} {cost:>8.0f}"
          f"   (+30-50 % with thinking at LOW; {MODEL} batch ${PRICE_IN}/${PRICE_OUT} per Mtok)")
    if not args.dry_run:
        print(f"request files under {req}")


def submit(args) -> None:
    c = client()
    store = gcs(args.bucket) if VERTEX else None
    state = load_state(args.data_dir)
    n = 0
    for task, fam in selected(args):
        key = task + ("#retry" if args.retry else "")
        p = args.data_dir / "_requests" / (f"{task}.retry.jsonl" if args.retry else f"{task}.jsonl")
        if key in state or not p.exists():
            continue
        if args.max_jobs and n >= args.max_jobs:
            break
        name = key.replace("#", "-")
        try:
            if store:                    # Vertex: the source is a Cloud Storage object
                src = gcs_put(*store, f"rfgm/requests/{name}.jsonl", p)
            else:                        # Gemini API: the source is an uploaded file
                src = c.files.upload(file=str(p), config={"display_name": name, "mime_type": "jsonl"}).name
            job = c.batches.create(model=MODEL, src=src, config={"display_name": f"rfgm-{name}"})
        except Exception as e:                       # quota (RESOURCE_EXHAUSTED) or network: stop, re-run later
            print(f"{key}: submit failed — {e}\nstopping; re-run submit later")
            break
        state[key] = {"job": job.name, "src": src, "dest": dest_of(job, src),
                      "state": job.state.name, "n": sum(1 for _ in p.open()), "fetched": False}
        save_state(args.data_dir, state)
        n += 1
        print(f"{key}: {job.name} ({state[key]['n']} items)")
    print(f"{n} jobs submitted this run; {sum(1 for v in state.values() if not v['fetched'])} in flight or unfetched")


def dest_of(job, src: str) -> str | None:
    """Where the answers will land: the Cloud Storage prefix the SDK derives
    from the source (`<src without .jsonl>/dest`) when the service has not
    filled it in yet, and nothing at all on the Gemini API, whose results come
    from the Files API instead."""
    uri = getattr(getattr(job, "dest", None), "gcs_uri", None)
    return uri or (src[: -len(".jsonl")] + "/dest" if src.startswith("gs://") else None)


DONE = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def refresh(c, d: Path, state: dict) -> None:
    for key, v in state.items():
        if not v["fetched"] and v["state"] not in DONE:
            v["state"] = c.batches.get(name=v["job"]).state.name
    save_state(d, state)


def status(args) -> None:
    state = load_state(args.data_dir)
    refresh(client(), args.data_dir, state)
    by = Counter(v["state"] + (" (fetched)" if v["fetched"] else "") for v in state.values())
    for k, n in sorted(by.items()):
        print(f"{n:>5}  {k}")
    print(f"{len(state)} jobs of {len(originals())} tasks; "
          f"{sum(1 for _ in args.data_dir.glob('*.jsonl'))} task files written")


def result_key(rec: dict, by_text: dict[str, str]) -> str | None:
    """Which item a result line belongs to: the `key` we sent (Gemini API), or
    the echoed prompt (Vertex, which drops every field outside `request`)."""
    if rec.get("key"):
        return rec["key"]
    try:
        return by_text.get(rec["request"]["contents"][0]["parts"][0]["text"])
    except (KeyError, IndexError, TypeError):
        return None


def parse_results(raw: bytes, by_text: dict[str, str]) -> dict[str, dict | str]:
    """key -> parsed answer dict, or an error string; plus the token usage
    under '_usage' and the count of unmatched lines under '_orphans'."""
    out: dict = {}
    usage, orphans = Counter(), 0
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = result_key(rec, by_text)
        if key is None:
            orphans += 1
            continue
        if "response" not in rec:
            out[key] = f"api error: {json.dumps(rec.get('error') or rec.get('status'))[:200]}"
            continue
        resp = rec["response"]
        u = resp.get("usageMetadata", {})
        usage.update(prompt=u.get("promptTokenCount", 0), out=u.get("candidatesTokenCount", 0),
                     thoughts=u.get("thoughtsTokenCount", 0))
        try:
            out[key] = json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            out[key] = f"unparseable response: {e}"
    out["_usage"], out["_orphans"] = dict(usage), orphans
    return out


def fetch(args) -> None:
    c = client()
    store = gcs(args.bucket) if VERTEX else None
    state = load_state(args.data_dir)
    refresh(c, args.data_dir, state)
    for key, v in list(state.items()):
        if v["fetched"]:
            continue
        if v["state"] in DONE - {"JOB_STATE_SUCCEEDED"}:
            print(f"{key}: {v['state']} — cleared, `submit` will resubmit it")
            del state[key]
            save_state(args.data_dir, state)
            continue
        if v["state"] != "JOB_STATE_SUCCEEDED":
            continue
        task, retry = key.split("#")[0], key.endswith("#retry")
        fam = dict(originals())[task]
        its = items(task, fam)
        by_id = {it["id"]: it for it in its}
        by_text = {user_content(it): f"{task}:{it['id']}" for it in its}
        raw = (gcs_read(store[0], v["dest"] or dest_of(c.batches.get(name=v["job"]), v["src"])) if store
               else c.files.download(file=c.batches.get(name=v["job"]).dest.file_name))
        answers = parse_results(raw, by_text)
        usage, orphans = answers.pop("_usage"), answers.pop("_orphans")
        if orphans:
            print(f"{key}: {orphans} result lines matched no item — not written")
        rows, rejects = [], []
        for k, out in answers.items():
            it = by_id[int(k.split(":")[1])]
            out = out if isinstance(out, str) else normalize(out)
            why = out if isinstance(out, str) else validate(it, out)
            (rejects if why else rows).append({**it, "reject": why} if why else row(it, out))
        out_p, rej_p = args.data_dir / f"{task}.jsonl", args.data_dir / "_rejects" / f"{task}.jsonl"
        rej_p.parent.mkdir(exist_ok=True)
        if retry:          # second round: append the recovered rows, keep only the twice-rejected
            rows = sorted([json.loads(l) for l in out_p.open()] + rows, key=lambda r: r["id"])
        out_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        rej_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejects))
        if rejects and not retry:   # one more chance for the rejected items
            (args.data_dir / "_requests" / f"{task}.retry.jsonl").write_text(
                "".join(json.dumps(request(task, fam, by_id[r["id"]]), ensure_ascii=False) + "\n" for r in rejects))
        v.update(fetched=True, rows=len(rows), rejects=len(rejects), usage=usage)
        save_state(args.data_dir, state)
        print(f"{key}: {len(rows)} rows, {len(rejects)} rejected"
              + (" -> retry file written" if rejects and not retry else ""))


def report(args) -> None:
    state = load_state(args.data_dir)
    usage = Counter()
    for v in state.values():
        usage.update(v.get("usage", {}))
    if usage:
        spent = (usage["prompt"] * PRICE_IN + (usage["out"] + usage["thoughts"]) * PRICE_OUT) / 1e6
        print(f"spent: {usage['prompt'] / 1e6:.1f} Mtok in, {usage['out'] / 1e6:.1f} out + "
              f"{usage['thoughts'] / 1e6:.1f} thinking = ${spent:.0f} at batch prices\n")
    else:
        print("(no batch state: these sets were produced by `online`, which does not keep a usage file)\n")
    # --family / --tasks narrow the report the same way they narrow a run
    want = {task for task, _ in selected(args)} if (args.family or args.tasks) else None
    print(f"{'task':<36} {'rows':>6} {'rej':>5} {'rej%':>5} {'stem':>5} {'choice':>6}  (median tokens, rewritten)")
    show = set(REVIEW_TASKS) if args.show else set()
    leak = Counter()
    for p in sorted(args.data_dir.glob("*.jsonl")):
        if want is not None and p.stem not in want:
            continue
        rows = [json.loads(l) for l in p.open()]
        rej = args.data_dir / "_rejects" / p.name
        n_rej = sum(1 for _ in rej.open()) if rej.exists() else 0
        if not rows:
            continue
        st = statistics.median(tokens(r["stem"]) for r in rows)
        ch = statistics.median(tokens(c) for r in rows for c in r["choices"])
        # The rewriter is never told which option is right, but it can often
        # work it out, and a model that writes the true statement more fully
        # than the false ones leaves a cue any small model can follow without
        # knowing anything. Measured against the originals, which have the
        # same artefact to begin with: what matters is whether we added to it.
        for r in rows:
            for tag, four in (("rewritten", r["choices"]), ("original", r["orig_choices"])):
                lens = [tokens(c) for c in four]
                leak[f"{tag}_longest"] += lens[r["gold"]] == max(lens) and lens.count(max(lens)) == 1
                leak[f"{tag}_gold"] += lens[r["gold"]]
                leak[f"{tag}_all"] += sum(lens)
            leak["n"] += 1
        print(f"{p.stem:<36} {len(rows):>6} {n_rej:>5} {100 * n_rej / (len(rows) + n_rej):>4.0f}% {st:>5.0f} {ch:>6.0f}")
        if p.stem in show:
            for r in random.Random(0).sample(rows, min(5, len(rows))):
                print(f"    #{r['id']} Q: {r['orig_question']}\n        {r['orig_choices']}\n"
                      f"       S: {r['stem']}\n        {r['choices']}  gold={r['gold']}")
    if leak["n"]:
        print(f"\nanswer-shape leak ({leak['n']:,} items; the gold is the single longest of four in 25 % "
              "of items by chance, and the rewrite should not raise that)")
        for tag in ("original", "rewritten"):
            gold = leak[f"{tag}_gold"] / leak["n"]
            other = (leak[f"{tag}_all"] - leak[f"{tag}_gold"]) / (3 * leak["n"])
            print(f"  {tag:<10} gold longest {100 * leak[f'{tag}_longest'] / leak['n']:>5.1f} %   "
                  f"mean tokens: gold {gold:>5.1f}, distractors {other:>5.1f}")

    reasons = Counter()
    for rej in args.data_dir.glob("_rejects/*.jsonl"):
        if want is not None and rej.stem not in want:
            continue
        reasons.update(re.sub(r":.*", "", json.loads(l)["reject"]) for l in rej.open())
    if reasons:
        print("\nreject reasons:", dict(reasons.most_common()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["pilot", "online", "build", "submit", "status", "fetch", "report"])
    p.add_argument("--family", choices=list(TEMPLATES))
    p.add_argument("--tasks", help="comma-separated original task names")
    p.add_argument("--n", type=int, default=10, help="pilot: items per task")
    p.add_argument("--dry-run", action="store_true", help="build: estimate only")
    p.add_argument("--max-jobs", type=int, default=0, help="submit: at most N new jobs this run (0 = all)")
    p.add_argument("--retry", action="store_true", help="submit: the rejected items of fetched tasks")
    p.add_argument("--force", action="store_true", help="pilot: redo tasks whose pilot file exists")
    p.add_argument("--L", type=int, help="keep only tasks in the languages an L-cell trains (e.g. 50); "
                                         "the ladder scores no others, so the rest is paid for and never read")
    p.add_argument("--scheme", default="A", help="the data scheme --L refers to (default A)")
    p.add_argument("--limit", type=int, default=0, help="online: at most N new items per task (0 = all)")
    p.add_argument("--concurrency", type=int, default=12,
                   help="online: parallel requests (default 12; 24 trips the shared quota)")
    p.add_argument("--retry-rejects", action="store_true",
                   help="online: re-run the items a previous pass rejected (clears the reject file)")
    p.add_argument("--drop-self-ref", action="store_true",
                   help="online: skip items whose options refer to other options; off by default "
                        "so the set stays comparable with the originals and the rf_ twins")
    p.add_argument("--show", action="store_true", help="report: print five sample items of every REVIEW_TASKS task")
    p.add_argument("--bucket", help="Vertex only: the Cloud Storage bucket for the batch objects "
                                    "(default $RFGM_GCS_BUCKET, else <project>-msnr-rfgm)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"default {DATA_DIR}")
    args = p.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
