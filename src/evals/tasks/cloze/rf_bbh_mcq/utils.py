"""Split BBH's `input` into stem + options (make_cloze_tasks.py writes this)."""
import re

OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def split(text):
    """(stem, labels, texts) of an item, or None when it has no usable
    option block -- two of BBH's items print no list at all, and one of
    snarks' prints a single option, which is not a choice."""
    stem, sep, body = text.partition("\nOptions:")
    if not sep:
        return None
    lines = [l.strip() for l in body.strip().split("\n") if l.strip()]
    ms = [OPT_RE.match(l) for l in lines]
    if len(lines) < 2 or not all(ms):
        return None
    texts = [m.group(2).strip() for m in ms]
    if not all(texts):
        return None
    return stem.strip(), ["(%s)" % m.group(1) for m in ms], texts


def usable(row):
    got = split(row["input"])
    return got is not None and row["target"].strip() in got[1]


def process_docs(dataset):
    def add(row):
        stem, labels, texts = split(row["input"])
        return {"stem": stem, "labels": labels, "texts": texts,
                "gold": labels.index(row["target"].strip())}
    return dataset.filter(usable).map(add)
