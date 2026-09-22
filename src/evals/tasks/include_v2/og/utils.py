"""Drop items without four usable choices (make_include_v2_tasks.py writes this)."""


def process_docs(dataset):
    return dataset.filter(
        lambda r: r["answer"] in ("A", "B", "C", "D")
        and len(r["choices"]) == 4
        and all(isinstance(c, str) and c.strip() for c in r["choices"])
        and isinstance(r["question"], str) and r["question"].strip())
