"""Drop items without four usable choices_en (make_include_v2_tasks.py writes this)."""


def process_docs(dataset):
    return dataset.filter(
        lambda r: r["answer"] in ("A", "B", "C", "D")
        and len(r["choices_en"]) == 4
        and all(isinstance(c, str) and c.strip() for c in r["choices_en"])
        and isinstance(r["question_en"], str) and r["question_en"].strip())
