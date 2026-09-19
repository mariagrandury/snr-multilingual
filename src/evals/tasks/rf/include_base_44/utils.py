"""Drop rows whose prompt fields are missing (make_rf_tasks.py writes this)."""
FIELDS = ("question", "option_a", "option_b", "option_c", "option_d")


def process_docs(dataset):
    return dataset.filter(lambda r: all(isinstance(r[k], str) and r[k].strip()
                                        for k in FIELDS))
