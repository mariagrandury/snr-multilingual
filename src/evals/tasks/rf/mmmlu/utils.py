"""Drop rows whose prompt fields are missing (make_rf_tasks.py writes this)."""
FIELDS = ('Question', 'A', 'B', 'C', 'D')


def process_docs(dataset):
    return dataset.filter(lambda r: all(isinstance(r[k], str) and r[k].strip()
                                        for k in FIELDS))
