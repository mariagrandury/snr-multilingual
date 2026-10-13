"""Drop rows whose prompt fields are missing (make_rf_tasks.py writes this)."""
FIELDS = ('question', 'op1', 'op2', 'op3', 'op4')


def process_docs(dataset):
    return dataset.filter(lambda r: all(isinstance(r[k], str) and r[k].strip()
                                        for k in FIELDS))
