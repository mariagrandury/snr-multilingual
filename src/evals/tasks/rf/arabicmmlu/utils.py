"""rf_arabicmmlu: question (+ context) only, the present options as choices (make_rf_tasks.py writes this)."""
OPTS = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]


def process_docs(dataset):
    def _doc(r):
        opts = []
        for k in OPTS:
            if r[k] is None or not str(r[k]).strip():
                break
            opts.append(str(r[k]).strip())
        q = r["Question"] if not r["Context"] else f"{r['Context']}\n\n{r['Question']}"
        return {"rf_text": q.strip(), "rf_choices": opts, "rf_gold": "ABCDE".index(r["Answer Key"])}
    return dataset.map(_doc)
