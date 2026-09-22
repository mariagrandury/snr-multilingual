"""cultural_bench_easy twins: per-country split, answer to index, blank rows
dropped (make_rf_tasks.py writes this; it stands in for the harness's own
utils.process_<country>, which the twin inherits but which keeps None options)."""
FIELDS = ("prompt_question", "prompt_option_a", "prompt_option_b",
          "prompt_option_c", "prompt_option_d")


def _country(slug):
    def process(dataset):
        ds = dataset.filter(
            lambda r: r["country"].lower().replace(" ", "_") == slug
            and all(isinstance(r[k], str) and r[k].strip() for k in FIELDS))
        return ds.map(lambda r: {**r, "answer": "ABCD".index(r["answer"])
                                 if r["answer"] in ("A", "B", "C", "D") else r["answer"]})
    return process


process_argentina = _country("argentina")
process_australia = _country("australia")
process_canada = _country("canada")
process_chile = _country("chile")
process_china = _country("china")
process_india = _country("india")
process_japan = _country("japan")
process_mexico = _country("mexico")
process_morocco = _country("morocco")
process_nigeria = _country("nigeria")
process_peru = _country("peru")
process_russia = _country("russia")
process_saudi_arabia = _country("saudi_arabia")
process_south_africa = _country("south_africa")
process_spain = _country("spain")
process_united_kingdom = _country("united_kingdom")
process_united_states = _country("united_states")
process_vietnam = _country("vietnam")
process_zimbabwe = _country("zimbabwe")