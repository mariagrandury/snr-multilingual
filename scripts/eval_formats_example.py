"""Score the worked example of the formats-and-scoring figure (evalFormats,
data-panel="detailed" in docs/interactive/views/slides.js) with a small public
model, the way the harness scores a multiple-choice item: the log-likelihood of
each option as the continuation of the prompt.

For every format (letter, cloze, continuation) and option it writes the summed
log-likelihood in nats, the same per byte of the option (what length-normalised
accuracy compares) and the bits per byte (bBPB, read on the gold option).

    pip install torch transformers
    python3 scripts/eval_formats_example.py   # writes docs/interactive/data/eval_formats_example.json
"""
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"
OUT = Path(__file__).resolve().parents[1] / "docs/interactive/data/eval_formats_example.json"
QUESTION = "Which is the capital of Spain?"
OPTIONS = ["Paris", "Zurich", "Madrid", "Buenos Aires"]
GOLD = 2
FORMATS = {
    "letter": (QUESTION + "\n" + "\n".join(f"{l}. {o}" for l, o in zip("ABCD", OPTIONS)) + "\nAnswer:", [" A", " B", " C", " D"]),
    "cloze": (f"Question: {QUESTION}\nAnswer:", [" " + o for o in OPTIONS]),
    "continuation": ("The capital of Spain is", [" " + o for o in OPTIONS]),
}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).eval()


def loglik(context: str, continuation: str) -> float:
    """Summed log-probability (nats) of the continuation's tokens given the context."""
    ctx = tok(context)["input_ids"]
    full = tok(context + continuation)["input_ids"]
    with torch.no_grad():
        logp = model(torch.tensor([full])).logits[0].log_softmax(-1)
    return sum(logp[i - 1, full[i]].item() for i in range(len(ctx), len(full)))


out = {"model": MODEL, "gold": GOLD, "formats": {}}
for name, (prompt, conts) in FORMATS.items():
    rows = []
    for c in conts:
        ll, nbytes = loglik(prompt, c), len(c.encode("utf-8"))
        rows.append(dict(option=c, loglik=round(ll, 3), per_byte=round(ll / nbytes, 3),
                         bpb=round(-ll / (math.log(2) * nbytes), 3)))
    pick = max(range(len(rows)), key=lambda i: rows[i]["per_byte"])
    out["formats"][name] = dict(prompt=prompt, options=rows, pick=pick)
    print(name, "picks", rows[pick]["option"], [(r["option"], r["loglik"], r["bpb"]) for r in rows])
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("wrote", OUT)
