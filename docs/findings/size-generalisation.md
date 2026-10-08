---
title: "RQ10 · Past the reference"
---

# RQ10 · Does a ranking that holds at 1.7B still hold at 3B?

**The question.** Every other page judges a small proxy by whether it makes
the decisions the 1.7B model makes. But the 1.7B model is itself a proxy for
the larger models one actually wants to train. We trained a few 3B models to
ask the one question the rest of the study cannot: is the reference itself
predictive of the next size up?

**How we measure it.** On the training choices the 3B models compare, we
compute the decision accuracy of every smaller model against the 3B model,
and on the same pairs against the 1.7B model. If the lines match, a proxy
that reads the 1.7B decision also reads the 3B one.

<div class="viz" data-viz="aboveReference"></div>

Only four 3B runs exist so far, so each point rests on few design pairs:
read the trend, not single points.

[Full analysis →](../signal-noise/analysis.md)
