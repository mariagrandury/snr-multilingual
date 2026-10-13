---
title: Findings
---

# Findings

We set out three questions, in the order a developer should ask them when
designing their ablation configurations.

1. **[Measurability](measurability.md).** At which model sizes and
   training-token counts can we expect an evaluation to produce a score
   distinguishable from noise?
2. **[Agreement](agreement.md).** At which proxy model sizes and per-language
   training token counts does the ranking of the proxy agree with the ranking
   of the fully trained reference model?
3. **[Anticipation](anticipation.md).** Which statistics computed from the
   proxy runs alone are predictive of this decision reliability, such that a
   developer can estimate how well they can trust their proxy evaluation
   configuration without training the reference model?

Each page follows the corresponding section of the paper, with interactive
versions of its figures.
