---
description: Audit a paper draft or README for claims that the evidence does not support
argument-hint: "<path to draft .tex/.md, default papers/01-cage>"
---
Audit `${ARGUMENTS:-papers/01-cage}` against the epistemic rules in CLAUDE.md.
For every quantitative or comparative claim:
- find the results file that backs it (`results/**/results.json` or `summary.md`); if none, flag it;
- check the number matches the file exactly;
- flag scenario/model numbers stated as measurements (rule E1);
- flag raw modularity without a null model (E2), baselines other than block placement presented as
  the headline (E3), betweenness used as evidence (E4), and novelty claims contradicted by
  docs/review_findings.md §2.
Output a table: claim | location | backing file | verdict (supported / wrong number / unsupported /
mislabelled) | fix. Do not edit the draft; propose exact replacement text.
