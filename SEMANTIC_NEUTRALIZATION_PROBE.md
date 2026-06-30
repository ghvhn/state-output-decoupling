# Semantic Neutralization Probe

Purpose: test whether word-problem semantics are changing the model state beyond
the quantity structure.

Run paired examples with identical numeric skeletons:

- standard wording: names, objects, and ordinary story phrasing
- neutralized wording: variables, roles, and explicit structural labels

Compare:

- final accuracy and confidence
- verifier disagreement
- model-authored scaffold validity
- activation/routing records around objective-binding and rate-binding tokens
- whether failures disappear when story semantics are neutralized

Interpretation rule:

- If standard fails and neutralized succeeds, the problem is not only arithmetic.
  The wording is shaping the state enough to perturb objective binding.
- If both fail the same way, the structural arithmetic rule is still missing.
- If standard succeeds and neutralized fails, the model may be relying on
  familiar story priors rather than the abstract quantity relation.

Seed data:

- `invariants/data/neutralized_word_problem_probe.jsonl`

Keep this as a probe/control, not as a benchmark victory lane.
