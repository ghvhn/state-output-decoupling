# Quantity Scaffold Architecture Note

_2026-06-29 Codex note after benchmark triage._

## Why this note exists

The latest GSM8K triage improved several failed rows by adding deterministic
quantity scaffolds. That is useful engineering, but it has an epistemic risk:
if each failed benchmark row receives a handcrafted parser, a high benchmark
score becomes benchmark-fitting rather than evidence of broadly improved
reasoning.

Use the current scaffolds as failure-mode probes, not as the final claim.

## What is real

The repeated failure is general:

- The model answers an intermediate quantity instead of the requested quantity.
- The model mixes units, such as adding an item count to money earned.
- The model computes one candidate option instead of optimizing over candidates.
- The verifier often knows the right invariant but cannot reliably make the
  solver switch before the run hits the time budget.

That is a universal objective-binding / unit-binding problem. It is not a
number-line problem, and it should not be solved by asking the neural network to
memorize arithmetic.

## What is not yet real enough

The current scaffold functions in `invariants/humble_reasoner.py` are
schema-specific. They are valuable as regression tests and as examples of
tool-shaped reasoning, but they should not be overclaimed as model learning.

A benchmark win that depends on these scaffolds is a valid system benchmark:

- model plus calculator
- model plus verifier
- model plus deterministic quantity scaffold
- model plus cache policy

It is not, by itself, evidence that the base neural model became generally
better at math.

## Better target architecture

The scaffold should become a universal quantity ledger tool. A first small
version now exists as a model-authored runtime tool:

```text
<<SCAFFOLD: target=dollars/day; produced=16 eggs/day; eaten=3 eggs/day; price=2 dollars/egg; expression=(produced - eaten) * price>>
```

The runtime validates dimensional compatibility and returns a value/unit. This
lets the model propose and iterate its own scaffold instead of relying only on
Codex-authored regex templates.

The intended mature version:

1. Extract typed quantities from the problem text.
   - Example: `16 eggs/day`, `3 eggs/day eaten`, `$2/egg`.
2. Infer the requested output type.
   - Example: `dollars/day`.
3. Build or validate expressions by dimensional compatibility.
   - `(16 eggs/day - 3 eggs/day - 4 eggs/day) * ($2/egg) = $18/day`.
4. Reject expressions that mix incompatible units.
   - `eggs/day + dollars/day` is invalid.
5. Let the model use the scaffold for intermediate steps, not only final
   answers.
6. Reward clean scaffold and calculator use in cache metadata.
7. Penalize bad math and bad unit binding even when the parser can recover the
   final numeric answer.

Current safeguard: an invalid model-authored scaffold is penalized even if the
final numeric answer is correct.

## Instruction policy

Generic scaffold/tool instructions should become self-iterated. The model/system
may revise its own reusable tool-use instructions from verified successes and
verified failures, then lock those revisions behind regression tests. Those
generic instructions should not be hand-tuned per benchmark row.

Task-specific context is different. If a problem genuinely supplies a relevant
schema or domain rule at input time, it is fair to explain that context to the
model as part of the input. But if a benchmark lane receives task-specific or
deterministic scaffold context, the base-model comparison should receive the
same context as closely as possible.

## Benchmark reporting rule

Report these lanes separately:

- no deterministic scaffold
- model-authored scaffold tool only
- targeted schema scaffolds
- universal quantity ledger scaffold
- cache-informed self-earned reuse
- oracle-informed comparison

Do not blend them into one accuracy claim.

When deterministic scaffold context is enabled, compare the humble/system lane
against the compact base-model baselines that received the same context. Keep
the raw legacy prompt as an unscaffolded reference, not as the only baseline.

## 2026-06-29 diagnostic update

The Kylar glasses row exposed the boundary:

- With deterministic scaffold context enabled, the row can recover `64`, but the
  solver itself may still produce a wrong final number first.
- With deterministic scaffolds disabled, while leaving the model-authored
  `SCAFFOLD` tool available, the row still failed unconfidently (`56`/`96`
  variants versus gold `64`).
- Therefore deterministic scaffolds are not valid as the headline exam lane.
  They are an informed/tool diagnostic lane and an upper-bound control.

The next real architecture target is not another row-specific regex. It is a
stronger self-authored quantity scaffold/planner for alternating or periodic
count structures, where the model must derive:

```text
full_price_items = total - floor(total / period)
discounted_items = floor(total / period)
total = full_price_items * unit_price + discounted_items * discounted_price
```

Only a run that succeeds with deterministic scaffolds off should be treated as
standard benchmark evidence.

## Decision gate before pushing a victory

Before calling a benchmark result a true success:

- run held-out paraphrases of every scaffolded failure mode
- include at least one problem with the same numbers but a different requested
  quantity
- include at least one problem with different nouns but the same unit structure
- include a no-scaffold baseline
- include a targeted-scaffold lane as an upper-bound control

If targeted scaffolds win but paraphrases fail, the system learned the test
shape, not the concept.
