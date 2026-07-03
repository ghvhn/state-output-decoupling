# Confidence, Process Difficulty, And Learned Limitations

Do not collapse these into one scalar:

- answer confidence: whether the current final answer has enough clean support
- process difficulty: how much repair, synthesis, verifier disagreement, or time
  was required
- intervention risk: whether the system used oracle lessons, cache hits,
  deterministic scaffolds, or other non-clean supports

A complicated solution can still end in a confident answer when the solver
expression and independent verifier agree cleanly. A simple-looking solution can
be untrustworthy when it answers the wrong requested quantity or contains a
structural contradiction.

Urgency should choose the next step, not lower truth standards. With enough
time, the system can afford scaffold repair, independent verification, and
careful concept extraction. With little time, it should prefer the simplest
trusted expression path, avoid introducing new complex scaffold syntax, and
return a calibrated answer rather than rushing into more synthesis loops.

Smooth logic is a separate quality target:

- fewer structural contradictions
- fewer emergency repairs
- shorter path from asked quantity to expression
- verifier checks that finish before the deadline
- confidence based on support quality, not on how hard the run felt

Scoring limitations should be visible to the model as the environment contract,
not as hidden punishment:

- final numeric answer is scored
- microscopic floating-point roundoff is tolerated
- meaningful rounding is still wrong
- answering an intermediate or wrong requested quantity fails

Recurring limitations should preferably become self-learned, tagged lessons:

- answered revenue when asked for profit
- rounded a meaningful fraction
- double-charged every periodic discounted item
- multiplied before subtracting removed/kept items

Keep these tags separate from clean benchmark wins.
