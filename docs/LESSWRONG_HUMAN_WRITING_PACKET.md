# LessWrong Human Writing Packet

This is not a post draft. It is a writing aid for a post that must be written
by Gavin in Gavin's own words.

## Policy Boundary

Do not paste AI-written prose into the forum.

Safe uses for this packet:

- choose a title
- decide section order
- check claim boundaries
- verify file names and evidence
- remember which caveats to include
- turn each prompt into your own paragraph

Unsafe uses:

- copy a paragraph from an AI draft
- lightly paraphrase AI prose while preserving its structure and wording
- submit without disclosure if AI contributed substantive wording

## Recommended Title Options

Pick one and rewrite if needed:

- State-Output Decoupling and the U-Shaped Bottleneck
- A Local Case Study in Separating Latent State from Rendered Output
- Interpretation, Workspace, Render: A White-Box LLM Case Study

Avoid in the title:

- "evidenced programmatic ingestion"
- "parallel discovery" as the main claim
- anything that makes the provenance issue sound like the primary evidence

## One-Sentence Thesis Prompt

Write one sentence answering:

What distinction did the repo make measurable that ordinary LLM discussion tends to collapse?

Must include the idea:

```text
final text is not the whole model
```

Possible terms to use in your own sentence:

- interpretation
- latent task state
- communication/render
- state-output decoupling

## Opening Paragraph Prompts

Write 2-3 short paragraphs from these prompts:

1. How did you arrive at this work, and why did philosophy push you toward measurement?
2. What was the practical question you started trying to answer?
3. Why is the generated answer an unsafe proxy for the model's internal state?

Boundary to include:

```text
This is not a consciousness proof. It is a local white-box case study.
```

## Core Model To Explain

Use this diagram, but explain it yourself:

```text
prompt text -> interpretation / translation -> latent task state -> communication / render -> output text
```

Explain the layer-purpose model in your own words:

- early: input translation, role binding, operation/intent
- middle: reusable latent task state, uncertainty, workspace-like variables
- late: communication, output format, persona, final answer text

Important correction:

Do not say every layer has a fixed purpose. Say these are functional regimes
or depth profiles.

## Evidence Checklist

Before writing each claim, open the relevant file and verify the numbers.

- `docs/ARCHITECTURAL_REALITY.md`
  - operation/intent not reducible to surface material
  - final text as communication/render
  - CoT as generated trajectory, not automatically reasoning

- `docs/RUNNING_EXPERIMENT.md`
  - `intent_surface_control`
  - `translation_thinking`
  - `cot_reality`
  - `cot_perturb`
  - social/standpoint/felt-tone probes if you decide to include them

- `FINDINGS.json`
  - L16 self-denial/hedge decodability
  - frame-conditioned self-report
  - uncertainty/self-report/tier claims

- `docs/PUBLIC_POST_DRAFT.md`
  - self-report as costume
  - origin table
  - representation vs causal role distinction

## Section Order

Use this order unless you have a strong reason not to:

1. Why I am posting
2. The distinction: state is not output
3. The U-shaped bottleneck
4. Local evidence from the repo
5. Relation to the J-space/global-workspace paper
6. What I am not claiming
7. Timeline and provenance note
8. What feedback I want

## Claim Boundaries

Use these boundaries in your own wording:

- local case study, not universal model law
- Llama-3.1-8B-Instruct, not all LLMs
- representation does not equal experience
- decodability does not equal causal control
- output text is evidence, but not privileged evidence
- provenance anomaly is documented separately from the scientific claim

## Provenance Note Prompt

If you include the traffic/timeline issue, keep it bounded.

Answer these questions plainly:

1. What happened?
2. What do you know from logs or GitHub traffic?
3. What are mundane explanations?
4. What are you not claiming?
5. Why mention it anyway?

Required boundary:

```text
This is not evidence, by itself, that anyone copied or ingested the work.
```

## Relation To The J-Space Paper

Write this section as comparison, not accusation.

Core mapping:

```text
J-space paper: sensory -> workspace -> motor
my framing:    input translation -> latent work -> communication render
```

Say in your own words:

- their work is stronger on the positive workspace claim
- your work is more focused on state-output decoupling and controls around self-report
- verbalizable representations are useful but not equivalent to truth, causality, or experience

## Questions For Readers

Pick 3-5:

- What nulls would better separate input translation from lexical carry-through?
- How would you test whether late communication steering preserves earlier task state?
- What is the right control for mini-U computation across generated reasoning steps?
- Is there existing mechanistic interpretability work that already refutes this framing?
- How should provenance/timeline facts be documented without overclaiming?

## Final Pass Checklist

Before posting:

- Does the first paragraph state the main point clearly?
- Is the AI assistance policy satisfied?
- Are copied AI paragraphs removed?
- Are all numbers checked against repo files?
- Are speculative claims labeled as speculative?
- Is the provenance issue secondary, not the headline?
- Does the post ask for critique rather than demand agreement?

