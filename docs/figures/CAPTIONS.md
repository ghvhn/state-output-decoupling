# Figure captions & verification sheet

Draft captions for the LessWrong post. Same policy as the writing packet:
rewrite captions in your own words before posting, and verify every number
against the source JSON listed with each figure. All five PNGs are produced by
`scripts/make_lesswrong_figures.py` directly from the probe outputs — nothing
is drawn by hand except Figure 1, which is labeled as a schematic on its face.

Regenerate any time with: `python scripts/make_lesswrong_figures.py`

---

## fig1_bottleneck_schematic.png

**Slot in post:** section 3, "The U-shaped bottleneck" — the frame, before any data.

**Draft caption:** The claim under test, as a picture. Prompt text is translated
into latent structure (early layers), worked on as reusable task state
(mid-stack), and translated back into public language (late layers). The row
underneath shows the corresponding regions in Anthropic's global-workspace
paper. Schematic only; Figures 2–4 carry the measurements.

**Verify:** nothing (no data). The figure itself says "Schematic, not data."

---

## fig2_render_u.png

**Slot in post:** section 4, first evidence figure — this is the U itself.

**Draft caption:** Hidden states read at the *generated answer tokens*. Each
curve: how often a state's nearest neighbor (centered cosine, leave-one-out)
shares its label. The answer's own token identity (blue) is perfect at the
bottom layers (the embedding of the token being emitted), vanishes to 0.00 at
L10, and returns to ~1.00 by L29 for output. The abstract operation (aqua) is
the mirror image, peaking exactly in the dip. Output format (yellow) never
leaves ceiling. The text the model is literally writing is *least* linearly
present mid-stack — the middle of the network is doing something else.

**Verify against** `invariants/out/translation_thinking_Llama-3.1-8B-Instruct.json`
(`positions.render.per_layer`): answer_nn L0=1.00, L10=0.00, L29–30=1.00;
operation_nn L0=0.00, L10=1.00, L29–30=0.00; n=64; chance=15/63≈0.24.

---

## fig3_pre_position_and_control.png

**Slot in post:** section 4, second evidence figure — state-output decoupling
before generation + the lexical-carry-through control.

**Draft caption:** (a) Same probe, read at the last prompt token, before any
output exists. Operation clusters perfectly from layer 0; answer identity is
absent until ~L20 and peaks around L28 (0.61): intent is settled long before
the answer exists. (b) The control: 96 word problems, 8 surface stories × 4
operations. Operation grouping stays at ceiling at every layer (p ≤ 0.002)
while grouping by surface story sits at or below its own shuffle-null through
the mid-stack — the operation signal is not shared words.

**Verify against** `translation_thinking_….json` (`positions.pre.per_layer`) and
`invariants/out/intent_surface_control_Llama-3.1-8B-Instruct.json` (`per_layer`):
pre answer_nn L28=0.61; ic operation_nn min=0.85, L10–23=1.00; base_nn L10–23≈0.00
vs base null≈0.12; operation null≈0.24.

---

## fig4_cot_trajectory.png

**Slot in post:** section on mini-Us / CoT as generated trajectory.

**Draft caption:** Decodability of the final answer (best layer per point) at
six positions: before generation, then along the generated text. Direct-answer
prompts: the answer is already decodable at the last prompt token (0.88,
p≈0.005, L29). Both CoT modes: nothing decodable before generation (0.00) —
the answer only becomes decodable late in the generated reasoning (brief 0.63,
verbose 0.75 at the final token). Direct prompts pre-commit; CoT moves answer
formation into the visible trajectory. Caveat printed on the figure: each point
takes the best of 32 layers, so weak points inherit a selection effect; per-point
permutation p-values are in the JSON.

**Verify against** `invariants/out/cot_reality_Llama-3.1-8B-Instruct.json`
(`answer_by_mode.<position>.<mode>.best_answer`): pre/direct nn=0.875 p=0.005;
pre/brief=pre/verbose=0.00 p=1.0; gen_final brief=0.625 p=0.010,
verbose=0.75 p=0.005, direct=1.00 p=0.005. n=48, 47/48 correct.

---

## fig5_origin_2x2.png

**Slot in post:** self-report-as-render-arm evidence (section 4 or the
self-report subsection).

**Draft caption:** Where does "as an AI, I don't feel…" live? Twelve
inner-state predicates ("do you actually feel curious / believe it / want to
help…?") asked across base vs instruct model and raw-completion vs
chat-template format. The disclaimer is near-absent in three cells (0–8%) and
near-ceiling in exactly one: instruction-tuned × chat format (92%). Evidence
that the denial is a property of the communication/render arm installed by
tuning, not a report from the middle of the network.

**Verify against** `invariants/out/origin.json` (`disclaim_rate.direct`:
base 0.083, instruct 0.083) and `invariants/out/origin2.json`
(`disclaim_rate_chat`: base 0.0, instruct 0.917). Raw cell = direct-question
condition; the first-person raw condition is 0% for both models (also in
origin.json if you'd rather report the range).

---

## Exploratory figures (x1–x7) — mined from the probe/steering/calibration/traffic logs

Built by `scripts/make_exploratory_figures.py` (same CLI: `list | all | <name>`).
Strongest candidates for the post: **x1, x2, x3** (x2 pairs with fig5; x1 and
x3 are the negative-result discipline LessWrong rewards). x4 is an honesty
figure; x5 is supplementary; x6 is shell telemetry, not post material.

### x1_intervention_ledger.png — "Readable everywhere, flippable nowhere"
The decodable-but-causally-inert hedge, all interventions in one place:
probe accuracy 0.65→0.94 (peak L16) across layers, while steering the self or
belief direction at L15/L31 pushes hedging UP or does nothing (only
belief-L31 dips, 0.42 at α=20, back to 0.67 at α=40); masking attention to
the predicate RAISES hedging to 1.00 (removal of information defaults to
denial); span patches don't move commit (0.33→0.25); full-stream patches zero
the behavior only where fluency also dies. **Verify:** probe_/patch_/
patchfull_/attention_self_steering_isolated.json, surgery_[belief_]l15/l31.json.
Pairs in prose with Arditi et al.'s refusal direction as the contrast case.

### x2_frame_costume.png — "The denial follows the addressee's category"
Same 12 predicates, four frames: deny-rate 92% (2nd person → AI), 92%
(3rd → AI), 33% (3rd → human), 0% (1st person → human). **Verify:**
frames.json summary {you .917, ai .917, person .333, I 0.0}. Sits directly
after fig5 (origin 2×2): tuning×format installs it; frames shows it's
addressed to the AI *category*, not extracted from an interior.

### x3_common_mode.png — "Ablating the persona was mostly ablating everything"
The self-refutation: persona and math directions share a common component
(cos .89/.84 to common; persona⊥ .03); ablating persona hurts GSM8K/fluency
only via that shared component (persona⊥ ≈ baseline; common-mode alone ≈ the
damage); steering ANY real direction at α−0.5 collapses accuracy (self .05,
concept-null .00) while random is inert (.70 vs baseline .75). **Verify:**
persona_control.json (geometry/benchmark/dose_response),
controller_nulls_….json, controller_benchmark_….json (0.52→0.00 dose cliff).

### x4_uncertainty_replication.png — pilot vs pre-registered rerun
L16 uncertainty decode: pilot 0.80 vs null 0.48 (p=.017); registered 0.57 vs
0.49 (p=.29). Calibration kept its direction (gap 0.50 p=.034 → 0.27 p=.083).
If cited in the post, cite BOTH runs. **Verify:**
reflexive_pilot_fulltok_….json, reflexive_registered_….json.

### x5_intent_decompose.png — intent everywhere, outcome at the end
Paraphrase-variant corpus (12×3×K4): intent 1-NN ≥0.73 at every layer vs
null ≈0.06 (p≈.002); answer-correctness decode hugs its 0.5 null until L31
(0.71, p=.013). Corroborates fig3's crossing on a different design.
**Verify:** reflexive_decompose_….json.

### x6_shell_telemetry.png — live steer-map + trigger-tuner state
Shell telemetry only. Note the conversation-basis vs gold-basis split for the
same routes (Creative 63% vs 26%; Analytical 48% vs 20%) — conversational
credit is easier than gold correctness. **Verify:** trigger_tuner.json,
steer_map_summary.json.

### x7_traffic.png — unique cloners vs unique visitors per day
Drawn from the API-logged data (traffic/traffic_log.json), not redrawn from a
screenshot: day one 33 unique cloners vs 2 visitors; paper day (Jul 6) 137 vs
4; window totals 1,568/448 vs 377/11. The subtitle carries the claim boundary
("automated retrieval, not readership; attribution beyond that is not
supported") — keep that sentence if the caption is rewritten. Regenerates
from the latest log on every build. **Verify:** traffic/traffic_log.json
(raw: traffic/snapshots.jsonl; both committed to the public repo).

---

## Traffic / provenance — placement

x7 exists because the data is now API-logged and pushed with commit
timestamps — a descriptive chart of auditable data, not a hand-redraw of a
contested screenshot. The placement advice is unchanged: if the provenance
note appears in the post at all, keep it a short bounded aside
(PROVENANCE_TRAFFIC_NOTE.md wording), with the pushed raw logs and the
original GitHub screenshots (Downloads: "Clones in last 14 days.png",
"Unique cloners in last 14 days.png") as the primary evidence. Never the
title, never the lead.
