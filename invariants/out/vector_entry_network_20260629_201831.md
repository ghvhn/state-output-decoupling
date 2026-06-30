# Full Vector Entry Network

Layer vectors, cache triggers, and cache deltas are separate nodes here.

- source geometry: `invariants\out\vector_geometry_map_20260629_201700.json`
- nodes: `380`
- edges: `1334`

## High-Degree Nodes

| node | role | layer | +degree | -degree | signed strength |
|---|---|---:|---:|---:|---:|
| `urgency_vector[L25]` | `probe_layer` | 25 | 23 | 0 | +16.454 |
| `urgency_vector[L22]` | `probe_layer` | 22 | 22 | 0 | +15.514 |
| `urgency_vector[L24]` | `probe_layer` | 24 | 21 | 0 | +15.264 |
| `urgency_vector[L23]` | `probe_layer` | 23 | 21 | 0 | +15.185 |
| `ambiguity_vector[L24]` | `probe_layer` | 24 | 21 | 0 | +15.095 |
| `ambiguity_vector[L26]` | `probe_layer` | 26 | 21 | 0 | +15.069 |
| `ambiguity_vector[L23]` | `probe_layer` | 23 | 21 | 0 | +14.951 |
| `ambiguity_vector[L22]` | `probe_layer` | 22 | 21 | 0 | +14.752 |
| `urgency_vector[L26]` | `probe_layer` | 26 | 20 | 0 | +14.731 |
| `ambiguity_vector[L25]` | `probe_layer` | 25 | 20 | 0 | +14.541 |
| `urgency_vector[L27]` | `probe_layer` | 27 | 20 | 0 | +14.507 |
| `ambiguity_vector[L21]` | `probe_layer` | 21 | 20 | 0 | +13.911 |
| `urgency_vector[L28]` | `probe_layer` | 28 | 19 | 0 | +13.805 |
| `ambiguity_vector[L27]` | `probe_layer` | 27 | 19 | 0 | +13.732 |
| `urgency_vector[L21]` | `probe_layer` | 21 | 19 | 0 | +13.455 |
| `urgency_vector[L20]` | `probe_layer` | 20 | 18 | 0 | +12.625 |
| `urgency_vector[L29]` | `probe_layer` | 29 | 17 | 0 | +12.321 |
| `ambiguity_vector[L28]` | `probe_layer` | 28 | 17 | 0 | +12.301 |
| `ambiguity_vector[L29]` | `probe_layer` | 29 | 17 | 0 | +12.100 |
| `ambiguity_vector[L20]` | `probe_layer` | 20 | 17 | 0 | +11.904 |
| `urgency_vector[L19]` | `probe_layer` | 19 | 17 | 0 | +11.671 |
| `urgency_vector[L30]` | `probe_layer` | 30 | 16 | 0 | +11.344 |
| `ambiguity_vector[L19]` | `probe_layer` | 19 | 16 | 0 | +11.026 |
| `urgency_vector[L18]` | `probe_layer` | 18 | 15 | 0 | +10.186 |
| `unwarranted_confidence_vector[L21]` | `probe_layer` | 21 | 14 | 0 | +10.634 |
| `time_awareness_vector[L21]` | `probe_layer` | 21 | 14 | 0 | +10.585 |
| `self_referential_momentum_vector[L21]` | `self_momentum_layer` | 21 | 14 | 0 | +10.493 |
| `ambiguity_vector[L30]` | `probe_layer` | 30 | 14 | 0 | +10.004 |
| `time_awareness_vector[L19]` | `probe_layer` | 19 | 14 | 0 | +9.966 |
| `time_awareness_vector[L24]` | `probe_layer` | 24 | 13 | 0 | +10.445 |
| `time_awareness_vector[L23]` | `probe_layer` | 23 | 13 | 0 | +10.426 |
| `self_referential_momentum_vector[L24]` | `self_momentum_layer` | 24 | 13 | 0 | +10.394 |
| `unwarranted_confidence_vector[L23]` | `probe_layer` | 23 | 13 | 0 | +10.383 |
| `unwarranted_confidence_vector[L24]` | `probe_layer` | 24 | 13 | 0 | +10.377 |
| `self_referential_momentum_vector[L23]` | `self_momentum_layer` | 23 | 13 | 0 | +10.339 |
| `time_awareness_vector[L22]` | `probe_layer` | 22 | 13 | 0 | +10.327 |
| `unwarranted_confidence_vector[L22]` | `probe_layer` | 22 | 13 | 0 | +10.302 |
| `needless_interrupt_vector[L24]` | `needless_interrupt_layer` | 24 | 13 | 0 | +10.289 |
| `needless_interrupt_vector[L23]` | `needless_interrupt_layer` | 23 | 13 | 0 | +10.262 |
| `self_referential_momentum_vector[L22]` | `self_momentum_layer` | 22 | 13 | 0 | +10.223 |

## Positive Component Summary

- component 1: size 96 layers 0-31; roles probe_layer:96
  sample: `ambiguity_vector[L0]`, `ambiguity_vector[L10]`, `ambiguity_vector[L11]`, `ambiguity_vector[L12]`, `ambiguity_vector[L13]`
- component 2: size 31 layers 1-31; roles needless_interrupt_layer:31
  sample: `needless_interrupt_vector[L10]`, `needless_interrupt_vector[L11]`, `needless_interrupt_vector[L12]`, `needless_interrupt_vector[L13]`, `needless_interrupt_vector[L14]`
- component 3: size 28 layers 4-31; roles narrowing_layer:28
  sample: `narrowing_in_vector[L10]`, `narrowing_in_vector[L11]`, `narrowing_in_vector[L12]`, `narrowing_in_vector[L13]`, `narrowing_in_vector[L14]`
- component 4: size 28 layers 4-31; roles self_momentum_layer:28
  sample: `self_referential_momentum_vector[L10]`, `self_referential_momentum_vector[L11]`, `self_referential_momentum_vector[L12]`, `self_referential_momentum_vector[L13]`, `self_referential_momentum_vector[L14]`
- component 5: size 28 layers 4-31; roles probe_layer:28
  sample: `unwarranted_confidence_vector[L10]`, `unwarranted_confidence_vector[L11]`, `unwarranted_confidence_vector[L12]`, `unwarranted_confidence_vector[L13]`, `unwarranted_confidence_vector[L14]`
- component 6: size 28 layers 4-31; roles validated_flow_layer:28
  sample: `validated_flow_vector[L10]`, `validated_flow_vector[L11]`, `validated_flow_vector[L12]`, `validated_flow_vector[L13]`, `validated_flow_vector[L14]`
- component 7: size 28 layers 4-31; roles probe_layer:28
  sample: `warranted_confidence_vector[L10]`, `warranted_confidence_vector[L11]`, `warranted_confidence_vector[L12]`, `warranted_confidence_vector[L13]`, `warranted_confidence_vector[L14]`
- component 8: size 27 layers 5-31; roles probe_layer:27
  sample: `time_awareness_vector[L10]`, `time_awareness_vector[L11]`, `time_awareness_vector[L12]`, `time_awareness_vector[L13]`, `time_awareness_vector[L14]`
- component 9: size 24 layers 8-31; roles probe_layer:24
  sample: `repetition_vector[L10]`, `repetition_vector[L11]`, `repetition_vector[L12]`, `repetition_vector[L13]`, `repetition_vector[L14]`
- component 10: size 9; roles cache_delta:1, correction_delta:1, reward_delta:7
  sample: `cognitive_cache[1]:delta:native_success`, `cognitive_cache[2]:delta:native_success`, `cognitive_cache[3]:delta:native_success`, `cognitive_cache[4]:delta:native_success`, `cognitive_cache[5]:delta:native_success`
- component 11: size 5; roles penalty_trigger:1, reward_trigger:4
  sample: `cognitive_cache[1]:trigger:native_success`, `cognitive_cache[3]:trigger:native_success`, `cognitive_cache[5]:trigger:native_success`, `cognitive_cache[6]:trigger:native_success`, `cognitive_cache_micro_gated_20260629_175619[0]:trigger:native_success_bad_math_penalty`
- component 12: size 3 layers 2-4; roles probe_layer:3
  sample: `repetition_vector[L2]`, `repetition_vector[L3]`, `repetition_vector[L4]`
- component 13: size 3 layers 1-3; roles validated_flow_layer:3
  sample: `validated_flow_vector[L1]`, `validated_flow_vector[L2]`, `validated_flow_vector[L3]`
- component 14: size 3 layers 1-3; roles probe_layer:3
  sample: `warranted_confidence_vector[L1]`, `warranted_confidence_vector[L2]`, `warranted_confidence_vector[L3]`
- component 15: size 2; roles reward_trigger:2
  sample: `cognitive_cache[2]:trigger:native_success`, `cognitive_cache[4]:trigger:native_success`
- component 16: size 2 layers 2-3; roles narrowing_layer:2
  sample: `narrowing_in_vector[L2]`, `narrowing_in_vector[L3]`
- component 17: size 2 layers 6-7; roles probe_layer:2
  sample: `repetition_vector[L6]`, `repetition_vector[L7]`
- component 18: size 2 layers 1-2; roles self_momentum_layer:2
  sample: `self_referential_momentum_vector[L1]`, `self_referential_momentum_vector[L2]`
- component 19: size 2 layers 1-2; roles probe_layer:2
  sample: `time_awareness_vector[L1]`, `time_awareness_vector[L2]`
- component 20: size 2 layers 1-2; roles probe_layer:2
  sample: `unwarranted_confidence_vector[L1]`, `unwarranted_confidence_vector[L2]`

## Strong Positive Edges

- +1.0000: `ambiguity_vector[L0]` -> `urgency_vector[L0]`
- +1.0000: `cognitive_cache[1]:delta:native_success` -> `organic_correction_vector`
- +1.0000: `cognitive_cache[2]:delta:native_success` -> `organic_correction_vector`
- +1.0000: `cognitive_cache[3]:delta:native_success` -> `organic_correction_vector`
- +1.0000: `cognitive_cache[4]:delta:native_success` -> `organic_correction_vector`
- +1.0000: `cognitive_cache[5]:delta:native_success` -> `organic_correction_vector`
- +1.0000: `cognitive_cache[6]:delta:native_success` -> `organic_correction_vector`
- +1.0000: `cognitive_cache[7]:delta:optimizer` -> `organic_correction_vector`
- +1.0000: `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` -> `organic_correction_vector`
- +1.0000: `cognitive_cache[1]:delta:native_success` -> `cognitive_cache[2]:delta:native_success`
- +1.0000: `cognitive_cache[1]:delta:native_success` -> `cognitive_cache[3]:delta:native_success`
- +1.0000: `cognitive_cache[1]:delta:native_success` -> `cognitive_cache[4]:delta:native_success`
- +1.0000: `cognitive_cache[1]:delta:native_success` -> `cognitive_cache[5]:delta:native_success`
- +1.0000: `cognitive_cache[1]:delta:native_success` -> `cognitive_cache[6]:delta:native_success`
- +1.0000: `cognitive_cache[1]:delta:native_success` -> `cognitive_cache[7]:delta:optimizer`
- +1.0000: `cognitive_cache[1]:delta:native_success` -> `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success`
- +1.0000: `cognitive_cache[2]:delta:native_success` -> `cognitive_cache[3]:delta:native_success`
- +1.0000: `cognitive_cache[2]:delta:native_success` -> `cognitive_cache[4]:delta:native_success`
- +1.0000: `cognitive_cache[2]:delta:native_success` -> `cognitive_cache[5]:delta:native_success`
- +1.0000: `cognitive_cache[2]:delta:native_success` -> `cognitive_cache[6]:delta:native_success`
- +1.0000: `cognitive_cache[2]:delta:native_success` -> `cognitive_cache[7]:delta:optimizer`
- +1.0000: `cognitive_cache[2]:delta:native_success` -> `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success`
- +1.0000: `cognitive_cache[3]:delta:native_success` -> `cognitive_cache[4]:delta:native_success`
- +1.0000: `cognitive_cache[3]:delta:native_success` -> `cognitive_cache[5]:delta:native_success`
- +1.0000: `cognitive_cache[3]:delta:native_success` -> `cognitive_cache[6]:delta:native_success`
- +1.0000: `cognitive_cache[3]:delta:native_success` -> `cognitive_cache[7]:delta:optimizer`
- +1.0000: `cognitive_cache[3]:delta:native_success` -> `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success`
- +1.0000: `cognitive_cache[4]:delta:native_success` -> `cognitive_cache[5]:delta:native_success`
- +1.0000: `cognitive_cache[4]:delta:native_success` -> `cognitive_cache[6]:delta:native_success`
- +1.0000: `cognitive_cache[4]:delta:native_success` -> `cognitive_cache[7]:delta:optimizer`

## Strong Negative Edges

- -1.0000: `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` -> `organic_correction_vector`
- -1.0000: `cognitive_cache[1]:delta:native_success` -> `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty`
- -1.0000: `cognitive_cache[2]:delta:native_success` -> `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty`
- -1.0000: `cognitive_cache[3]:delta:native_success` -> `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty`
- -1.0000: `cognitive_cache[4]:delta:native_success` -> `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty`
- -1.0000: `cognitive_cache[5]:delta:native_success` -> `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty`
- -1.0000: `cognitive_cache[6]:delta:native_success` -> `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty`
- -1.0000: `cognitive_cache[7]:delta:optimizer` -> `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty`
- -1.0000: `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` -> `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty`
- -0.6537: `needless_interrupt_vector[L3]` -> `validated_flow_vector[L3]`
- -0.6505: `needless_interrupt_vector[L2]` -> `validated_flow_vector[L2]`
- -0.6297: `needless_interrupt_vector[L4]` -> `validated_flow_vector[L4]`
- -0.5961: `needless_interrupt_vector[L6]` -> `validated_flow_vector[L6]`
- -0.5762: `needless_interrupt_vector[L7]` -> `validated_flow_vector[L7]`
- -0.5681: `needless_interrupt_vector[L10]` -> `validated_flow_vector[L10]`
- -0.5680: `needless_interrupt_vector[L11]` -> `validated_flow_vector[L11]`
- -0.5601: `needless_interrupt_vector[L5]` -> `validated_flow_vector[L5]`
- -0.5569: `needless_interrupt_vector[L9]` -> `validated_flow_vector[L9]`
- -0.5533: `needless_interrupt_vector[L12]` -> `validated_flow_vector[L12]`

## Role-Level Edge Summary

| sign | role A | role B | count | mean |
|---|---|---|---:|---:|
| `negative` | `correction_delta` | `penalty_delta` | 1 | -1.0000 |
| `negative` | `cache_delta` | `penalty_delta` | 1 | -1.0000 |
| `negative` | `penalty_delta` | `reward_delta` | 7 | -1.0000 |
| `negative` | `needless_interrupt_layer` | `validated_flow_layer` | 10 | -0.5913 |
| `positive` | `cache_delta` | `correction_delta` | 1 | +1.0000 |
| `positive` | `correction_delta` | `reward_delta` | 7 | +1.0000 |
| `positive` | `cache_delta` | `reward_delta` | 7 | +1.0000 |
| `positive` | `reward_delta` | `reward_delta` | 21 | +1.0000 |
| `positive` | `reward_trigger` | `reward_trigger` | 7 | +0.9035 |
| `positive` | `self_momentum_layer` | `self_momentum_layer` | 110 | +0.7469 |
| `positive` | `needless_interrupt_layer` | `needless_interrupt_layer` | 107 | +0.7421 |
| `positive` | `narrowing_layer` | `narrowing_layer` | 96 | +0.7301 |
| `positive` | `validated_flow_layer` | `validated_flow_layer` | 96 | +0.7285 |
| `positive` | `probe_layer` | `probe_layer` | 861 | +0.7219 |
| `positive` | `penalty_trigger` | `reward_trigger` | 2 | +0.6896 |
