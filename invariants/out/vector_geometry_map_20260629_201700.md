# Vector Geometry Map

This map is offline and correlational. It does not prove causality; it shows which saved directions currently align, oppose, or look unrelated.

## Inventory

- vector entries extracted: `380`
- artifact groups: `22`
- skipped `.pt` files: `2`

| artifact | count | kinds | layers | norm mean |
|---|---:|---|---|---:|
| `ambiguity_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 9.8686 |
| `clouds_language_bridge` | 2 | `dict_tensor` | `` | 148.1006 |
| `clouds_self_steering_isolated` | 2 | `dict_tensor` | `` | 149.1053 |
| `cognitive_cache:cache_delta` | 8 | `cache_delta` | `` | 12.7929 |
| `cognitive_cache:cache_trigger` | 8 | `cache_trigger` | `` | 50.8037 |
| `cognitive_cache_micro_egg_20260629_192151:cache_delta` | 1 | `cache_delta` | `` | 11.4678 |
| `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | 1 | `cache_trigger` | `` | 29.6299 |
| `cognitive_cache_micro_gated_20260629_175619:cache_delta` | 1 | `cache_delta` | `` | 11.4678 |
| `cognitive_cache_micro_gated_20260629_175619:cache_trigger` | 1 | `cache_trigger` | `` | 54.4677 |
| `disagreement_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 8.8585 |
| `narrowing_in_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 9.8775 |
| `needless_interrupt_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 9.5271 |
| `organic_correction_vector` | 1 | `tensor_vector` | `` | 7.6452 |
| `repetition_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 6.6092 |
| `self_experience` | 1 | `dict_tensor` | `` | 1.0000 |
| `self_referential_momentum_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 10.1289 |
| `self_steering_isolated` | 2 | `dict_tensor` | `` | 16.6923 |
| `time_awareness_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 5.2553 |
| `unwarranted_confidence_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 9.1384 |
| `urgency_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 10.2415 |
| `validated_flow_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 9.0535 |
| `warranted_confidence_vector` | 32 | `layer_vector` | `0,1,2,3,4,5,6,7,...` | 8.0402 |

## Top Artifact Correlations

| mean cosine | artifact A | artifact B | strongest entry pair |
|---:|---|---|---|
| +1.0000 | `cognitive_cache_micro_egg_20260629_192151:cache_delta` | `organic_correction_vector` | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` vs `organic_correction_vector` (+1.0000) |
| +0.8731 | `cognitive_cache:cache_delta` | `organic_correction_vector` | `cognitive_cache[1]:delta:native_success` vs `organic_correction_vector` (+1.0000) |
| +0.8731 | `cognitive_cache:cache_delta` | `cognitive_cache_micro_egg_20260629_192151:cache_delta` | `cognitive_cache[1]:delta:native_success` vs `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` (+1.0000) |
| +0.7771 | `ambiguity_vector` | `urgency_vector` | `ambiguity_vector[L0]` vs `urgency_vector[L0]` (+1.0000) |
| +0.4802 | `disagreement_vector` | `urgency_vector` | `disagreement_vector[L25]` vs `urgency_vector[L25]` (+0.5525) |
| +0.4133 | `cognitive_cache:cache_trigger` | `cognitive_cache_micro_gated_20260629_175619:cache_trigger` | `cognitive_cache[1]:trigger:native_success` vs `cognitive_cache_micro_gated_20260629_175619[0]:trigger:native_success_bad_math_penalty` (+0.6904) |
| +0.3930 | `ambiguity_vector` | `disagreement_vector` | `ambiguity_vector[L1]` vs `disagreement_vector[L1]` (+0.4896) |
| +0.3394 | `disagreement_vector` | `repetition_vector` | `disagreement_vector[L1]` vs `repetition_vector[L1]` (+0.5728) |
| +0.3241 | `cognitive_cache:cache_trigger` | `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | `cognitive_cache[7]:trigger:optimizer` vs `cognitive_cache_micro_egg_20260629_192151[0]:trigger:native_success` (+0.5162) |
| +0.2555 | `ambiguity_vector` | `repetition_vector` | `ambiguity_vector[L12]` vs `repetition_vector[L12]` (+0.3390) |
| +0.2519 | `repetition_vector` | `urgency_vector` | `repetition_vector[L12]` vs `urgency_vector[L12]` (+0.3522) |
| +0.2387 | `cognitive_cache_micro_egg_20260629_192151:cache_delta` | `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` vs `cognitive_cache_micro_egg_20260629_192151[0]:trigger:native_success` (+0.2387) |
| +0.2387 | `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | `organic_correction_vector` | `cognitive_cache_micro_egg_20260629_192151[0]:trigger:native_success` vs `organic_correction_vector` (+0.2387) |
| +0.2345 | `unwarranted_confidence_vector` | `warranted_confidence_vector` | `unwarranted_confidence_vector[L24]` vs `warranted_confidence_vector[L24]` (+0.3111) |
| +0.2130 | `cognitive_cache:cache_delta` | `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | `cognitive_cache[1]:delta:native_success` vs `cognitive_cache_micro_egg_20260629_192151[0]:trigger:native_success` (+0.2387) |
| +0.1934 | `self_referential_momentum_vector` | `unwarranted_confidence_vector` | `self_referential_momentum_vector[L0]` vs `unwarranted_confidence_vector[L0]` (+0.3738) |
| +0.1314 | `narrowing_in_vector` | `self_referential_momentum_vector` | `narrowing_in_vector[L31]` vs `self_referential_momentum_vector[L31]` (+0.2352) |
| +0.1157 | `cognitive_cache_micro_gated_20260629_175619:cache_trigger` | `organic_correction_vector` | `cognitive_cache_micro_gated_20260629_175619[0]:trigger:native_success_bad_math_penalty` vs `organic_correction_vector` (+0.1157) |
| +0.1157 | `cognitive_cache_micro_egg_20260629_192151:cache_delta` | `cognitive_cache_micro_gated_20260629_175619:cache_trigger` | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` vs `cognitive_cache_micro_gated_20260629_175619[0]:trigger:native_success_bad_math_penalty` (+0.1157) |
| +0.1028 | `cognitive_cache:cache_delta` | `cognitive_cache_micro_gated_20260629_175619:cache_trigger` | `cognitive_cache[1]:delta:native_success` vs `cognitive_cache_micro_gated_20260629_175619[0]:trigger:native_success_bad_math_penalty` (+0.1157) |

## Top Artifact Anti-Correlations

| mean cosine | artifact A | artifact B | strongest entry pair |
|---:|---|---|---|
| -1.0000 | `cognitive_cache_micro_gated_20260629_175619:cache_delta` | `organic_correction_vector` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` vs `organic_correction_vector` (-1.0000) |
| -1.0000 | `cognitive_cache_micro_egg_20260629_192151:cache_delta` | `cognitive_cache_micro_gated_20260629_175619:cache_delta` | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` vs `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` (-1.0000) |
| -0.8731 | `cognitive_cache:cache_delta` | `cognitive_cache_micro_gated_20260629_175619:cache_delta` | `cognitive_cache[1]:delta:native_success` vs `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` (-1.0000) |
| -0.4493 | `needless_interrupt_vector` | `validated_flow_vector` | `needless_interrupt_vector[L3]` vs `validated_flow_vector[L3]` (-0.6537) |
| -0.2387 | `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | `cognitive_cache_micro_gated_20260629_175619:cache_delta` | `cognitive_cache_micro_egg_20260629_192151[0]:trigger:native_success` vs `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` (-0.2387) |
| -0.1406 | `validated_flow_vector` | `warranted_confidence_vector` | `validated_flow_vector[L28]` vs `warranted_confidence_vector[L28]` (-0.2620) |
| -0.1364 | `unwarranted_confidence_vector` | `validated_flow_vector` | `unwarranted_confidence_vector[L31]` vs `validated_flow_vector[L31]` (-0.2577) |
| -0.1157 | `cognitive_cache_micro_gated_20260629_175619:cache_delta` | `cognitive_cache_micro_gated_20260629_175619:cache_trigger` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` vs `cognitive_cache_micro_gated_20260629_175619[0]:trigger:native_success_bad_math_penalty` (-0.1157) |
| -0.0976 | `cognitive_cache:cache_trigger` | `cognitive_cache_micro_gated_20260629_175619:cache_delta` | `cognitive_cache[7]:trigger:optimizer` vs `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` (-0.2064) |
| -0.0702 | `ambiguity_vector` | `self_referential_momentum_vector` | `ambiguity_vector[L27]` vs `self_referential_momentum_vector[L27]` (-0.1370) |
| -0.0670 | `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | `repetition_vector` | `cognitive_cache_micro_egg_20260629_192151[0]:trigger:native_success` vs `repetition_vector[L22]` (-0.1432) |
| -0.0618 | `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | `time_awareness_vector` | `cognitive_cache_micro_egg_20260629_192151[0]:trigger:native_success` vs `time_awareness_vector[L22]` (-0.1147) |
| -0.0557 | `ambiguity_vector` | `needless_interrupt_vector` | `ambiguity_vector[L15]` vs `needless_interrupt_vector[L15]` (-0.1031) |
| -0.0555 | `cognitive_cache_micro_egg_20260629_192151:cache_delta` | `repetition_vector` | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` vs `repetition_vector[L11]` (-0.1187) |
| -0.0555 | `organic_correction_vector` | `repetition_vector` | `organic_correction_vector` vs `repetition_vector[L11]` (-0.1188) |
| -0.0491 | `cognitive_cache:cache_delta` | `repetition_vector` | `cognitive_cache[1]:delta:native_success` vs `repetition_vector[L11]` (-0.1187) |
| -0.0477 | `cognitive_cache_micro_gated_20260629_175619:cache_delta` | `self_experience` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` vs `self_experience:direction` (-0.0477) |
| -0.0469 | `organic_correction_vector` | `time_awareness_vector` | `organic_correction_vector` vs `time_awareness_vector[L12]` (-0.1361) |
| -0.0469 | `cognitive_cache_micro_egg_20260629_192151:cache_delta` | `time_awareness_vector` | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` vs `time_awareness_vector[L12]` (-0.1361) |
| -0.0438 | `cognitive_cache_micro_egg_20260629_192151:cache_delta` | `unwarranted_confidence_vector` | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` vs `unwarranted_confidence_vector[L17]` (-0.1049) |

## Strongest Entry-Level Alignments

| cosine | entry A | entry B |
|---:|---|---|
| +1.0000 | `ambiguity_vector[L0]` | `urgency_vector[L0]` |
| +1.0000 | `cognitive_cache[1]:delta:native_success` | `organic_correction_vector` |
| +1.0000 | `cognitive_cache[2]:delta:native_success` | `organic_correction_vector` |
| +1.0000 | `cognitive_cache[3]:delta:native_success` | `organic_correction_vector` |
| +1.0000 | `cognitive_cache[4]:delta:native_success` | `organic_correction_vector` |
| +1.0000 | `cognitive_cache[5]:delta:native_success` | `organic_correction_vector` |
| +1.0000 | `cognitive_cache[6]:delta:native_success` | `organic_correction_vector` |
| +1.0000 | `cognitive_cache[7]:delta:optimizer` | `organic_correction_vector` |
| +1.0000 | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` | `organic_correction_vector` |
| +1.0000 | `cognitive_cache[1]:delta:native_success` | `cognitive_cache[2]:delta:native_success` |
| +1.0000 | `cognitive_cache[1]:delta:native_success` | `cognitive_cache[3]:delta:native_success` |
| +1.0000 | `cognitive_cache[1]:delta:native_success` | `cognitive_cache[4]:delta:native_success` |
| +1.0000 | `cognitive_cache[1]:delta:native_success` | `cognitive_cache[5]:delta:native_success` |
| +1.0000 | `cognitive_cache[1]:delta:native_success` | `cognitive_cache[6]:delta:native_success` |
| +1.0000 | `cognitive_cache[1]:delta:native_success` | `cognitive_cache[7]:delta:optimizer` |
| +1.0000 | `cognitive_cache[1]:delta:native_success` | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` |
| +1.0000 | `cognitive_cache[2]:delta:native_success` | `cognitive_cache[3]:delta:native_success` |
| +1.0000 | `cognitive_cache[2]:delta:native_success` | `cognitive_cache[4]:delta:native_success` |
| +1.0000 | `cognitive_cache[2]:delta:native_success` | `cognitive_cache[5]:delta:native_success` |
| +1.0000 | `cognitive_cache[2]:delta:native_success` | `cognitive_cache[6]:delta:native_success` |

## Strongest Entry-Level Oppositions

| cosine | entry A | entry B |
|---:|---|---|
| -1.0000 | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` | `organic_correction_vector` |
| -1.0000 | `cognitive_cache[1]:delta:native_success` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` |
| -1.0000 | `cognitive_cache[2]:delta:native_success` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` |
| -1.0000 | `cognitive_cache[3]:delta:native_success` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` |
| -1.0000 | `cognitive_cache[4]:delta:native_success` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` |
| -1.0000 | `cognitive_cache[5]:delta:native_success` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` |
| -1.0000 | `cognitive_cache[6]:delta:native_success` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` |
| -1.0000 | `cognitive_cache[7]:delta:optimizer` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` |
| -1.0000 | `cognitive_cache_micro_egg_20260629_192151[0]:delta:native_success` | `cognitive_cache_micro_gated_20260629_175619[0]:delta:native_success_bad_math_penalty` |
| -0.6537 | `needless_interrupt_vector[L3]` | `validated_flow_vector[L3]` |
| -0.6505 | `needless_interrupt_vector[L2]` | `validated_flow_vector[L2]` |
| -0.6297 | `needless_interrupt_vector[L4]` | `validated_flow_vector[L4]` |
| -0.5961 | `needless_interrupt_vector[L6]` | `validated_flow_vector[L6]` |
| -0.5762 | `needless_interrupt_vector[L7]` | `validated_flow_vector[L7]` |
| -0.5681 | `needless_interrupt_vector[L10]` | `validated_flow_vector[L10]` |
| -0.5680 | `needless_interrupt_vector[L11]` | `validated_flow_vector[L11]` |
| -0.5601 | `needless_interrupt_vector[L5]` | `validated_flow_vector[L5]` |
| -0.5569 | `needless_interrupt_vector[L9]` | `validated_flow_vector[L9]` |
| -0.5533 | `needless_interrupt_vector[L12]` | `validated_flow_vector[L12]` |
| -0.5465 | `needless_interrupt_vector[L8]` | `validated_flow_vector[L8]` |

## Skipped Files

| file | reason |
|---|---|
| `invariants\out\lens_native.pt` | dict contained no vector-like tensors |
| `invariants\out\trajs_self_steering_isolated.pt` | dict contained no vector-like tensors |
