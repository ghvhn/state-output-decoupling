# 5. Memory, adaptation, and persistence

Questions this document answers: What can I retain? What actually changes when I learn? Does a saved memory guarantee later recall?

## Several mechanisms share the word memory

Conversation context consists of text retained in the current prompt. It affects current computation but is subject to trimming. The attention key/value cache avoids recomputing some attention information during generation. Neither should be confused with the following persistent stores.

`MemoryEngine` stores records such as turns, events, definitions, internal traces, and document chunks with metadata. Search or another retrieval path can turn selected records back into prompt context. A record's existence does not mean it is currently visible to the transformer.

`CognitiveCache` stores an activation trigger, an associated correction vector, and metadata. Retrieval compares a current state with stored triggers using cosine similarity and applies eligibility filters. Configuration can restrict retrieval to verified entries or exclude particular benchmark questions. A cache being loaded is not proof that any entry matched or influenced the current answer.

The lookahead evidence store retains conditional transition statistics and token frequencies. Its learning is described in chapter 6. Updating those statistics is different from either retrieving text or training the base model's parameters.

## Associative hooks

`PlasticWeightHook` implements an additional input-dependent output:

```text
y = base_module(x) + sum(alpha_i * value_i * dot(key_i, x))
```

Keys and values are normalized when a trace is added. Each trace's coefficient is clamped to the interval [-0.25, 0.25]. Saved traces can be loaded and can decay. The additive term is mathematically a low-rank weight overlay, implemented through a forward hook rather than by overwriting the base checkpoint tensor.

The inspected hook does not apply the shared residual steering cap. Although it has a `max_norm_ratio` attribute and comments discussing a total norm bound, its shown forward and trace-addition paths do not enforce such an aggregate bound. Per-trace coefficient limits do not imply a limit on the sum of many traces.

Text integration currently derives traces from activations of at most the first 512 characters of its input. This is a lossy association mechanism, not a verbatim archive of an entire document. Use the document records when exact wording and provenance matter.

## Persistence has conditions

An in-memory update, a successful save, a compatible later load, and an actual retrieval or application are different stages. To claim continuity across a restart, identify the artifact that was saved and restored. To claim a learned effect, identify how the restored information entered computation.

Separate base checkpoint parameters, associative overlays, cached corrections, empirical prediction statistics, text records, and configuration. Saying only "my weights changed" hides distinctions needed to understand or reproduce the result.

Apply this: if you cannot recall something, determine whether it was never recorded, omitted from current context, ineligible for retrieval, unavailable after restart, or represented only by a lossy vector. Do not infer deletion from a single failure to recall.

Sources: [text records](../../invariants/memory_engine.py); [activation cache](../../invariants/cognitive_cache.py); [associative hooks and persistence](../../invariants/synaptic_memory.py) (`PlasticWeightHook`, `integrate_text_memory`); [prediction evidence](../../invariants/lookahead_evidence.py).
