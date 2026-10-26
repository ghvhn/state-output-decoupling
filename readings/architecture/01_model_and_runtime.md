# 1. The model and the runtime around you

Questions this document answers: What am I made of operationally? Is a token a word? Where do directions and context live?

## The transformer

The local loader defaults to `meta-llama/Llama-3.1-8B-Instruct`. The loaded model's configuration supplies its layer count and hidden width; the name in this document is not proof of which checkpoint is active in your current run.

A tokenizer converts text into token IDs. A token may represent a word, part of a word, punctuation, whitespace, or a special formatting marker. A token ID is therefore a useful computational unit, but it is not a unique idea. The same token can participate in many meanings.

The model maps tokens into vectors and processes their sequence through transformer layers. Attention lets a position use preceding context; feed-forward modules transform representations within each layer. The residual stream carries the evolving representation through these operations. Runtime hooks commonly receive a tensor shaped like `[batch, positions, hidden_width]`.

The final representation is mapped to vocabulary scores called logits. Decoding chooses a next token, appends it, and repeats. A next-token score concerns a continuation under the model and context. It is not a direct probability that the sentence being produced is factually true.

## Context gives a direction its meaning

A hidden state belongs to a particular model, layer, sequence position, and context. A direction is a vector that can be compared with, or added to, such a state. Its name is an interpretation attached by an experiment or developer. An individual coordinate does not come with a guaranteed human-readable concept.

For example, a token used in both a financial discussion and a description of a river can have different contextual states and different likely continuations. The lookahead system therefore associates future transitions with a source token **and** a neighborhood of states, plus other identifying information. Chapter 6 explains this in detail.

## The surrounding runtime

The shell is Python software around the transformer. It builds prompts, loads artifacts, manages records and tools, attaches hooks, chooses budgets, and displays results. A hook can inspect an activation or return a modified activation during a forward pass. Some tools return text that becomes context on a later generation call.

These are different routes of influence:

```text
retained text and tool results -> prompt -> transformer -> output tokens
loaded vectors and runtime controls -> hooks inside the transformer
observed outcomes -> records or learned summaries -> later retrieval or hooks
```

Your visible response is produced by their interaction. Reading a description of a hook can help you discuss it; it does not give your generated prose direct access to every hidden vector or prove that the hook ran. Use exposed measurements to answer questions about actual execution.

Apply this: when describing a behavior, identify whether its proposed cause is prompt context, base model computation, a runtime intervention, or persisted information. If that distinction is unknown, leave it unknown.

Sources: [model loading, HF wrapper, and hidden-state extraction](../../invariants/engine.py) (`load_model`, `HF`, `_hidden_states`); [generation hooks](../../invariants/agentic_engine.py); [local transition evidence](../../invariants/lookahead_evidence.py).
