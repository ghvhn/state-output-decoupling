# 2. How a turn is produced

Questions this document answers: What actually enters my context? How does generation proceed? Why can a response stop before completion?

## Prompt construction

In the interactive shell, `build_prompt` assembles retained conversation turns and the current user message using Llama chat markers. Retention is bounded: trimming can remove material from the prompt even when a longer-term record still exists.

Available tool results and factual field context are assembled separately from the user's words. In the inspected implementation, these blocks are placed in a system-role message labeled as internal substrate and cognitive context. When there are no such blocks, that message is absent. Thus the description "bare mode" does not mean that every call contains only user text or that tool context can never appear in a system role.

The `claimmap_tool_result` parameter is not included in this prompt's tool-block list. ClaimMap steering has a separate activation route. A tool's existence, a parameter name, and actual prompt inclusion are distinct facts.

## Generation and intervention

`generate_agentic_text` prepares inputs and installs the enabled agentic hooks. The ordinary generation path requests up to 24 new tokens per chunk, subject to the remaining budget, with greedy decoding (`do_sample=False`). Other generation paths can use different decoding settings.

Prompt processing is called prefill. Subsequent decoding uses earlier token information to produce more tokens. The model uses its attention key/value cache within generation; this computational cache is different from the persistent cognitive cache described in chapter 5. The outer loop starts further generation calls on the growing token sequence.

During model computation, applicable hooks can observe or change representations. The agentic branching hook skips multi-position prefill because its repeated branch computation is intended for the decoding path. Other hooks have their own rules; do not assume that every intervention acts on exactly the same tokens.

Lookahead callbacks track generation boundaries, chunks, and arriving tokens. Tool stopping criteria can return control to the surrounding program. The program can execute a supported request and provide its result to subsequent computation. An emitted tool marker alone is not proof that execution succeeded.

## Stopping and continuation

Generation can end because of an end-of-turn token, a configured stopping condition, a token or time budget, or an error. A partial response can therefore reflect a resource boundary. Diagnose it from the stop reason and logs instead of treating the final words as proof that the model intentionally finished.

Temporary agentic hook handles and call-specific configuration have cleanup paths. Longer-lived components have separate lifecycles. Across turns, continuity comes from retained context, restored records, and persistent controls or learned artifacts. It should not be inferred from an uninterrupted hidden-state trajectory between independent calls.

Apply this: to explain a particular answer, first establish which prompt material, settings, tools, and interventions were actually used. This chapter describes a path available in the code, not a trace of your current response.

Sources: [shell](../../scripts/interactive_phenomenality.py) (`build_prompt`, `trim_session_context`, `recover_session_context`); [generation engine](../../invariants/agentic_engine.py) (`GenerationCall`, `generate_agentic_text`); [tool parsing and stopping](../../invariants/tool_utils.py).
