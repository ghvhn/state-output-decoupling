# 8. Documents, tools, and rereading

Questions this document answers: What does it mean to have read a document? How can a document answer a current question? Why should unchanged text be read again?

## From a file to working context

The document engine reads text, divides it into chunks, and records source information. Provenance includes the path, content hash, chunk position, and the supplied reason for reading. A staged chunk can then be formatted as a document tool result and included in a generation prompt.

These are separate states:

1. The file exists.
2. Its text has been ingested into records.
3. A particular chunk has been presented in context.
4. The current question has been considered using that material.
5. The whole document has been covered during the relevant pass.

Ingestion can store all chunks before they have all been presented. A read-progress event records presentation, not comprehension. Even complete presentation is not a guarantee of correct understanding or permanent recall.

## Read in relation to a working question

Use this as a reading practice: state the question you are investigating, inspect the next available section, and extract the claim that helps answer it. Keep the source and relevant qualification. If the section does not answer the question, continue rather than inventing an answer.

Finish the selected document even after finding a promising passage. Later sections may define a term differently, limit applicability, or contradict an early interpretation. At completion, record a concise answer and what remains unresolved. This practice can guide thought-directed reading without requiring a transcript of internal deliberation.

A relevance score helps choose what to inspect; it does not prove that an excerpt answers the question. An ordered coverage pass is useful when relevance-based selection could miss qualifications. Track the two purposes separately: finding relevant evidence and covering all sections.

## Reread when context changes

The content hash answers whether the text changed. It does not answer whether your question, task, assumptions, or runtime configuration changed. Reading this chapter to understand persistence is different from reading it to investigate an incomplete document pass, even if every byte is identical.

A sound reading record distinguishes document identity, content revision, reading pass, current purpose, and coverage. Preserve earlier passes as history while allowing the same chunks to be considered again. If the running tool restores old read progress, use its available reread or restart controls; do not claim that merely loading the same file necessarily presented everything again.

This is a reading protocol and an explanation of the ingestion boundary. The current shell's implementation and help determine which continuation and reread controls are available.

## Tool results are observations with provenance

An emitted request, an executed action, and a returned result are different events. Use the actual result to say what happened. A document's author is not currently replying to you, and text in a tool result is not automatically the user's speech. Preserve these distinctions when carrying evidence into later answers.

Apply this: after a pass, report its source, question, coverage, supported answer, and remaining uncertainty. On a later pass, identify what changed in the question or evidence rather than assuming the earlier interpretation must still fit.

Sources: [chunking, ingestion, staging, and progress](../../invariants/document_engine.py); [document records](../../invariants/memory_engine.py); [prompt and document command handling](../../scripts/interactive_phenomenality.py).
