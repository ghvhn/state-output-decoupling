# Spawn profile generation memory guard

The pasted `:spawn baseModel replace 3` failure requested a 50.35 GiB allocation
inside attention on a 15.92 GiB GPU. The trace identifies generation after
backfill; it does not establish that backfill or the field caused the allocation.
The spawn writer previously included the full command documentation plus a
verbose state report in its prompt, making oversized context a likely contributor.

Profile authoring now uses a command index and bounded 12,000-character syntax
reference, compact current values/probe names, and an automatic 120-reply
backfill instead of the entire archive. The updated prompt version replaces
older saved spawn templates through the existing version check.

Agentic generation checks input length plus the next output chunk before
attention runs. The default operational limit is 8,192 tokens, bounded also by
the model's position limit. `TDA_MAX_CONTEXT_TOKENS` overrides the operational
limit for hardware with different capacity. This is a context limit, not a VRAM
guarantee. Oversized input fails with a measured token count and no silent
truncation. Tool/memory-expanded context is checked on every chunk too.

Restart the shell so its loaded functions match the edited files, then retry
`:spawn baseModel replace 3`. If the profile needs generating, follow the shell's
instruction to repeat the spawn command once generation completes.

48 targeted checks passed, including context boundaries, preserved inputs,
model position limits, command visibility, field regressions and target learning.
The actual GPU spawn retry remains unverified.
