"""Fail before oversized attention allocations without silently dropping context."""
import os


def check_context_budget(model, inputs, reserve=0):
    limit = int(os.environ.get('TDA_MAX_CONTEXT_TOKENS', '8192'))
    if limit < 1:
        raise ValueError('TDA_MAX_CONTEXT_TOKENS must be positive')
    configured = getattr(getattr(model, 'config', None), 'max_position_embeddings', None)
    if isinstance(configured, int) and configured > 0:
        limit = min(limit, configured)
    size = int(inputs['input_ids'].shape[-1])
    if size + reserve > limit:
        raise ValueError(f'Generation context needs {size} input + {reserve} output tokens; '
                         f'limit is {limit}. Shorten the prompt/reference or set '
                         'TDA_MAX_CONTEXT_TOKENS for the available hardware. No context was discarded.')
    return size
