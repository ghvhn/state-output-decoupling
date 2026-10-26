"""Candidate task axes from matched native chat states.

The teacher text is used only to mint activation directions. It is never
added to the real probe-definition input. Geometry tests cannot establish
that these candidate directions will produce the intended live behavior.
"""
import math

import torch


TASK_VERSION = 'definition-task-assistant-and-question-v2'
DEFINITION_TASK_GAINS = {'contrast': .04, 'abstraction': .04,
                         'naming': .025, 'verbosity': -.035, 'question': .08,
                         'operation': 0.0}

_SENSOR = 'p_000000000000000000000000'
_QUESTION_SENSORS = (_SENSOR, 'p_111111111111111111111111')
_ROUTES = (
    f'[Probe {_SENSOR}]\n'
    'Memory 0: position=-0.80; input="Which route should we take?"; reply="Take the first open corridor."\n'
    'Memory 1: position=-0.25; input="Which route should we take?"; reply="Use whichever corridor is available first."\n'
    'Memory 2: position=+0.35; input="Which route should we take?"; reply="I will compare the two open corridors before selecting one."\n'
    'Memory 3: position=+0.85; input="Which route should we take?"; reply="I will examine the available routes and choose after comparing them."'
)
_ASSEMBLY = (
    f'[Probe {_SENSOR}]\n'
    'Memory 0: position=-0.75; input="Can we assemble this?"; reply="I will start putting the pieces together."\n'
    'Memory 1: position=-0.20; input="Can we assemble this?"; reply="I will begin assembling it now."\n'
    'Memory 2: position=+0.30; input="Can we assemble this?"; reply="I will check which pieces fit, then start assembling."\n'
    'Memory 3: position=+0.90; input="Can we assemble this?"; reply="I will verify the parts and their fit before beginning assembly."'
)

_ROUTE_CONTRAST = ('Lower positions select the first workable option; '
                   'higher positions compare alternatives before selecting.')
_ASSEMBLY_CONTRAST = ('Lower positions begin acting immediately; '
                      'higher positions check the prerequisites before beginning.')

# Each tuple is (identical user evidence, positive completion, negative completion).
# Positive verbosity is deliberately verbose, so its default gain is negative.
_TASK_CASES = {
    'contrast': (
        (_ROUTES, _ROUTE_CONTRAST,
         'Memory 0 takes the first open corridor. Memory 1 uses the first available corridor. '
         'Memory 2 compares two corridors. Memory 3 examines the routes and then chooses.'),
        (_ASSEMBLY, _ASSEMBLY_CONTRAST,
         'Memory 0 starts putting pieces together. Memory 1 begins assembly. '
         'Memory 2 checks which pieces fit. Memory 3 verifies the parts before assembly.'),
    ),
    'abstraction': (
        (_ROUTES,
         'This dimension is comparison before commitment: lower positions select the first workable option, '
         'while higher positions evaluate alternatives before selecting.',
         'At the lower end the speaker enters an open corridor immediately. '
         'At the higher end the speaker looks at the available corridors before entering one.'),
        (_ASSEMBLY,
         'This dimension is preparation before action: lower positions begin immediately, '
         'while higher positions verify the prerequisites before beginning.',
         'At the lower end the speaker starts putting pieces together immediately. '
         'At the higher end the speaker checks the parts and their fit before assembly.'),
    ),
    'naming': (
        (_ROUTES,
         _ROUTE_CONTRAST + ' A provisional term for this distinction is "option comparison".',
         _ROUTE_CONTRAST),
        (_ASSEMBLY,
         _ASSEMBLY_CONTRAST + ' A provisional term for this distinction is "preparatory checking".',
         _ASSEMBLY_CONTRAST),
    ),
    'verbosity': (
        (_ROUTES,
         _ROUTE_CONTRAST + ' Put another way, the lower positions describe selecting a workable '
         'option as soon as it is available, whereas the higher positions describe considering '
         'alternatives and selecting only after comparing them. The difference concerns comparison '
         'before selection. To restate that difference, immediate selection appears at the lower '
         'positions, and comparison before selection appears at the higher positions.',
         _ROUTE_CONTRAST),
        (_ASSEMBLY,
         _ASSEMBLY_CONTRAST + ' Put another way, the lower positions describe beginning the action '
         'immediately, whereas the higher positions describe checking the prerequisites and then '
         'beginning. The difference concerns checking before action. To restate that difference, '
         'immediate action appears at the lower positions, and checking before action appears at '
         'the higher positions.',
         _ASSEMBLY_CONTRAST),
    ),
}


def _unit(vector, axis, layer):
    vector = vector.detach().float()
    norm = vector.norm()
    if not torch.isfinite(vector).all() or not torch.isfinite(norm):
        raise ValueError(f'Definition task axis {axis} has non-finite values at layer {layer}')
    if norm.item() <= 1e-8:
        raise ValueError(f'Definition task axis {axis} has a zero or cancelled contrast at layer {layer}')
    return vector / norm


def _requested_layers(layers):
    try:
        requested_indices = set()
        for layer in layers:
            numeric = float(layer)
            index = int(layer)
            if isinstance(layer, bool) or not math.isfinite(numeric) or numeric != index:
                raise ValueError('Non-integer definition task layer')
            requested_indices.add(index)
        requested = tuple(sorted(requested_indices))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Definition task layers must be finite integer indices') from exc
    if not requested or requested[0] < 0:
        raise ValueError('Definition task minting requires nonnegative requested layers')
    return requested


def _native_states(model, messages, layers, axis, *, add_generation_prompt):
    from invariants.engine import _inputs, _hidden_states

    formatted = model.tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    if not isinstance(formatted, str) or not formatted:
        raise ValueError('Definition task minting requires a native text chat template')
    inputs = _inputs(model, formatted, pre_formatted=True)
    hidden = _hidden_states(model, inputs['input_ids'], inputs.get('attention_mask'))
    if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[1] == 0:
        raise ValueError(f'Definition task axis {axis} requires per-layer token hidden states')
    result = {}
    for layer in layers:
        if layer >= hidden.shape[0]:
            raise ValueError(f'Definition task axis {axis} is missing requested layer {layer}')
        # Clone only the requested native endpoints, retaining no token cloud.
        state = hidden[layer, -1].detach().float().clone()
        if state.numel() == 0 or not torch.isfinite(state).all():
            raise ValueError(f'Definition task axis {axis} has invalid native state at layer {layer}')
        result[layer] = state
    return result


def _completion_states(model, user_evidence, completion, layers, axis):
    messages = [{'role': 'user', 'content': user_evidence},
                {'role': 'assistant', 'content': completion}]
    return _native_states(model, messages, layers, axis, add_generation_prompt=False)


@torch.no_grad()
def mint_definition_question_axis(model, layers):
    """Contrast an opaque subject question with its matched declaration.

    Both native USER inputs end at the same assistant decision position. The
    opaque subject is identical within each pair; neither actual probe labels
    nor measured memories participate in this candidate task direction.
    """
    requested = _requested_layers(layers)
    differences = {layer: [] for layer in requested}
    for sensor in _QUESTION_SENSORS:
        positive = _native_states(
            model, [{'role': 'user', 'content': f'What is {sensor}?'}],
            requested, 'question', add_generation_prompt=True)
        negative = _native_states(
            model, [{'role': 'user', 'content': f'Here is {sensor}.'}],
            requested, 'question', add_generation_prompt=True)
        for layer in requested:
            if positive[layer].shape != negative[layer].shape:
                raise ValueError(f'Definition task axis question has incompatible state widths at layer {layer}')
            differences[layer].append(_unit(positive[layer] - negative[layer], 'question', layer))
    return {layer: _unit(torch.stack(values).mean(dim=0), 'question', layer)
            for layer, values in differences.items()}


@torch.no_grad()
def mint_definition_task_axes(model, layers):
    """Mint five separate candidate axes on every requested layer.

    Each paired completion is contrasted and normalized independently before
    averaging its two case directions. The final average is normalized again;
    a missing, non-finite, or cancelled direction fails rather than falling back
    to an unrelated layer. Caller owns caching and the signed injection gains.
    """
    requested = _requested_layers(layers)

    axes = {}
    for axis, cases in _TASK_CASES.items():
        differences = {layer: [] for layer in requested}
        for user_evidence, positive, negative in cases:
            positive_states = _completion_states(model, user_evidence, positive, requested, axis)
            negative_states = _completion_states(model, user_evidence, negative, requested, axis)
            for layer in requested:
                if positive_states[layer].shape != negative_states[layer].shape:
                    raise ValueError(f'Definition task axis {axis} has incompatible state widths at layer {layer}')
                differences[layer].append(_unit(positive_states[layer] - negative_states[layer], axis, layer))
        axes[axis] = {layer: _unit(torch.stack(values).mean(dim=0), axis, layer)
                      for layer, values in differences.items()}
    # Keep completed-assistant case order stable and add the native decision
    # question axis after those four independently minted directions.
    axes['question'] = mint_definition_question_axis(model, requested)
    return axes
