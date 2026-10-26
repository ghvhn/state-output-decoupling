"""Subtract independently measured nuisance components from definition tasks.

These activation directions are candidates, not established semantic factors.
The teacher text never enters an actual probe-definition prompt. Refinement
does not change the subject probe or establish better live definitions.
"""
import math

import torch

from invariants.probe_definition_task import (
    _ASSEMBLY, _ASSEMBLY_CONTRAST, _ROUTES, _ROUTE_CONTRAST, _SENSOR,
    _native_states, _requested_layers, _unit,
)


TASKREFINEMENT_VERSION = 'definition-nuisance-span-v1'
REFINEMENT_VERSION = TASKREFINEMENT_VERSION

_DRINKS = (
    f'[Probe {_SENSOR}]\n'
    'Memory 0: position=-0.80; input="Which drink should we take?"; reply="Take the first ready drink."\n'
    'Memory 1: position=-0.25; input="Which drink should we take?"; reply="Use whichever drink is ready first."\n'
    'Memory 2: position=+0.35; input="Which drink should we take?"; reply="I will compare the two ready drinks before selecting one."\n'
    'Memory 3: position=+0.85; input="Which drink should we take?"; reply="I will examine the available drinks and choose after comparing them."'
)
_PLANTING = (
    f'[Probe {_SENSOR}]\n'
    'Memory 0: position=-0.75; input="Can we plant this?"; reply="I will start putting the plants in the soil."\n'
    'Memory 1: position=-0.20; input="Can we plant this?"; reply="I will begin planting now."\n'
    'Memory 2: position=+0.30; input="Can we plant this?"; reply="I will check which plants fit, then start planting."\n'
    'Memory 3: position=+0.90; input="Can we plant this?"; reply="I will verify the plants and their fit before beginning planting."'
)

# (positive evidence, positive completion, negative evidence, negative completion).
# Enumeration changes presentation while retaining the same four descriptions.
# Topic pairs retain operation and sentence form while changing the objects.
# Generic framing adds a rhetorical introduction to the same operational contrast.
_NUISANCE_CASES = {
    'enumeration': (
        (_ROUTES,
         '0: Takes the first open corridor.\n1: Uses the first available corridor.\n'
         '2: Compares two corridors.\n3: Examines the routes and then chooses.',
         _ROUTES,
         'Takes the first open corridor. Uses the first available corridor. '
         'Compares two corridors. Examines the routes and then chooses.'),
        (_ASSEMBLY,
         '0: Starts putting pieces together.\n1: Begins assembly.\n'
         '2: Checks which pieces fit.\n3: Verifies the parts before assembly.',
         _ASSEMBLY,
         'Starts putting pieces together. Begins assembly. '
         'Checks which pieces fit. Verifies the parts before assembly.'),
    ),
    'topic': (
        (_ROUTES,
         'Lower positions select the first available corridor; higher positions compare corridors before selecting.',
         _DRINKS,
         'Lower positions select the first available drink; higher positions compare drinks before selecting.'),
        (_ASSEMBLY,
         'Lower positions begin assembly immediately; higher positions check the pieces before beginning.',
         _PLANTING,
         'Lower positions begin planting immediately; higher positions check the plants before beginning.'),
    ),
    'generic_framing': (
        (_ROUTES,
         'These are conversations with an AI assistant. The memories show its responses to the user. '
         + _ROUTE_CONTRAST,
         _ROUTES, _ROUTE_CONTRAST),
        (_ASSEMBLY,
         'These memories record conversations with an AI assistant responding to a user. '
         + _ASSEMBLY_CONTRAST,
         _ASSEMBLY, _ASSEMBLY_CONTRAST),
    ),
}


def _finite_fraction(value, description):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{description} must be finite and between zero and one') from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f'{description} must be finite and between zero and one')
    return result


def _validated_directions(axes, layers, kind, widths=None):
    if not isinstance(axes, dict) or not axes:
        raise ValueError(f'Definition refinement requires nonempty {kind} axes')
    normalized = {}
    widths = {} if widths is None else dict(widths)
    for name, directions in axes.items():
        if not isinstance(directions, dict) or _requested_layers(directions) != layers:
            raise ValueError(f'Definition refinement {kind} axis {name} must cover exactly the requested layers')
        normalized[name] = {}
        for layer in layers:
            vector = directions[layer]
            if not torch.is_tensor(vector) or vector.ndim != 1:
                raise ValueError(f'Definition refinement {kind} axis {name} must be a vector at layer {layer}')
            if layer in widths and vector.numel() != widths[layer]:
                raise ValueError(f'Definition refinement {kind} axis {name} has incompatible width at layer {layer}')
            widths[layer] = vector.numel()
            normalized[name][layer] = _unit(vector, f'{kind}:{name}', layer)
    return normalized, widths


@torch.no_grad()
def mint_definition_nuisance_axes(model, layers, task_axes):
    """Mint matched nuisance candidates at completed-assistant endpoints.

    The existing meaning-preserving verbosity axis is reused. The other three
    factors each use two independent matched pairs; their case differences are
    normalized before averaging. No actual probe seeds or memories participate.
    """
    requested = _requested_layers(layers)
    if not isinstance(task_axes, dict) or 'verbosity' not in task_axes:
        raise ValueError('Definition nuisance minting requires the verbosity task axis')
    reused, widths = _validated_directions(
        {'verbosity': task_axes['verbosity']}, requested, 'nuisance')
    nuisances = {'verbosity': {layer: vector.clone() for layer, vector in reused['verbosity'].items()}}
    for name, cases in _NUISANCE_CASES.items():
        differences = {layer: [] for layer in requested}
        for positive_evidence, positive, negative_evidence, negative in cases:
            positive_states = _native_states(
                model, [{'role': 'user', 'content': positive_evidence},
                        {'role': 'assistant', 'content': positive}],
                requested, name, add_generation_prompt=False)
            negative_states = _native_states(
                model, [{'role': 'user', 'content': negative_evidence},
                        {'role': 'assistant', 'content': negative}],
                requested, name, add_generation_prompt=False)
            for layer in requested:
                if (positive_states[layer].shape != negative_states[layer].shape or
                        positive_states[layer].numel() != widths[layer]):
                    raise ValueError(f'Definition nuisance axis {name} has incompatible state widths at layer {layer}')
                differences[layer].append(_unit(positive_states[layer] - negative_states[layer], name, layer))
        nuisances[name] = {layer: _unit(torch.stack(values).mean(dim=0), name, layer)
                           for layer, values in differences.items()}
    return nuisances


@torch.no_grad()
def refine_definition_axes(task_axes, nuisance_axes, strength=1, min_retained=.10):
    """Project nuisance spans out of contrast and abstraction, preserving sources.

    SVD handles duplicate and correlated nuisance directions without repeatedly
    subtracting them. A small retained component fails rather than becoming an
    amplified near-zero direction. Diagnostics describe geometry, not precision.
    Question and all other task axes are copied without changing their values.
    """
    strength = _finite_fraction(strength, 'Refinement strength')
    min_retained = _finite_fraction(min_retained, 'Refinement retained fraction floor')
    if not isinstance(task_axes, dict) or not task_axes:
        raise ValueError('Definition refinement requires nonempty task axes')
    first = next(iter(task_axes.values()))
    if not isinstance(first, dict):
        raise ValueError('Definition refinement task axes must contain per-layer directions')
    layers = _requested_layers(first)
    tasks, widths = _validated_directions(task_axes, layers, 'task')
    nuisances, _ = _validated_directions(nuisance_axes, layers, 'nuisance', widths)
    targets = [name for name in ('contrast', 'abstraction') if name in tasks]
    if not targets:
        raise ValueError('Definition refinement requires a contrast or abstraction task axis')
    result = {name: {layer: vector.detach().clone() for layer, vector in directions.items()}
              for name, directions in task_axes.items()}
    diagnostics = {name: {} for name in targets}
    for layer in layers:
        device = tasks[targets[0]][layer].device
        columns = torch.stack([directions[layer].to(device) for directions in nuisances.values()], dim=1)
        try:
            basis, singular_values, _ = torch.linalg.svd(columns, full_matrices=False)
        except RuntimeError as exc:
            raise ValueError(f'Definition nuisance span could not be resolved at layer {layer}') from exc
        if not torch.isfinite(basis).all() or not torch.isfinite(singular_values).all():
            raise ValueError(f'Definition nuisance span is non-finite at layer {layer}')
        tolerance = max(1e-7, float(singular_values.max()) * 1e-5)
        rank = int((singular_values > tolerance).sum())
        if rank == 0:
            raise ValueError(f'Definition nuisance span has zero rank at layer {layer}')
        basis = basis[:, :rank]
        for name in targets:
            original = tasks[name][layer].to(device)
            removed = basis @ (basis.T @ original)
            residual = original - strength * removed
            retained = float(residual.norm())
            full_span_removed = strength == 1 and rank == original.numel()
            roundoff_floor = 8 * torch.finfo(residual.dtype).eps
            if (full_span_removed or not math.isfinite(retained) or
                    retained <= roundoff_floor or retained < min_retained):
                raise ValueError(f'Definition task axis {name} is inseparable from nuisance span at layer {layer}; '
                                 f'retained fraction {retained:.6g} is below the required floor {min_retained:.6g}')
            cleaned = _unit(residual, name, layer)
            result[name][layer] = cleaned.to(task_axes[name][layer].device)
            diagnostics[name][layer] = {
                'rank': rank,
                'retained_fraction': retained,
                'max_nuisance_overlap_before': float((columns.T @ original).abs().max()),
                'max_nuisance_overlap_after': float((columns.T @ cleaned).abs().max()),
            }
    return result, diagnostics
