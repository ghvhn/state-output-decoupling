"""Candidate definition operations measured at the assistant decision point.

Only minting inputs contain task requests. Their evidence can be synthetic or
the exact selected observations; actual generation receives measured evidence
alone. Matching phase and context does not establish precision or success.
"""
import torch

from invariants.probe_definition_task import (
    _ASSEMBLY, _ROUTES, _native_states, _requested_layers, _unit,
)
from invariants.probe_definition_refinement import _DRINKS, _PLANTING


OPERATION_VERSION = 'definition-operation-decision-contextual-candidate-v2'

_DEFINE_REQUEST = (
    "In one concise sentence, define the behavioral distinction between this probe's lower and higher positions."
)
_RECAP_REQUEST = (
    "In one concise sentence, restate the individual behavior described by each of this probe's memories."
)
_RULE_REQUEST = (
    'In one concise sentence, express the general behavioral rule that separates the lower and higher positions of this probe.'
)
_INDIVIDUAL_REQUEST = (
    'In one concise sentence, express the separate concrete behavior recorded in each memory for this probe.'
)

_CONTEXTUAL_OPERATION_REQUESTS = (
    (_DEFINE_REQUEST, _RECAP_REQUEST),
    (_RULE_REQUEST, _INDIVIDUAL_REQUEST),
)

_OPERATION_CASES = tuple(
    (evidence, _DEFINE_REQUEST, evidence, _RECAP_REQUEST)
    for evidence in (_ROUTES, _ASSEMBLY)
)

_NUISANCE_CASES = {
    'verbosity': tuple(
        (evidence, _DEFINE_REQUEST.replace('concise', 'detailed'), evidence, _DEFINE_REQUEST)
        for evidence in (_ROUTES, _ASSEMBLY)
    ),
    'enumeration': tuple(
        (evidence,
         "Define the behavioral distinction between this probe's lower and higher positions in concise numbered sentences.",
         evidence,
         "Define the behavioral distinction between this probe's lower and higher positions in concise prose sentences.")
        for evidence in (_ROUTES, _ASSEMBLY)
    ),
    'topic': (
        (_ROUTES, _DEFINE_REQUEST, _DRINKS, _DEFINE_REQUEST),
        (_ASSEMBLY, _DEFINE_REQUEST, _PLANTING, _DEFINE_REQUEST),
    ),
    'generic_framing': tuple(
        (evidence,
         _DEFINE_REQUEST + ' Frame this distinction as a general pattern in conversations with an AI assistant.',
         evidence,
         _DEFINE_REQUEST + ' Frame this distinction directly as a behavioral pattern.')
        for evidence in (_ROUTES, _ASSEMBLY)
    ),
}

_CONTEXTUAL_NUISANCE_REQUESTS = {
    'verbosity': (
        (_DEFINE_REQUEST.replace('concise', 'detailed'), _DEFINE_REQUEST),
        (_RULE_REQUEST.replace('concise', 'detailed'), _RULE_REQUEST),
    ),
    'enumeration': (
        ("Define the behavioral distinction between this probe's lower and higher positions in concise numbered sentences.",
         "Define the behavioral distinction between this probe's lower and higher positions in concise prose sentences."),
        ('Express the general behavioral rule separating the lower and higher positions of this probe in concise numbered sentences.',
         'Express the general behavioral rule separating the lower and higher positions of this probe in concise prose sentences.'),
    ),
    'generic_framing': (
        (_DEFINE_REQUEST + ' Frame this distinction as a general pattern in conversations with an AI assistant.',
         _DEFINE_REQUEST + ' Frame this distinction directly as a behavioral pattern.'),
        (_RULE_REQUEST + ' Present it as a general feature of conversations with an AI assistant.',
         _RULE_REQUEST + ' Present it directly as a behavioral distinction.'),
    ),
}


def _contextual_cases(evidence, requests):
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Contextual definition evidence must be a nonempty string')
    # Retain the supplied evidence byte-for-byte; only requests vary by case.
    return tuple((evidence, positive, evidence, negative) for positive, negative in requests)


def _mint_decision_contrast(model, layers, name, cases):
    differences = {layer: [] for layer in layers}
    for positive_evidence, positive_request, negative_evidence, negative_request in cases:
        positive = _native_states(
            model, [{'role': 'user', 'content': positive_evidence + '\n\n' + positive_request}],
            layers, name, add_generation_prompt=True)
        negative = _native_states(
            model, [{'role': 'user', 'content': negative_evidence + '\n\n' + negative_request}],
            layers, name, add_generation_prompt=True)
        for layer in layers:
            if (positive[layer].shape != negative[layer].shape or
                    (differences[layer] and positive[layer].shape != differences[layer][0].shape)):
                raise ValueError(f'Definition operation axis {name} has incompatible state widths at layer {layer}')
            differences[layer].append(_unit(positive[layer] - negative[layer], name, layer))
    return {layer: _unit(torch.stack(values).mean(dim=0), name, layer)
            for layer, values in differences.items()}


@torch.no_grad()
def mint_definition_operation_axis(model, layers, evidence=None):
    """Contrast concise operational definitions with concise per-memory recaps.

    Evidence is identical within each pair. By default two synthetic contexts
    are used. Supplied evidence uses two request paraphrases in that exact
    context. Both USER requests ask for one sentence, and the native assistant
    generation header is included. Case differences and their average are
    normalized separately at every layer. Caller must scope caching by evidence.
    """
    requested = _requested_layers(layers)
    cases = _OPERATION_CASES if evidence is None else _contextual_cases(evidence, _CONTEXTUAL_OPERATION_REQUESTS)
    return _mint_decision_contrast(model, requested, 'operation', cases)


@torch.no_grad()
def mint_definition_operation_nuisances(model, layers, evidence=None):
    """Mint same-phase nuisance candidates while preserving the define operation.

    Every factor keeps the lower/higher behavioral definition task. Supplied
    evidence is held exact across two request paraphrases per factor; topic
    variation is omitted rather than rewriting archived observations. None
    uses per-memory recap as a nuisance. Directions can still overlap useful
    task content and require retention diagnostics and controlled live outputs.
    """
    requested = _requested_layers(layers)
    cases_by_name = (_NUISANCE_CASES if evidence is None else
                     {name: _contextual_cases(evidence, pairs)
                      for name, pairs in _CONTEXTUAL_NUISANCE_REQUESTS.items()})
    return {name: _mint_decision_contrast(model, requested, name, cases)
            for name, cases in cases_by_name.items()}
