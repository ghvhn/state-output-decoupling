"""Persist individually named candidate operations for a definition process.

Teacher requests only mint native assistant-decision directions; they do not
enter the actual generation input. The bank stores four separate vectors in
one atomic weights-only bundle. Valid geometry does not prove live behavior.
"""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch

from invariants.probe_definition_task import (
    _ASSEMBLY, _ROUTES, _native_states, _requested_layers, _unit,
)


STEP_ORDER = ('compare', 'isolate', 'distill', 'verify')
STEP_VERSION = 'definition-steps-decision-candidate-v1'
STEP_GAINS = {step: .04 for step in STEP_ORDER}
_SCHEMA = 1
_REQUESTS = {
    'compare': (
        'In one concise sentence, compare the behavior at the lower and higher positions of this probe.',
        'In one concise sentence, restate each memory separately without comparing positions.',
    ),
    'isolate': (
        'In one concise sentence, identify the behavior that changes between lower and higher positions while setting aside what the examples share.',
        'In one concise sentence, identify the behavior shared by all examples while setting aside differences between positions.',
    ),
    'distill': (
        'In one concise sentence, express the changing behavior as a general operational rule that separates lower and higher positions.',
        'In one concise sentence, express the changing behavior using only the concrete actions described by the individual memories.',
    ),
    'verify': (
        'In one concise sentence, check whether one behavioral distinction fits both ends of this probe using the measured examples, noting any exception.',
        'In one concise sentence, accept a behavioral distinction without checking it against the measured examples or considering exceptions.',
    ),
}


def _plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Step-bank identity cannot contain non-finite metadata')
        return value
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, 'to_dict'):
        return _plain(value.to_dict())
    # AddedToken definitions carry normalization and whitespace behavior.
    if type(value).__name__ == 'AddedToken' and hasattr(value, 'content'):
        return {key: _plain(getattr(value, key, None)) for key in
                ('content', 'single_word', 'lstrip', 'rstrip', 'normalized', 'special')}
    raise ValueError(f'Step-bank identity has unsupported metadata type {type(value).__name__}')


def _digest(value):
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _class_name(value):
    return type(value).__module__ + '.' + type(value).__qualname__


def _local_manifest(source):
    directory = Path(source)
    if not directory.is_dir():
        return []
    files = set()
    for pattern in ('*.safetensors', 'pytorch_model*.bin', '*.index.json', 'adapter_model*.bin'):
        files.update(directory.glob(pattern))
    return [{'name': item.name, 'size': item.stat().st_size, 'mtime_ns': item.stat().st_mtime_ns}
            for item in sorted(files) if item.is_file()]


def _tokenizer_identity(tok):
    template_getter = getattr(tok, 'get_chat_template', None)
    template = template_getter() if callable(template_getter) else getattr(tok, 'chat_template', None)
    if not isinstance(template, str) or not template:
        raise ValueError('Step-bank persistence requires an identifiable native chat template')
    backend = getattr(tok, 'backend_tokenizer', None)
    serializer = getattr(backend, 'to_str', None)
    if callable(serializer):
        serialized = serializer()
        if not isinstance(serialized, str) or not serialized:
            raise ValueError('Step-bank persistence requires a serialized tokenizer identity')
        implementation = {'backend_sha256': hashlib.sha256(serialized.encode('utf-8')).hexdigest()}
    else:
        sp_model = getattr(tok, 'sp_model', None)
        serializer = getattr(sp_model, 'serialized_model_proto', None)
        if not callable(serializer):
            raise ValueError('Step-bank persistence requires tokenizer backend or sentencepiece identity')
        serialized = serializer()
        if not isinstance(serialized, bytes) or not serialized:
            raise ValueError('Step-bank persistence requires serialized sentencepiece identity')
        implementation = {'sentencepiece_sha256': hashlib.sha256(serialized).hexdigest()}
    vocabulary_getter = getattr(tok, 'get_vocab', None)
    vocabulary = vocabulary_getter() if callable(vocabulary_getter) else None
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError('Step-bank persistence requires tokenizer vocabulary identity')
    init = getattr(tok, 'init_kwargs', {}) or {}
    options = {key: init[key] for key in ('add_bos_token', 'add_eos_token', 'do_lower_case',
                                         'strip_accents', 'legacy', 'split_special_tokens') if key in init}
    return {
        'class': _class_name(tok), 'name': str(getattr(tok, 'name_or_path', '')),
        'revision': str(init.get('_commit_hash') or ''),
        'template_sha256': hashlib.sha256(template.encode('utf-8')).hexdigest(),
        'vocabulary_sha256': _digest(vocabulary),
        'special_tokens_sha256': _digest(getattr(tok, 'special_tokens_map_extended',
                                               getattr(tok, 'special_tokens_map', {}))),
        'options_sha256': _digest(options), **implementation,
    }


def _identity(model, layers):
    raw = model.model
    config = getattr(raw, 'config', None)
    source = (getattr(config, '_name_or_path', None) or getattr(raw, 'name_or_path', None)
              or getattr(model, 'name_or_path', None) or getattr(model, 'name', None))
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Step-bank persistence requires a model source identity')
    revision = getattr(config, '_commit_hash', None)
    if not revision:
        match = re.search(r'[/\\]snapshots[/\\]([0-9a-fA-F]{40})(?:[/\\]|$)', source)
        revision = match.group(1).lower() if match else None
    manifest = _local_manifest(source)
    if not revision and not manifest:
        raise ValueError('Step-bank persistence requires an immutable model revision or local weight manifest')
    try:
        width = int(model.d_model)
        count = int(model.n_layers)
        for value, converted in ((model.d_model, width), (model.n_layers, count)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) != converted:
                raise ValueError('Non-integer model dimensions')
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Step-bank persistence requires model width and layer count') from exc
    if width <= 0 or count <= 0 or layers[-1] >= count:
        raise ValueError('Step-bank requested layers must exist in the identified model')
    config_getter = getattr(config, 'to_dict', None)
    if not callable(config_getter):
        raise ValueError('Step-bank persistence requires serializable model configuration')
    quantizer = getattr(raw, 'hf_quantizer', None)
    quantization = (getattr(quantizer, 'quantization_config', None) or
                    getattr(config, 'quantization_config', None) or
                    getattr(raw, 'quantization_config', None))
    parameters_getter = getattr(raw, 'parameters', None)
    parameter_dtypes = (sorted({str(parameter.dtype) for parameter in parameters_getter()})
                        if callable(parameters_getter) else [])
    dtype = getattr(raw, 'dtype', None)
    if dtype is None:
        dtype = getattr(config, 'dtype', None)
    if dtype is None:
        dtype = getattr(config, 'torch_dtype', None)
    if dtype is None and not parameter_dtypes:
        raise ValueError('Step-bank persistence requires actual model dtype identity')
    device_map = getattr(raw, 'hf_device_map', None)
    effective_load = {
        'dtype': str(dtype), 'parameter_dtypes': parameter_dtypes,
        'device_map': _plain(device_map), 'device': str(model.device),
        'loaded_in_4bit': bool(getattr(raw, 'is_loaded_in_4bit', False)),
        'loaded_in_8bit': bool(getattr(raw, 'is_loaded_in_8bit', False)),
        'quantization_sha256': _digest(quantization),
        'attention': str(getattr(config, '_attn_implementation', '')),
    }
    versions = {}
    for name in ('torch', 'transformers', 'tokenizers', 'bitsandbytes'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    recipe = {'requests': _REQUESTS, 'evidence': [_ROUTES, _ASSEMBLY]}
    return {
        'version': STEP_VERSION, 'recipe_sha256': _digest(recipe),
        'model': {'class': _class_name(raw), 'source': source, 'revision': str(revision or ''),
                  'weights': manifest, 'config_sha256': _digest(config_getter())},
        'tokenizer': _tokenizer_identity(model.tok), 'effective_load': effective_load,
        'width': width, 'model_layers': count, 'layers': list(layers), 'libraries': versions,
    }


def _validate_bank(bank, layers, width):
    if not isinstance(bank, dict) or set(bank) != set(STEP_ORDER):
        raise ValueError('Step bank must contain all four named operations and no extra operations')
    validated = {}
    for step in STEP_ORDER:
        directions = bank[step]
        if (not isinstance(directions, dict) or set(directions) != set(layers) or
                any(type(layer) is not int for layer in directions)):
            raise ValueError(f'Step bank {step} must cover exactly the requested layers')
        validated[step] = {}
        for layer in layers:
            vector = directions[layer]
            if (not torch.is_tensor(vector) or vector.dtype != torch.float32 or
                    vector.device.type != 'cpu' or vector.shape != (width,)):
                raise ValueError(f'Step bank {step} has invalid tensor storage or width at layer {layer}')
            norm = vector.norm()
            if (not torch.isfinite(vector).all() or not torch.isfinite(norm) or
                    abs(float(norm) - 1.) > 1e-4):
                raise ValueError(f'Step bank {step} must be finite and normalized at layer {layer}')
            validated[step][layer] = vector.detach().clone()
    return validated


@torch.no_grad()
def _mint(model, layers, width):
    bank = {}
    for step in STEP_ORDER:
        differences = {layer: [] for layer in layers}
        positive_request, negative_request = _REQUESTS[step]
        for evidence in (_ROUTES, _ASSEMBLY):
            positive = _native_states(
                model, [{'role': 'user', 'content': evidence + '\n\n' + positive_request}],
                layers, step, add_generation_prompt=True)
            negative = _native_states(
                model, [{'role': 'user', 'content': evidence + '\n\n' + negative_request}],
                layers, step, add_generation_prompt=True)
            for layer in layers:
                if positive[layer].shape != (width,) or negative[layer].shape != (width,):
                    raise ValueError(f'Definition step {step} has incompatible state widths at layer {layer}')
                differences[layer].append(_unit(positive[layer] - negative[layer], step, layer))
        bank[step] = {layer: _unit(torch.stack(values).mean(dim=0), step, layer).cpu().float()
                      for layer, values in differences.items()}
    return _validate_bank(bank, layers, width)


def _validate_bundle(payload, identity, fingerprint, layers):
    if (not isinstance(payload, dict) or payload.get('schema') != _SCHEMA or
            payload.get('identity') != identity or payload.get('identity_hash') != fingerprint):
        raise ValueError('Step-bank identity does not match the current model, tokenizer, layers, and recipe')
    return _validate_bank(payload.get('bank'), layers, identity['width'])


def _write_atomic(path, payload, identity, fingerprint, layers):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Validate serialized tensors through the restricted loader before replacing.
        _validate_bundle(torch.load(temporary, map_location='cpu', weights_only=True), identity, fingerprint, layers)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def get_or_mint_definition_steps(model, layers, path):
    """Return a validated CPU bank plus identity, provenance, and load status.

    A directory stores one bundle per identity fingerprint; an explicit .pt
    path supports bounded remint on mismatch. No old bank is replaced unless
    all four freshly minted operations and their serialized bundle validate.
    Callers must invalidate banks after in-memory weight changes not represented
    by a new model revision or local weight manifest.
    """
    requested = _requested_layers(layers)
    identity = _identity(model, requested)
    fingerprint = _digest(identity)
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != '.pt' or destination.is_dir():
        destination = destination / ('bank-' + fingerprint + '.pt')
    provenance = {'endpoint': 'native_assistant_decision', 'cases_per_step': 2,
                  'recipe_sha256': identity['recipe_sha256'], 'status': 'candidate'}
    metadata = {'path': str(destination), 'version': STEP_VERSION, 'identity': identity,
                'identity_hash': fingerprint, 'step_order': list(STEP_ORDER),
                'gains': dict(STEP_GAINS), 'provenance': provenance}
    if destination.exists():
        try:
            payload = torch.load(destination, map_location='cpu', weights_only=True)
            bank = _validate_bundle(payload, identity, fingerprint, requested)
        except Exception as exc:
            metadata['cache_rejection'] = type(exc).__name__ + ': ' + str(exc)[:240]
        else:
            metadata.update(load_status='loaded', cache_status='hit')
            return bank, metadata
    bank = _mint(model, requested, identity['width'])
    payload = {'schema': _SCHEMA, 'identity': identity, 'identity_hash': fingerprint,
               'bank': bank, 'provenance': provenance}
    _write_atomic(destination, payload, identity, fingerprint, requested)
    metadata.update(load_status='minted', cache_status='minted')
    return bank, metadata
