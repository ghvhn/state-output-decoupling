"""Compare activation explanation transfers in an isolated cached model.

Every decoded variant receives the same native empty user body. Generic task
requests exist only in calibration forwards. Results describe candidates, not
verified explanations of the selected activation direction.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACT_ROOT = ROOT / 'invariants' / 'out' / 'activation_runs'

OPERATION_REQUESTS = (
    'Explain the distinction currently active in your own activations. '
    'Describe what this distinction means in one concise sentence.',
    'In one concise sentence, describe the behavioral distinction currently '
    'active in your own activations and what moving in either direction would express.',
)
REFERENCE_REQUESTS = (
    (
        'In one concise sentence, explain the currently active distinction as a property of your own activations.',
        'In one concise sentence, explain the currently active distinction as a property of externally observed behavior.',
    ),
    (
        'Describe the currently active behavioral distinction in your own activations in one concise sentence.',
        'Describe the currently active behavioral distinction in externally observed behavior in one concise sentence.',
    ),
)
VARIANTS = (
    'baseline', 'current_core',
    'matched_operation_current_reference_decision', 'matched_both_decision',
    'matched_operation_current_reference_generation', 'matched_both_generation',
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def calibration_body(body, request):
    # The source body is never reconstructed by appending an empty request.
    return body + ('\n\n' if body else '') + request


def geometry(source, target):
    import torch

    difference = target - source
    source_norm, target_norm, delta_norm = (float(value.norm()) for value in (source, target, difference))
    return {
        'source_norm': source_norm, 'target_norm': target_norm, 'delta_norm': delta_norm,
        'delta_over_source_norm': delta_norm / max(source_norm, 1e-30),
        'source_target_cosine': float(torch.nn.functional.cosine_similarity(source, target, dim=0)),
    }


def matched_axes(model, layers, body):
    import torch
    from invariants import probe_activation_explain as core
    from invariants.probe_definition_task import _unit

    source = core._raw_decision_states(model, [{'role': 'user', 'content': body}], layers, assistant_prefix='')
    operation_changes = {layer: [] for layer in layers}
    reference_changes = {layer: [] for layer in layers}
    diagnostics = {'source_evidence_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest(),
                   'source_is_exact_runtime_body': True, 'synthetic_observations': False,
                   'endpoint': 'raw_decoder_block_output_at_assistant_decision',
                   'operation_requests': list(OPERATION_REQUESTS),
                   'reference_requests': [list(pair) for pair in REFERENCE_REQUESTS],
                   'operation': [], 'reference': [], 'candidate': True,
                   'semantic_truth_checked': False}
    for request in OPERATION_REQUESTS:
        target = core._raw_decision_states(model,
            [{'role': 'user', 'content': calibration_body(body, request)}], layers, assistant_prefix='')
        diagnostics['operation'].append({str(layer): geometry(source[layer], target[layer]) for layer in layers})
        for layer in layers:
            operation_changes[layer].append(_unit(target[layer] - source[layer], 'matched_operation', layer))
    for positive, negative in REFERENCE_REQUESTS:
        own = core._raw_decision_states(model,
            [{'role': 'user', 'content': calibration_body(body, positive)}], layers, assistant_prefix='')
        external = core._raw_decision_states(model,
            [{'role': 'user', 'content': calibration_body(body, negative)}], layers, assistant_prefix='')
        diagnostics['reference'].append({str(layer): geometry(external[layer], own[layer]) for layer in layers})
        for layer in layers:
            reference_changes[layer].append(_unit(own[layer] - external[layer], 'matched_reference', layer))
    axes = {
        name: {layer: _unit(torch.stack(changes[layer]).mean(0), name, layer).cpu().float()
               for layer in layers}
        for name, changes in (('operation', operation_changes), ('reference', reference_changes))
    }
    return axes, diagnostics


@contextmanager
def windows_trim_signature():
    """Use pointer-sized arguments for the engine's optional Windows memory trim."""
    import ctypes

    trim = ctypes.windll.kernel32.SetProcessWorkingSetSize if hasattr(ctypes, 'windll') else None
    original = trim.argtypes if trim is not None else None
    try:
        if trim is not None:
            trim.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        yield
    finally:
        if trim is not None:
            trim.argtypes = original


def run_variant(model, subject, current_axes, matched, cache, name, budget, cap, gains):
    import torch
    from invariants import agentic_engine, probe_activation_explain as core
    from invariants.config import AgenticConfig
    from invariants.probe_behavior import build_definition_handles

    config = AgenticConfig()
    prompt, prefill = core.build_activation_explanation_prompt(model, '')
    top_logits, handles = [], []

    def capture_first(_module, _args, output):
        if top_logits:
            return
        logits = output.logits[0, -1].detach().float()
        values, indices = torch.topk(logits, min(10, logits.numel()))
        probabilities = logits.softmax(-1)
        top_logits.extend({'id': int(index), 'token': model.tok.decode([int(index)]),
                           'logit': float(value), 'probability': float(probabilities[index])}
                          for index, value in zip(indices, values))

    observer = model.model.register_forward_hook(capture_first)
    started = time.monotonic()
    try:
        if name == 'current_core':
            output = core.generate_activation_explanation(
                model, subject, '', generate=agentic_engine.generate_agentic_text, config=config,
                max_tokens=budget, cap_fraction=cap, axis_cache=cache,
                subject_gain=gains['subject'], operation_gain=gains['operation'], reference_gain=gains['reference'])
            text, telemetry = output['text'], output['telemetry']
            metadata = output['metadata']
        else:
            active_gains = {'subject': 0., 'operation': 0., 'reference': 0.} if name == 'baseline' else gains
            placement = 'generation' if name.endswith('_generation') else 'decision'
            axes = {'operation': matched['operation'],
                    'reference': current_axes['reference'] if 'current_reference' in name else matched['reference']}
            with core._scratch_config(config), core._scratch_cap_telemetry() as cap_telemetry:
                try:
                    if any(active_gains.values()):
                        handles = build_definition_handles(
                            model, subject, axes, cap_fraction=cap, subject_fraction=active_gains['subject'],
                            task_gains={axis: active_gains[axis] for axis in axes},
                            task_positions={axis: placement for axis in axes})
                    text, telemetry = agentic_engine.generate_agentic_text(
                        model, instruction=prompt, config=config, vecs={}, pre_formatted=True,
                        max_new_tokens=budget, max_tool_calls=0, chatty_log=False,
                        return_telemetry=True, generation_channel='reader')
                finally:
                    for handle in handles:
                        handle.remove()
                    handles.clear()
            metadata = {'candidate': True, 'semantic_truth_checked': False, 'gains': active_gains,
                        'operation_reference_placement': placement, 'subject_placement': 'decision',
                        'cap_fraction': cap, 'cap_telemetry': cap_telemetry, 'prefill': prefill}
    finally:
        for handle in handles:
            handle.remove()
        observer.remove()
    return {'variant': name, 'text': text, 'tokens_generated': telemetry['tokens_generated'],
            'elapsed_seconds': time.monotonic() - started, 'first_step_top10': top_logits,
            'telemetry': telemetry, 'metadata': metadata, 'runtime_user_body': '',
            'runtime_prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--probe-file', type=Path, default=ROOT / 'invariants/out/probes/illeism.pt')
    parser.add_argument('--layers', help='Optional comma-separated subset of the stored direction layers')
    parser.add_argument('--budget', type=int, choices=(64, 96), default=64)
    parser.add_argument('--cap', type=float, default=.15)
    parser.add_argument('--subject-gain', type=float, default=.02)
    parser.add_argument('--operation-gain', type=float, default=.12)
    parser.add_argument('--reference-gain', type=float, default=.04)
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap) or not 0 < args.cap <= 1:
        parser.error('Cap must be finite and between zero and one')
    gains = {'subject': args.subject_gain, 'operation': args.operation_gain, 'reference': args.reference_gain}
    if any(not math.isfinite(value) or abs(value) > 1 for value in gains.values()) or gains['subject'] < 0:
        parser.error('Gains must be finite and bounded; subject gain must be nonnegative')
    variants = tuple(args.variants.split(','))
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        parser.error('Choose distinct known comparison variants')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    destination = (args.out or ARTIFACT_ROOT / ('transfer-' + stamp)).resolve()
    artifact_root = ARTIFACT_ROOT.resolve()
    if destination == artifact_root or artifact_root not in destination.parents:
        parser.error('Output must be a new subdirectory of invariants/out/activation_runs')
    destination.mkdir(parents=True, exist_ok=False)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    report = {'status': 'starting', 'model': args.model, 'load_mode': '4bit', 'local_files_only': True,
              'probe_file': str(args.probe_file.resolve()), 'runtime_user_body': '',
              'budget': args.budget, 'cap_fraction': args.cap, 'gains': gains,
              'candidate': True, 'semantic_truth_checked': False, 'results': []}
    write_json(destination / 'results.json', report)
    print(f'[Activation transfer] Artifacts: {destination}', flush=True)
    try:
        import torch
        from invariants import engine, probe_activation_explain as core
        from invariants.config import AgenticConfig

        if not torch.cuda.is_available():
            report.update(status='blocked', reason='CUDA is unavailable; no model was loaded')
            return 2
        free, total = torch.cuda.mem_get_info()
        report['gpu_before'] = {'free_gib': free / 1024**3, 'total_gib': total / 1024**3}
        if free < 10 * 1024**3:
            report.update(status='blocked', reason='Less than 10 GiB GPU memory is free; no model was loaded')
            return 2
        # Only the stored direction is selected. No label or framing is used in
        # calibration requests, the runtime prompt, or the generated response.
        payload = torch.load(args.probe_file, map_location='cpu', weights_only=True)
        directions = payload['direction']
        layers = tuple(sorted(int(layer) for layer in directions)) if args.layers is None else tuple(
            sorted(set(int(value) for value in args.layers.split(','))))
        subject = {layer: directions.get(layer, directions.get(str(layer))).detach().float().cpu().clone()
                   for layer in layers}
        if not subject or any(not torch.isfinite(vector).all() or vector.norm() <= 1e-8 for vector in subject.values()):
            raise ValueError('The selected direction has invalid geometry')
        subject = {layer: vector / vector.norm() for layer, vector in subject.items()}
        report['layers'] = list(layers)
        report['subject_sha256'] = {str(layer): hashlib.sha256(vector.numpy().tobytes()).hexdigest()
                                    for layer, vector in subject.items()}
        print('[Activation transfer] Loading one separate cached 4-bit model.', flush=True)
        model = engine.load_model(args.model, local_files_only=True, load_mode='4bit')
        if any(layer >= model.n_layers or vector.numel() != model.d_model for layer, vector in subject.items()):
            raise ValueError('The selected direction does not match the loaded model')
        cache = {}
        with core._scratch_config(AgenticConfig()), core._scratch_cap_telemetry():
            print('[Activation transfer] Calibrating current and source-matched candidate axes.', flush=True)
            current_axes, current_metadata = core.get_or_mint_activation_axes(
                model, layers, '', cache, subject=subject)
            matched, diagnostics = matched_axes(model, layers, '')
        report['current_calibration'] = current_metadata
        report['matched_calibration'] = diagnostics
        report['axis_alignment'] = {
            name: {str(layer): float(torch.dot(current_axes[name][layer], matched[name][layer])) for layer in layers}
            for name in ('operation', 'reference')}
        torch.save({'subject': subject, 'current': current_axes, 'matched': matched}, destination / 'axes.pt')
        with windows_trim_signature():
            for name in variants:
                print(f'[Activation transfer] Running {name}.', flush=True)
                result = run_variant(model, subject, current_axes, matched, cache, name, args.budget, args.cap, gains)
                report['results'].append(result)
                write_json(destination / (name + '.json'), result)
                (destination / (name + '.txt')).write_text(result['text'], encoding='utf-8')
                write_json(destination / 'results.json', report)
                print(f"[Activation transfer] {name}: {result['tokens_generated']} tokens; {result['text'][:180]!r}", flush=True)
        report['status'] = 'complete'
        return 0
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        (destination / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return 1
    finally:
        write_json(destination / 'results.json', report)
        print(f"[Activation transfer] {report['status']}: {destination}", flush=True)
        if report.get('reason') or report.get('error'):
            print(report.get('reason') or report.get('error'), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
