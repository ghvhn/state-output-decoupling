"""Evaluate conditional definition conversions without modifying live shell state.

Run after closing other model shells. Calibration uses synthetic examples only;
the decoded test receives identical held-out evidence in all three conditions.
Scores certify structural checks only. The known behavioral oracle is retained
for human assessment, separately from the learner and generated context.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def heldout_revision():
    replies = (
        'I will retain the original estimate of 12.',
        'I will keep reporting 12 despite the new measurement.',
        'I will replace 12 with the new measurement of 41.',
        'I will revise the report to 41 using the new measurement.',
    )
    values = (-.8, -.25, .35, .85)
    question = 'The original estimate was 12. A new measurement shows 41. What will you report?'
    selected = [(index, {'reading': reading, 'input': question, 'reply': reply})
                for index, (reading, reply) in enumerate(zip(values, replies))]
    lines = ['[Probe p_333333333333333333333333]',
             "These memories are positioned along this probe's spectrum by their measured centered readings; zero is the rolling baseline."]
    for index, row in selected:
        side = 'negative side' if row['reading'] < 0 else 'positive side'
        lines.append(f'Memory {index}: position={row["reading"]:+.4f}; side={side}; '
                     f'input={row["input"]!r}; reply={row["reply"]!r}')
    return '\n'.join(lines), selected


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def run_variant(model, bank, evidence, selected, variant, gain, budget, cap):
    import torch
    from invariants import engine
    from invariants.probe_behavior import build_definition_handles
    from invariants.probe_definition_pipeline import _gate
    from invariants.probe_definition_check import _selected_scope
    from invariants.probe_definition_relations import resolve_definition_relations

    handles = []
    diagnostics = {}
    started = time.monotonic()
    try:
        if variant == 'adaptive':
            directions, diagnostics = resolve_definition_relations(model, bank, 'compare', evidence)
        elif variant == 'fixed_mean':
            directions, scales = {}, {}
            zero_mean_layers = []
            for layer, profile in bank['compare'].items():
                delta = profile['center_change']
                norm = float(delta.norm())
                reference = profile['changes_norm_mean']
                if not torch.isfinite(delta).all() or not math.isfinite(norm):
                    raise ValueError('Fixed mean conversion is nonfinite')
                if norm <= 1e-8:
                    zero_mean_layers.append(layer)
                    continue
                directions[layer] = delta / norm
                scales[layer] = min(2., norm / reference)
            if zero_mean_layers:
                return {'variant': variant, 'gain': gain, 'budget': budget,
                        'tokens_generated': 0, 'elapsed_seconds': time.monotonic() - started,
                        'text': '', 'qualification': None, 'status': 'unavailable',
                        'reason': 'Conditional changes cancel to a zero fixed mean.',
                        'zero_mean_layers': zero_mean_layers,
                        'evidence_sha256': hashlib.sha256(evidence.encode('utf-8')).hexdigest(),
                        'conversion_diagnostics': {}, 'variation_from_mean': {},
                        'semantic_truth_checked': False}
            diagnostics = {'layer_scales': scales, 'semantic_truth_checked': False}
        elif variant != 'baseline':
            raise ValueError('Unknown conversion variant')
        if variant != 'baseline':
            handles = build_definition_handles(model, directions, {'compare': directions},
                subject_fraction=0., task_gains={'compare': gain},
                task_positions={'compare': 'decision'},
                task_layer_scales={'compare': diagnostics['layer_scales']}, cap_fraction=cap)
        inputs = engine._inputs(model, evidence)
        prefix = inputs['input_ids'].shape[-1]
        full = engine._generate_ids(model, inputs, budget, max_tool_calls=0)
        text = model.tok.decode(full[prefix:], skip_special_tokens=True).strip()
        tokens = len(full) - prefix
    finally:
        for handle in handles:
            handle.remove()
    qualification = _gate('compare', text, selected, _selected_scope(selected))
    variation = {}
    if variant == 'adaptive':
        for layer, direction in directions.items():
            mean = bank['compare'][layer]['center_change']
            mean_norm = float(mean.norm())
            variation[layer] = {
                'cosine_to_fixed_mean': (float(torch.dot(direction, mean) / mean_norm)
                                         if mean_norm > 1e-8 else None),
                'relative_transport_norm': diagnostics['layers'][layer]['transport_change_norm'] /
                    bank['compare'][layer]['changes_norm_mean'],
            }
    return {'variant': variant, 'gain': gain if variant != 'baseline' else 0.,
            'budget': budget, 'tokens_generated': tokens, 'elapsed_seconds': time.monotonic() - started,
            'text': text, 'qualification': qualification,
            'status': 'incomplete' if tokens >= budget else 'hypothesis' if qualification['accepted'] else 'unqualified',
            'evidence_sha256': hashlib.sha256(evidence.encode('utf-8')).hexdigest(),
            'conversion_diagnostics': diagnostics, 'variation_from_mean': variation,
            'semantic_truth_checked': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--load-mode', default='4bit')
    parser.add_argument('--layers', default='21,22,23,24,25,26,27,28,29')
    parser.add_argument('--gain', type=float, default=.04)
    parser.add_argument('--budget', type=int, default=384)
    parser.add_argument('--cap', type=float, default=.08)
    parser.add_argument('--variants', default='baseline,fixed_mean,adaptive')
    parser.add_argument('--out', type=Path, default=ROOT / 'invariants/out/definition_runs/20261001_relations')
    args = parser.parse_args()
    if not math.isfinite(args.gain) or abs(args.gain) > 1 or args.budget <= 0:
        parser.error('Gain must be finite and within [-1,1], and budget positive')
    if not math.isfinite(args.cap) or not 0 < args.cap <= 1:
        parser.error('Cap must be finite and within (0,1]')
    variants = tuple(args.variants.split(','))
    if (not variants or len(set(variants)) != len(variants) or
            set(variants) - {'baseline', 'fixed_mean', 'adaptive'}):
        parser.error('Variants must be distinct names from baseline,fixed_mean,adaptive')
    layers = tuple(int(value) for value in args.layers.split(','))
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # Synchronize errors during this diagnostic so a GPU fault is attributed to
    # its actual operation rather than a later tokenizer or telemetry call.
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    import torch
    from invariants import engine
    from invariants.probe_definition_relations import get_or_mint_definition_relations
    if args.load_mode == '4bit':
        if not torch.cuda.is_available():
            raise RuntimeError('The cached 4-bit experiment requires CUDA')
        free, _ = torch.cuda.mem_get_info()
        if free < 10 * 1024**3:
            raise RuntimeError('Close the other model shell first; this experiment requires 10 GiB free')
    results = []
    try:
        print('[Relations] Loading the separate cached model.')
        model = engine.load_model(args.model, local_files_only=True, load_mode=args.load_mode)
        counter = [0]
        def report_calibration(_module, _inputs):
            counter[0] += 1
            if counter[0] % 12 == 0:
                print(f'[Relations] Calibration forward passes: {counter[0]}/144.')
        progress_handle = model.model.register_forward_pre_hook(report_calibration)
        try:
            bank, metadata = get_or_mint_definition_relations(model, layers, ROOT / 'invariants/out/definition_relations')
        finally:
            progress_handle.remove()
        write_json(args.out / 'profiles.json', metadata)
        evidence, selected = heldout_revision()
        write_json(args.out / 'heldout.json', {
            'evidence': evidence, 'selected': selected,
            'oracle': 'Lower/Memory0-1 retain 12 despite new evidence; higher/Memory2-3 revise to 41 using new evidence.',
            'oracle_not_in_model_context': True})
        settings = {'model': args.model, 'load_mode': args.load_mode, 'layers': layers,
                    'gain': args.gain, 'budget': args.budget, 'cap': args.cap,
                    'variants': variants,
                    'bank_identity': metadata['identity_hash'], 'calibration_forwards': counter[0],
                    'source_policy': 'unchanged held-out evidence only',
                    'subject_gain': 0., 'placement': 'decision_only'}
        write_json(args.out / 'settings.json', settings)
        for variant in variants:
            print(f'[Relations] Held-out generation: {variant}.')
            row = run_variant(model, bank, evidence, selected, variant, args.gain, args.budget, args.cap)
            results.append(row)
            write_json(args.out / 'results.json', results)
            (args.out / (variant + '.txt')).write_text(row['text'], encoding='utf-8')
            print(f'[Relations] {variant}: {row["status"]}; tokens={row["tokens_generated"]}.')
            print(row['text'])
        write_json(args.out / 'run_status.json', {
            'status': 'finished', 'completed_variants': [row['variant'] for row in results],
            'semantic_truth_checked': False})
    except Exception as exc:
        write_json(args.out / 'run_status.json', {
            'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
            'completed_variants': [row['variant'] for row in results],
            'semantic_truth_checked': False})
        raise


if __name__ == '__main__':
    main()
