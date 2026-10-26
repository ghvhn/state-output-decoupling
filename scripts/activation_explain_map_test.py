"""Compare a current-state map with its fixed-mean control on a cached model.

All runtime inputs are the native empty user message. Teacher requests and
teacher-forced token trajectories are isolated calibration artifacts.
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
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@contextmanager
def fixed_mean_control():
    """A labelled experimental ablation in this isolated process only."""
    from invariants import activation_explanation_map as maps
    original = maps.evaluate_compiled_map

    def evaluate(compiled, hidden):
        _delta, distance, clamped, outside = original(compiled, hidden)
        return compiled[0]['center_change'].expand_as(hidden), distance, clamped, outside

    try:
        maps.evaluate_compiled_map = evaluate
        yield
    finally:
        maps.evaluate_compiled_map = original


def run(model, subject, cache, variant, args):
    from contextlib import nullcontext
    from invariants import agentic_engine, probe_activation_explain as core
    from invariants.activation_explanation_output import assess_activation_explanation
    from invariants.config import AgenticConfig
    baseline = variant == 'baseline'
    with fixed_mean_control() if variant == 'fixed_mean_control' else nullcontext():
        result = core.generate_activation_explanation(
            model, subject, '', generate=agentic_engine.generate_agentic_text,
            config=AgenticConfig(), axis_cache=cache, map_enabled=not baseline,
            map_tokens=args.map_tokens, max_tokens=args.budget, cap_fraction=args.cap,
            subject_gain=0. if baseline else args.subject_gain,
            operation_gain=0. if baseline else args.operation_gain,
            reference_gain=0. if baseline else args.reference_gain)
    if variant == 'fixed_mean_control':
        result['metadata']['map_telemetry'].update(state_conditioned=False, static_model=True)
    result.update(variant=variant, runtime_user_body='',
        qualification=assess_activation_explanation(result['text'], result['telemetry'], args.budget))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--probe-file', type=Path, default=ROOT / 'invariants/out/probes/illeism.pt')
    parser.add_argument('--budget', type=int, choices=(64, 96), default=96)
    parser.add_argument('--map-tokens', type=int, default=24)
    parser.add_argument('--cap', type=float, default=.15)
    parser.add_argument('--subject-gain', type=float, default=.02)
    parser.add_argument('--operation-gain', type=float, default=.12)
    parser.add_argument('--reference-gain', type=float, default=.04)
    args = parser.parse_args(argv)
    if not 3 <= args.map_tokens <= 64 or not math.isfinite(args.cap) or not 0 < args.cap <= 1:
        parser.error('Calibration tokens must be 3..64 and cap must be finite in (0, 1]')
    for name in ('subject_gain', 'operation_gain', 'reference_gain'):
        value = getattr(args, name)
        if not math.isfinite(value) or not (-1 if name != 'subject_gain' else 0) <= value <= 1:
            parser.error('Gains must be finite and bounded')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    destination = ROOT / 'invariants' / 'out' / 'activation_runs' / ('state-map-' + stamp)
    destination.mkdir(parents=True, exist_ok=False)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    report = {'status': 'starting', 'model': args.model, 'local_files_only': True,
        'load_mode': '4bit', 'runtime_user_body': '', 'candidate': True,
        'semantic_truth_checked': False, 'results': [], 'budget': args.budget,
        'map_tokens': args.map_tokens, 'cap_fraction': args.cap,
        'gains': {axis: getattr(args, axis + '_gain') for axis in ('subject', 'operation', 'reference')}}
    write_json(destination / 'results.json', report)
    print(f'[State map] Artifacts: {destination}', flush=True)
    try:
        import torch
        from invariants import engine, activation_explanation_map as maps, probe_activation_explain as core
        from invariants.config import AgenticConfig
        from scripts.activation_explain_transfer_test import windows_trim_signature

        if not torch.cuda.is_available():
            report.update(status='blocked', reason='CUDA is unavailable; no model was loaded')
            return 2
        free, total = torch.cuda.mem_get_info()
        report['gpu_before'] = {'free_gib': free / 1024**3, 'total_gib': total / 1024**3}
        if free < 10 * 1024**3:
            report.update(status='blocked', reason='Less than 10 GiB GPU memory is free; no model was loaded')
            return 2
        payload = torch.load(args.probe_file, map_location='cpu', weights_only=True)
        subject = {int(layer): vector.detach().float().cpu().clone() for layer, vector in payload['direction'].items()}
        subject = {layer: vector / vector.norm() for layer, vector in subject.items()}
        report['subject_sha256'] = {str(layer): hashlib.sha256(vector.numpy().tobytes()).hexdigest() for layer, vector in subject.items()}
        print('[State map] Loading one separate cached 4-bit model.', flush=True)
        model = engine.load_model(args.model, local_files_only=True, load_mode='4bit')
        core._validated_direction(subject, tuple(sorted(subject)), 'comparison subject', width=model.d_model)
        cache = {}
        with core._scratch_config(AgenticConfig()), core._scratch_cap_telemetry():
            print('[State map] Fitting exact-prefix trajectories at the last selected layer.', flush=True)
            profiles, calibration = maps.calibrate_activation_map(
                model, (max(subject),), '', max_tokens=args.map_tokens, evaluate_held_out=True)
            # Cache precisely the same calibration for both comparison runs.
            from unittest.mock import patch
            with patch.object(maps, 'calibrate_activation_map', return_value=(profiles, calibration)):
                maps.get_or_mint_activation_map(model, subject, '', cache, max_tokens=args.map_tokens)
        torch.save({'profiles': profiles, 'calibration': calibration}, destination / 'maps.pt')
        report['calibration'] = calibration
        write_json(destination / 'results.json', report)
        with windows_trim_signature():
            for variant in ('baseline', 'fixed_mean_control', 'current_state_map'):
                print(f'[State map] Running {variant}.', flush=True)
                result = run(model, subject, cache, variant, args)
                report['results'].append(result)
                write_json(destination / (variant + '.json'), result)
                (destination / (variant + '.txt')).write_text(result['text'], encoding='utf-8')
                write_json(destination / 'results.json', report)
                print(f"[State map] {variant}: {result['telemetry']['tokens_generated']} tokens; {result['text'][:180]!r}", flush=True)
        report['status'] = 'complete'
        return 0
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        (destination / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return 1
    finally:
        write_json(destination / 'results.json', report)
        print(f"[State map] {report['status']}: {destination}", flush=True)
        if report.get('reason') or report.get('error'):
            print(report.get('reason') or report.get('error'), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
