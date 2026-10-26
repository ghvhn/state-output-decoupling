"""Measure a saved probe through a separate, offline cached model.

Only the numeric direction reaches the model. Both signs and a reversed-axis
control use the exact blank native assistant context, with no generation.
"""

import argparse
from datetime import datetime, timezone
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--probe-file', type=Path, default=ROOT / 'invariants/out/probes/illeism.pt')
    parser.add_argument('--epsilon', type=float, default=.01)
    parser.add_argument('--cap', type=float, default=.15)
    args = parser.parse_args(argv)
    if not math.isfinite(args.epsilon) or not 0 < args.epsilon <= .1:
        parser.error('Epsilon must be finite in (0, .1]')
    if not math.isfinite(args.cap) or args.cap < args.epsilon:
        parser.error('Cap must be finite and at least epsilon')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    destination = ROOT / 'invariants/out/activation_runs' / ('logit-distance-' + stamp)
    destination.mkdir(parents=True, exist_ok=False)
    report = {'status': 'starting', 'model': args.model, 'local_files_only': True,
        'load_mode': '4bit', 'runtime_user_body': '', 'semantic_truth_checked': False,
        'epsilon': args.epsilon, 'cap_fraction': args.cap, 'results': {}}
    write_json(destination / 'results.json', report)
    print(f'[Probe Logits] Artifacts: {destination}', flush=True)
    try:
        import torch
        from invariants import engine
        from invariants.probe_logit_distance import measure_probe_logit_distance, render_probe_logit_distance

        if not torch.cuda.is_available():
            report.update(status='blocked', reason='CUDA unavailable; no model loaded')
            return 2
        free, total = torch.cuda.mem_get_info()
        report['gpu_before'] = {'free_gib': free / 1024**3, 'total_gib': total / 1024**3}
        if free < 10 * 1024**3:
            report.update(status='blocked', reason='Less than 10 GiB GPU memory free; no model loaded')
            return 2
        payload = torch.load(args.probe_file, map_location='cpu', weights_only=True)
        subject = {int(layer): vector.detach().float().cpu().clone()
                   for layer, vector in payload['direction'].items()}
        print('[Probe Logits] Loading one separate cached 4-bit model.', flush=True)
        model = engine.load_model(args.model, local_files_only=True, load_mode='4bit')
        for name, direction in (('forward', subject), ('reversed', {layer: -value for layer, value in subject.items()})):
            print(f'[Probe Logits] Measuring {name} axis with five identical-context forwards.', flush=True)
            result = measure_probe_logit_distance(model, direction, '', epsilon=args.epsilon,
                cap_fraction=args.cap, check_half=True)
            report['results'][name] = result
            rendered = render_probe_logit_distance(result)
            (destination / (name + '.txt')).write_text(rendered, encoding='utf-8')
            write_json(destination / 'results.json', report)
            print(rendered, flush=True)
        first, reverse = report['results']['forward'], report['results']['reversed']
        effects = {row['id']: row['logprob_sensitivity'] for row in first['influential']}
        opposite = {row['id']: row['logprob_sensitivity'] for row in reverse['influential']}
        shared = set(effects) & set(opposite)
        report['reversal_check'] = {
            'shared_influential_tokens': len(shared),
            'distance_absolute_difference': abs(first['fisher_rao_negative_positive'] - reverse['fisher_rao_negative_positive']),
            'fisher_absolute_difference': abs(first['fisher_rms'] - reverse['fisher_rms']),
            'max_signed_effect_sum': max((abs(effects[index] + opposite[index]) for index in shared), default=None),
            'identical_context': first['metadata']['input_ids_sha256'] == reverse['metadata']['input_ids_sha256'],
        }
        report['status'] = 'complete'
        return 0
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        (destination / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return 1
    finally:
        write_json(destination / 'results.json', report)
        print(f"[Probe Logits] {report['status']}: {destination}", flush=True)
        if report.get('reason') or report.get('error'):
            print(report.get('reason') or report.get('error'), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
