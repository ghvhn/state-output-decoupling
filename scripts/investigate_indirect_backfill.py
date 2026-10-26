"""Controlled fresh generations on archived conversation contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--archive', type=Path, default=ROOT / 'invariants/out/interactive_memory.jsonl')
    args.add_argument('--probe-dir', type=Path, default=ROOT / 'invariants/out/probes')
    args.add_argument('--controller', type=Path, default=ROOT / 'invariants/out/trigger_tuner.plasticity.indirect.json')
    args.add_argument('--minimize', nargs='+', action='extend', default=[])
    args.add_argument('--maximize', nargs='+', action='extend', default=[])
    args.add_argument('--via', nargs='+', action='extend')
    args.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    args.add_argument('--load-mode', default='4bit', choices=['4bit', 'full', 'cpu', 'cpu16', 'slow'])
    args.add_argument('--scope')
    args.add_argument('--case-id', action='append', default=[], help='Archived assistant record ID, repeatable')
    args.add_argument('--limit', type=int, default=4)
    args.add_argument('--context-turns', type=int, default=8)
    args.add_argument('--helpers', type=int, default=3)
    args.add_argument('--tokens', type=int, default=64)
    args.add_argument('--alpha', type=float, default=.25)
    args.add_argument('--cap-fraction', type=float, default=.02)
    args.add_argument('--layers', nargs=2, type=int, metavar=('FIRST', 'LAST'), help='Inclusive decoder layer indices')
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--repeats', type=int, default=1)
    args.add_argument('--no-random-control', action='store_true')
    args.add_argument('--no-resume', action='store_true')
    args.add_argument('--output', type=Path, required=True)
    return args


def load_records(path, audit=None):
    records = []
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            padding = len(text) - len(text.lstrip('\x00'))
            if padding:
                text = text[padding:]
                note = f'Archive line {line}: ignored {padding} leading NUL padding bytes; original file preserved'
                print(f'[Replay] {note}', flush=True)
                if audit is not None:
                    audit.append(note)
            try:
                records.append(json.loads(text))
            except ValueError as exc:
                raise ValueError(f'Invalid archive JSON at line {line}') from exc
    return records


def load_probe_bank(model, probe_dir, controller):
    import torch
    probes = {}
    for path in sorted(Path(probe_dir).glob('*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        directions = payload.get('direction') or {}
        if not isinstance(directions, dict) or not directions:
            continue
        if any(int(layer) not in range(model.n_layers) or torch.as_tensor(vector).numel() != model.d_model
               for layer, vector in directions.items()):
            continue
        probes[path.stem] = dict(payload, measurement_source='live_assistant_prediction',
                                 role=controller.roles.get(path.stem, 'unclassified'))
    return probes


def route_controls(routes):
    """Union active route constraints; any automatic route retains discovery."""
    if any(route['controls'] is None for route in routes.values()):
        return None
    signed = {}
    extras = {}
    for route in routes.values():
        for helper in route['controls']:
            name, sign = helper['name'], helper.get('sign')
            signed[name] = sign if name not in signed else (sign if signed[name] == sign else None)
            meta = extras.setdefault(name, {})
            if helper.get('prior_weight') is not None and 'prior_weight' not in meta:
                meta['prior_weight'] = helper['prior_weight']
            if helper.get('layer_weights') and 'layer_weights' not in meta:
                meta['layer_weights'] = dict(helper['layer_weights'])
    return [{'name': name, 'sign': sign, **extras.get(name, {})} for name, sign in signed.items()] or None


def main(argv=None):
    args = parser().parse_args(argv)
    from invariants.indirect_steering import IndirectSteering
    from invariants.replay_investigation import select_archive_cases, run_investigation
    controller = IndirectSteering(args.controller)
    controls = args.via
    objectives = {}
    for direction, names in ((-1, args.minimize), (1, args.maximize)):
        for name in names:
            if name in objectives:
                raise ValueError(f'Duplicate objective: {name}')
            objectives[name] = direction
    if not objectives:
        routes = {name: route for name, route in controller.routes.items() if route.get('enabled', True)}
        objectives = {name: route['direction'] for name, route in routes.items()}
        if controls is None:
            controls = route_controls(routes)
    if not objectives:
        raise ValueError('Supply --minimize/--maximize, or configure active indirect routes first')
    archive_notes = []
    records = load_records(args.archive, audit=archive_notes)
    count = len(records) if args.case_id else args.limit
    cases = select_archive_cases(records, limit=count, context_turns=args.context_turns, scope=args.scope)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.get('case_id') in wanted or case.get('source_record_id') in wanted]
        if len(cases) != len(wanted):
            raise ValueError('One or more requested case IDs do not identify usable archived assistant responses')
        cases = cases[-args.limit:]
    if not cases:
        raise ValueError('The archive contains no usable paired conversation cases')
    archive_provenance = {'path': str(args.archive.resolve()),
                          'sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                          'read_notes': archive_notes}
    for case in cases:
        case['archive_input'] = archive_provenance
    print(f'[Replay] {len(cases)} reconstructed transcript case(s); original dynamic context is not fully known.', flush=True)
    print(f'[Replay] Objectives={objectives}; helpers={controls or "automatic"}; output={args.output}', flush=True)
    from invariants.engine import load_model
    model = load_model(args.model, local_files_only=True, load_mode=args.load_mode)
    probes = load_probe_bank(model, args.probe_dir, controller)
    required = set(objectives) | {item['name'] if isinstance(item, dict) else item.lstrip('+-') for item in (controls or [])}
    if required - probes.keys():
        raise ValueError(f'Unknown or incompatible probes: {sorted(required - probes.keys())}')
    band = args.layers or [max(0, int(model.n_layers * .4)), max(0, int(model.n_layers * .7) - 1)]
    if band[0] < 0 or band[0] > band[1] or band[1] >= model.n_layers:
        raise ValueError(f'Invalid inclusive layer range: {band}')
    print(f'[Replay] Device={model.device}; layers={band[0]}..{band[1]}; tokens={args.tokens}; alpha={args.alpha}; cap={args.cap_fraction}', flush=True)
    summary = run_investigation(model, probes, objectives, controls=controls, cases=cases,
        output_dir=args.output, alpha=args.alpha, cap_fraction=args.cap_fraction,
        layers=list(range(band[0], band[1] + 1)), seed=args.seed,
        max_new_tokens=args.tokens, max_helpers=args.helpers, repeats=args.repeats,
        random_control=not args.no_random_control, resume=not args.no_resume,
        progress=lambda message: print(message, flush=True))
    print(json.dumps({'source': summary.get('source'), 'output': str(args.output.resolve()),
                      'effects': summary.get('effects', []),
                      'completed_trials': summary.get('completed_trials'),
                      'expected_trials': summary.get('expected_trials')}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    main()
