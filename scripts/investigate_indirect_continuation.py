"""Select short probe-guided continuations before rendering the chosen answer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    args = argparse.ArgumentParser(description=__doc__)
    prompt = args.add_mutually_exclusive_group(required=True)
    prompt.add_argument('--prompt', help='User prompt; supplied once to the native chat template')
    prompt.add_argument('--messages', type=Path, help='JSON list of role/content messages ending with a user turn')
    args.add_argument('--minimize', nargs='+', action='extend', default=[])
    args.add_argument('--maximize', nargs='+', action='extend', default=[])
    args.add_argument('--via', nargs='+', action='extend')
    args.add_argument('--probe-dir', type=Path, default=ROOT / 'invariants/out/probes')
    args.add_argument('--controller', type=Path, default=ROOT / 'invariants/out/trigger_tuner.plasticity.indirect.json')
    args.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    args.add_argument('--load-mode', default='4bit', choices=['4bit', 'full', 'cpu', 'cpu16', 'slow'])
    args.add_argument('--tokens', type=int, default=64)
    args.add_argument('--stride', type=int, default=8)
    args.add_argument('--horizon', type=int, default=12)
    args.add_argument('--helpers', type=int, default=2)
    args.add_argument('--joint', type=int, default=2, help='Maximum helper axes applied together in one candidate')
    args.add_argument('--candidates', type=int, default=33, help='Maximum candidate arms, including baseline and all single helpers')
    args.add_argument('--mix', type=int, default=4, help='Mixture weight grid denominator, 1..64; 1 tests equal shares only')
    args.add_argument('--distribution', default='grid', help='Mixture distribution mode (grid, flexible, continuous, dirichlet, adaptive, softmax, orthogonal, subspace, layer_tilted, uniform)')
    args.add_argument('--temperature', type=float, default=1.0, help='Simplex mixture temperature (>0)')
    args.add_argument('--novel', action='store_true', help='Synthesize frameless orthogonalized subspace candidate arms')
    args.add_argument('--layer-tilt', type=float, default=0.0, help='Depth tilt across intervention layers in [-2.0, 2.0]')
    args.add_argument('--alpha', type=float, default=.25)
    args.add_argument('--cap-fraction', type=float, default=.02)
    args.add_argument('--layers', nargs=2, type=int, metavar=('FIRST', 'LAST'))
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--decisions', type=int)
    args.add_argument('--candidate-tokens', type=int)
    args.add_argument('--output', type=Path, required=True)
    return args


def main(argv=None):
    args = parser().parse_args(argv)
    from invariants.continuation_investigation import run_continuation_investigation
    from invariants.engine import load_model
    from invariants.indirect_steering import IndirectSteering
    from scripts.investigate_indirect_backfill import load_probe_bank, route_controls
    controller = IndirectSteering(args.controller)
    controls = args.via
    objectives = {}
    for sign, names in ((-1, args.minimize), (1, args.maximize)):
        for name in names:
            if name in objectives:
                raise ValueError(f'Duplicate objective: {name}')
            objectives[name] = sign
    if not objectives:
        routes = {name: route for name, route in controller.routes.items() if route.get('enabled', True)}
        if any(route.get('when') for route in routes.values()):
            raise ValueError('Continuation investigation requires explicit objectives for guarded routes')
        objectives = {name: route['direction'] for name, route in routes.items()}
        if controls is None:
            controls = route_controls(routes)
    if not objectives:
        raise ValueError('Supply --minimize/--maximize or configure active indirect objectives')
    messages = (json.loads(args.messages.read_text(encoding='utf-8-sig')) if args.messages
                else [{'role': 'user', 'content': args.prompt}])
    band = args.layers
    if band is not None and band[0] > band[1]:
        raise ValueError('Layer FIRST must be no greater than LAST')
    model = load_model(args.model, local_files_only=True, load_mode=args.load_mode)
    probes = load_probe_bank(model, args.probe_dir, controller)
    print(f'[Continuation] device={model.device}; stride={args.stride}; horizon={args.horizon}; objectives={objectives}', flush=True)
    extra_kwargs = {}
    if args.distribution != 'grid':
        extra_kwargs['distribution'] = args.distribution
    if args.temperature != 1.0:
        extra_kwargs['temperature'] = args.temperature
    if args.novel:
        extra_kwargs['compose_novel'] = True
    if args.layer_tilt != 0.0:
        extra_kwargs['layer_tilt'] = args.layer_tilt
    result = run_continuation_investigation(model, probes, objectives, controls=controls,
        prompt_messages=messages, output_dir=args.output, max_new_tokens=args.tokens,
        stride=args.stride, horizon=args.horizon, max_helpers=args.helpers,
        max_joint_helpers=args.joint, max_candidates=args.candidates,
        mixture_resolution=args.mix,
        alpha=args.alpha, cap_fraction=args.cap_fraction,
        layers=None if band is None else list(range(band[0], band[1] + 1)), seed=args.seed,
        max_decisions=args.decisions, candidate_token_budget=args.candidate_tokens,
        progress=lambda message: print(f'[Continuation] {message}', flush=True),
        **extra_kwargs)
    print('\n' + result['response'], flush=True)
    print(f"\n[Continuation] Selected {result['telemetry']['tokens_generated']} tokens; artifacts: {result['output_dir']}", flush=True)
    return result


if __name__ == '__main__':
    main()
