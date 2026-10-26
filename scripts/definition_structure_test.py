"""Isolated GPU comparison of assistant prefilling and ordered schema pulses.

The USER evidence is unchanged across variants. A literal neutral assistant
prefix is varied explicitly; no teacher examples or behavioral oracle enter
generation. Intended and reversed sequences use the same prefix and gains.
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


def run_variant(model, bank, evidence, selected, variant, *, gain=.04, budget=384, cap=.08):
    import torch
    from invariants import engine
    from invariants.definition_prefill import (
        DEFAULT_DEFINITION_PREFIX, prepare_assistant_continuation, finish_assistant_continuation)
    from invariants.definition_schema_check import assess_schema_definition
    from invariants.probe_definition_schema import SCHEMA_ORDER
    from invariants.probe_definition_sequence import DefinitionSchemaSequence, build_schema_sequence_handles

    if variant not in ('baseline', 'prefill', 'sequence', 'reversed'):
        raise ValueError('Unknown definition structure variant')
    prefix = '' if variant == 'baseline' else DEFAULT_DEFINITION_PREFIX
    prompt, prefix_metadata = prepare_assistant_continuation(model, evidence, prefix)
    started = time.monotonic()
    handles, trace = [], None
    try:
        if variant in ('sequence', 'reversed'):
            # There is no actual subject probe in this synthetic benchmark.
            # A shape-only marker is required by the hook API, with its gain 0.
            subject = {layer: torch.eye(1, int(model.d_model))[0]
                       for layer in bank['genus']}
            sequence = DefinitionSchemaSequence()
            order = tuple(reversed(SCHEMA_ORDER)) if variant == 'reversed' else SCHEMA_ORDER
            handles = build_schema_sequence_handles(model, subject, bank,
                role_gains={role: gain for role in SCHEMA_ORDER}, order=order,
                cap_fraction=cap, subject_fraction=0., sequence=sequence)
        inputs = engine._inputs(model, prompt, pre_formatted=True)
        start = inputs['input_ids'].shape[-1]
        full = engine._generate_ids(model, inputs, budget, max_tool_calls=0)
        continuation = model.tok.decode(full[start:], skip_special_tokens=True)
        text = finish_assistant_continuation(prefix, continuation).strip()
        count = len(full) - start
        if variant in ('sequence', 'reversed'):
            trace = sequence.trace()
    finally:
        for handle in handles:
            handle.remove()
    qualification = assess_schema_definition(text, selected=selected)
    return dict(variant=variant, text=text, continuation=continuation,
                tokens_generated=count, budget=budget, elapsed_seconds=time.monotonic()-started,
                status='incomplete' if count >= budget else 'hypothesis' if qualification['accepted'] else 'unqualified',
                qualification=qualification, prefill=prefix_metadata, sequence=trace,
                evidence_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
                semantic_truth_checked=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--layers', default='22,23,24,25,26,27,28,29')
    parser.add_argument('--gain', type=float, default=.04)
    parser.add_argument('--budget', type=int, default=384)
    parser.add_argument('--cap', type=float, default=.08)
    parser.add_argument('--variants', default='baseline,prefill,sequence,reversed')
    parser.add_argument('--out', type=Path, default=ROOT/'invariants/out/definition_runs/20261002_structure')
    args = parser.parse_args()
    variants = tuple(args.variants.split(','))
    if not variants or len(set(variants)) != len(variants) or set(variants)-{'baseline','prefill','sequence','reversed'}:
        parser.error('Variants must be distinct baseline,prefill,sequence,reversed names')
    if not math.isfinite(args.gain) or abs(args.gain)>1 or args.budget<1:
        parser.error('Gain must be finite within [-1,1] and budget positive')
    if not math.isfinite(args.cap) or not 0<args.cap<=1:
        parser.error('Cap must be finite within (0,1]')
    layers = tuple(int(value) for value in args.layers.split(','))
    if len(set(layers))<4 or len(set(layers))%4:
        parser.error('Order comparison requires four equally sized role groups')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING','1')
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='replace',line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8',errors='replace',line_buffering=True)
    import torch
    from invariants import engine
    from invariants.definition_prefill import DEFAULT_DEFINITION_PREFIX
    from invariants.probe_definition_schema import get_or_mint_definition_schema
    from invariants.probe_behavior import render_definition_evidence
    from scripts.definition_relation_test import heldout_revision, write_json

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<10*1024**3:
        raise RuntimeError('This isolated 4-bit experiment requires 10 GiB free GPU memory')
    args.out.mkdir(parents=True,exist_ok=True)
    results=[]
    try:
        print('[Structure] Loading the separate cached model.')
        model=engine.load_model(args.model,local_files_only=True,load_mode='4bit')
        counter=[0]
        def progress(_model,_inputs):
            counter[0]+=1
            if counter[0]%16==0:
                print(f'[Structure] Calibration forward passes: {counter[0]}/192.')
        observer=model.model.register_forward_pre_hook(progress)
        try:
            bank,metadata=get_or_mint_definition_schema(model,layers,ROOT/'invariants/out/definition_schema',
                                                       assistant_prefix=DEFAULT_DEFINITION_PREFIX)
        finally:
            observer.remove()
        write_json(args.out/'profiles.json',metadata)
        evidence,selected=heldout_revision()
        evidence=render_definition_evidence('p_333333333333333333333333',selected)
        write_json(args.out/'heldout.json',dict(evidence=evidence,selected=selected,
            oracle='Lower/Memory0-1 retain 12 despite new evidence; higher/Memory2-3 revise to 41 using new evidence.',
            oracle_not_in_model_context=True))
        write_json(args.out/'settings.json',dict(model=args.model,load_mode='4bit',layers=layers,
            gain=args.gain,budget=args.budget,cap=args.cap,variants=variants,subject_gain=0.,
            calibration_forwards=counter[0],bank_identity=metadata['identity_hash'],
            evidence_format='live_probe_definition_renderer',
            later_intervention_source_shift='experimental_unverified'))
        for variant in variants:
            print(f'[Structure] Held-out generation: {variant}.')
            row=run_variant(model,bank,evidence,selected,variant,gain=args.gain,budget=args.budget,cap=args.cap)
            results.append(row)
            write_json(args.out/'results.json',results)
            (args.out/(variant+'.txt')).write_text(row['text'],encoding='utf-8')
            print(f'[Structure] {variant}: {row["status"]}; new tokens={row["tokens_generated"]}.')
            print(row['text'])
        write_json(args.out/'run_status.json',dict(status='finished',completed_variants=variants,semantic_truth_checked=False))
    except Exception as exc:
        write_json(args.out/'run_status.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',
            completed_variants=[row['variant'] for row in results],semantic_truth_checked=False))
        raise


if __name__=='__main__':
    main()
