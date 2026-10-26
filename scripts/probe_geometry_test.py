"""Read-only CPU check of saved probe diagnostics against a cached output head.

This loads no transformer blocks and generates no tokens. Reports stay in a
separate output directory; it never opens the live conception ledger for writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invariants.probe_behavior import definition_key
from invariants.probe_definition_geometric import analyze_probe_geometry, render_geometric_report


class FrozenOutputHead:
    def __init__(self, weight, config):
        from types import SimpleNamespace
        self.config = config
        self.lm_head = SimpleNamespace(weight=weight)

    def get_output_embeddings(self):
        return self.lm_head


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--probe', action='append', required=True)
    parser.add_argument('--probe-dir', type=Path, default=Path('invariants/out/probes'))
    parser.add_argument('--behavior', type=Path, default=Path('invariants/out/trigger_tuner.plasticity.behavior.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if any(not name or not all(char.isalnum() or char == '_' for char in name) for name in args.probe):
        parser.error('Probe names must be exact names without path components.')
    paths = [args.behavior] + [args.probe_dir / (name + '.pt') for name in args.probe]
    before = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    checkpoint = args.checkpoint.resolve()
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    head_key = 'lm_head.weight'
    shard_name = index['weight_map'][head_key]
    if not isinstance(shard_name, str) or not re.fullmatch(r'[A-Za-z0-9._-]+\.safetensors', shard_name):
        raise ValueError('Checkpoint index must name a snapshot shard without path components.')
    shard = checkpoint / shard_name
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    records = json.loads(args.behavior.read_text(encoding='utf-8'))['records']
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    results = []
    with safe_open(str(shard), framework='pt', device='cpu') as archive:
        # safetensors exposes the CPU tensor backed by the snapshot. The core
        # converts one bounded row chunk, preserving the original head dtype.
        weight = archive.get_tensor(head_key)
        model = FrozenOutputHead(weight, config)
        for name in args.probe:
            probe = torch.load(args.probe_dir / (name + '.pt'), map_location='cpu', weights_only=True)
            namespace = definition_key(name, probe)
            report = analyze_probe_geometry(name, probe, model, tokenizer,
                                            records.get(namespace, []), namespace=namespace)
            report['validation_source'] = {
                'checkpoint': str(checkpoint), 'head_key': head_key,
                'mode': 'cached_cpu_output_head_only', 'model_forward_passes': 0,
                'generated_tokens': 0, 'gpu_used': False,
                'limitation': 'Raw cached head projection does not test a quantized live generation trajectory.',
            }
            (args.output / (name + '.json')).write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            (args.output / (name + '.txt')).write_text(render_geometric_report(report), encoding='utf-8')
            results.append({'probe': name, 'status': report['status'],
                            'layers': len(report['vocabulary']['layers']),
                            'reply_support': report['behavior']['reply']['support'],
                            'input_support': report['behavior']['input']['support']})
            print(f"{name}: {len(report['vocabulary']['layers'])} layers; reply contrast {report['behavior']['reply']['status']}", flush=True)
    after = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise RuntimeError('A saved probe or behavior source changed during validation.')
    summary = {'source_files_unchanged': True, 'source_sha256': before,
               'model_forward_passes': 0, 'generated_tokens': 0, 'gpu_used': False,
               'results': results}
    (args.output / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
