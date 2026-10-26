"""Train or query an experimental memory adapter on a local HF model.

Run from the repository root with python -m scripts.train_neural_memory --help.
"""
import argparse
import json
from pathlib import Path

import torch

from invariants.engine import load_model
from invariants.neural_memory import NeuralMemory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--examples', type=Path, help='JSON list of cue/continuation objects')
    parser.add_argument('--query')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args()
    if not args.examples and not args.query:
        parser.error('Provide --examples or --query')
    host = load_model(args.model, local_files_only=True)
    layers = host.model.model.layers
    layer = args.layer % len(layers)
    memory = NeuralMemory(host.d_model, args.rank)
    memory.to(device=next(layers[layer].parameters()).device)
    if args.checkpoint.exists():
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if saved['model'] != args.model or saved['layer'] != layer:
            raise ValueError('Checkpoint model/layer does not match this run')
        memory.load_state_dict(saved['weights'])
    memory.attach(layers[layer])
    try:
        if args.examples:
            examples = json.loads(args.examples.read_text(encoding='utf-8'))
            rows, targets = [], []
            for example in examples:
                cue = host.tok.encode(example['cue'], add_special_tokens=True)
                continuation = host.tok.encode(example['continuation'], add_special_tokens=False)
                if not cue or not continuation:
                    raise ValueError('Each example needs a nonempty cue and continuation')
                rows.append(cue + continuation)
                targets.append([-100] * len(cue) + continuation)
            if not rows:
                raise ValueError('No training examples')
            width = max(map(len, rows))
            pad = host.tok.pad_token_id
            if pad is None:
                pad = host.tok.eos_token_id
            ids = torch.tensor([row + [pad] * (width-len(row)) for row in rows], device=host.device)
            labels = torch.tensor([row + [-100] * (width-len(row)) for row in targets], device=host.device)
            mask = torch.tensor([[1]*len(row) + [0]*(width-len(row)) for row in rows], device=host.device)
            report = memory.learn(host.model, ids, labels, attention_mask=mask,
                                  steps=args.steps, learning_rate=args.learning_rate)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': args.model, 'layer': layer, 'weights': memory.state_dict()}, args.checkpoint)
            print(json.dumps({k: v for k, v in report.items() if k != 'loss_history'}))
        if args.query:
            tokens = host.tok(args.query, return_tensors='pt').to(host.device)
            with torch.no_grad():
                generated = host.model.generate(**tokens, max_new_tokens=args.max_tokens, do_sample=False)
            print(host.tok.decode(generated[0, tokens['input_ids'].shape[1]:], skip_special_tokens=True))
    finally:
        memory.detach()


if __name__ == '__main__':
    main()
