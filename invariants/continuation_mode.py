"""Session-local controls for probe-guided branching across ordinary turns."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import uuid


from invariants.continuation_investigation import ALLOWED_CONTINUATION_DISTRIBUTIONS

DEFAULTS = dict(tokens=64, stride=8, horizon=12, helpers=2, joint=2, candidates=33, mix=4, alpha=.25, cap=.02,
                seed=0, context=8, decisions=None, candidate_tokens=None,
                layers=None, output=None, distribution=None, temperature=None,
                novel=None, layer_tilt=None)


class ContinuationMode:
    def __init__(self):
        self.enabled = False
        self.options = deepcopy(DEFAULTS)
        self.completed_turns = self.failed_turns = 0
        self.last_output = self.last_error = None

    def status(self):
        return dict(enabled=self.enabled, options=deepcopy(self.options),
                    completed_turns=self.completed_turns, failed_turns=self.failed_turns,
                    last_output=self.last_output, last_error=self.last_error,
                    lifetime='current shell session; current controller owner')

    def configure(self, tokens, n_layers=None):
        tokens = list(tokens)
        action = tokens.pop(0) if tokens else 'status'
        if action not in {'on', 'off', 'status'}:
            raise ValueError('Usage: continue on [options] | off | status')
        if action != 'on':
            if tokens:
                raise ValueError(f'continue {action} accepts no options')
            if action == 'off':
                self.enabled = False
            return self.status()
        staged = deepcopy(self.options)
        while tokens:
            key = tokens.pop(0)
            if key not in staged:
                raise ValueError(f'Unknown continuation option: {key}')
            if key == 'layers':
                if len(tokens) < 2:
                    raise ValueError('layers requires inclusive FIRST LAST indices')
                first, last = int(tokens.pop(0)), int(tokens.pop(0))
                if first < 0 or last < first or (n_layers is not None and last >= n_layers - 1):
                    raise ValueError('Intervention layers must be ordered and below the final readout layer')
                staged[key] = [first, last]
                continue
            if not tokens:
                raise ValueError(f'{key} requires a value')
            value = tokens.pop(0)
            if key == 'output':
                if not str(value).strip():
                    raise ValueError('output requires a directory')
                staged[key] = str(value)
            elif key == 'distribution':
                val_str = str(value).strip().lower()
                if val_str not in ALLOWED_CONTINUATION_DISTRIBUTIONS:
                    raise ValueError(f'distribution must be one of {", ".join(ALLOWED_CONTINUATION_DISTRIBUTIONS)}')
                staged[key] = val_str
            elif key == 'novel':
                val_str = str(value).strip().lower()
                if val_str not in {'on', 'off', 'true', 'false', '1', '0'}:
                    raise ValueError('novel requires on or off')
                staged[key] = val_str in {'on', 'true', '1'}
            elif key == 'layer_tilt':
                val_str = str(value).strip().lower()
                if val_str in {'on', 'off'}:
                    staged[key] = 0.35 if val_str == 'on' else 0.0
                else:
                    number = float(value)
                    if not math.isfinite(number) or abs(number) > 2.0:
                        raise ValueError('layer_tilt must be a finite number in [-2.0, 2.0]')
                    staged[key] = number
            elif key in {'alpha', 'cap', 'temperature'}:
                number = float(value)
                if not math.isfinite(number) or number <= 0:
                    raise ValueError(f'{key} must be positive and finite')
                staged[key] = number
            else:
                number = int(value)
                if number < (0 if key == 'seed' else 1):
                    raise ValueError(f'{key} must be a positive integer (seed may be zero)')
                if key == 'mix' and number > 64:
                    raise ValueError('mix must be between 1 and 64')
                staged[key] = number
        # Revalidate a retained band when switching the model in this session.
        if staged['layers'] is not None and n_layers is not None and staged['layers'][1] >= n_layers - 1:
            raise ValueError('Saved intervention layers do not fit the current model')
        self.options, self.enabled = staged, True
        return self.status()

    def arguments(self, prompt):
        if not self.enabled:
            raise ValueError('Continuation mode is off')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Supply an ordinary user prompt')
        args = ['continue']
        for key, value in self.options.items():
            if value is None:
                continue
            if key == 'layers':
                args.extend(['layers', *(str(item) for item in value)])
            elif key == 'output':
                child = f'turn_{self.completed_turns + 1:06d}_{uuid.uuid4().hex[:12]}'
                args.extend(['output', str(Path(value) / child)])
            elif key == 'novel':
                args.extend(['novel', 'on' if value else 'off'])
            else:
                args.extend([key, str(value)])
        return [*args, '--', prompt]

    def record_result(self, result):
        self.completed_turns += 1
        self.last_output = result.get('output_dir')
        self.last_error = None

    def record_failure(self, error):
        self.failed_turns += 1
        self.last_error = str(error)
