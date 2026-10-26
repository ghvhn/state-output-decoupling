"""Scheduled definition operations applied through activation directions.

Token windows are an operational heuristic, not a detector of completed
comparison, isolation, distillation, or verification. The trace describes
observed model forwards and residual injection, not semantic success.
"""

import math

import torch
from invariants.probe_definition_steps import STEP_ORDER, STEP_GAINS


DEFAULT_STEP_TOKENS = 96
PROCESS_VERSION = 'definition-step-output-windows-v1'


def _token_window(value):
    try:
        number = float(value)
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Definition step tokens must be a positive finite integer') from exc
    if isinstance(value, bool) or not math.isfinite(number) or number != count or count <= 0:
        raise ValueError('Definition step tokens must be a positive finite integer')
    return count


def _gains(values, defaults=None):
    result = dict(STEP_GAINS if defaults is None else defaults)
    if values is not None:
        if not hasattr(values, 'items'):
            raise ValueError('Definition step gains must map scheduled steps to finite signed gains')
        for name, value in values.items():
            if name not in STEP_ORDER:
                raise ValueError(f'Unknown definition process step {name}')
            try:
                gain = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Definition step gain for {name} must be finite and between -1 and 1') from exc
            if not math.isfinite(gain) or abs(gain) > 1:
                raise ValueError(f'Definition step gain for {name} must be finite and between -1 and 1')
            result[name] = gain
    return result


class DefinitionStepProcess:
    """One invocation's absolute output-position schedule and diagnostics."""

    def __init__(self, step_tokens=DEFAULT_STEP_TOKENS, *, step_gains=None):
        self.step_tokens = _token_window(step_tokens)
        self._gains = _gains(step_gains)
        self._epoch = 0
        self._positions = None
        self._ranges = None
        self._decision_position = None
        self._furthest = -1
        self._entered = set()
        self._events = []
        self._injected = {step: set() for step in STEP_ORDER}
        self._claimed = False

    @property
    def budget_positions(self):
        return self.step_tokens * len(STEP_ORDER)

    @property
    def gains(self):
        return dict(self._gains)

    def begin_forward(self):
        """Called once at the model root, before any participating layer."""
        self._epoch += 1
        self._positions = None
        self._ranges = None

    def positions(self, sequence_length):
        """Derive positions once per forward, then reuse them across layers."""
        if self._epoch == 0:
            raise RuntimeError('Definition process positions require the model forward prehook')
        if self._positions is not None:
            if len(self._positions) != sequence_length:
                raise ValueError('Definition process layers must share a sequence length within each forward')
            return self._positions
        if sequence_length <= 0:
            raise ValueError('Definition process needs a nonempty token sequence')
        if self._decision_position is None:
            self._decision_position = sequence_length - 1
            start = -self._decision_position
        elif sequence_length == 1:
            # A cached decode contains the next generated input position.
            start = self._furthest + 1
        else:
            # Re-prefills restore each historical token's original operation.
            # Repeating an already-seen prefix cannot advance the cursor.
            start = -self._decision_position
        self._positions = tuple(range(start, start + sequence_length))
        self._furthest = max(self._furthest, self._positions[-1])
        ranges = []
        for index, step in enumerate(STEP_ORDER):
            window_start = index * self.step_tokens
            window_stop = window_start + self.step_tokens
            begin = max(start, window_start)
            end = min(start + sequence_length, window_stop)
            if begin < end:
                ranges.append((step, begin - start, end - start))
            if self._furthest >= window_start and step not in self._entered:
                self._entered.add(step)
                self._events.append({'step': step, 'start': window_start, 'stop': window_stop,
                                     'first_observed_forward': self._epoch})
        self._ranges = tuple(ranges)
        return self._positions

    def ranges(self):
        if self._ranges is None:
            raise RuntimeError('Definition process ranges require residual positions first')
        return self._ranges

    def record_additions(self, start, nonzero):
        """Deduplicate actual capped additions across layers and rebuilds."""
        for offset, added in enumerate(nonzero):
            position = start + offset
            if not added or not 0 <= position < self.budget_positions:
                continue
            step = STEP_ORDER[position // self.step_tokens]
            if self._gains[step]:
                self._injected[step].add(position)

    def trace(self):
        windows = []
        for index, step in enumerate(STEP_ORDER):
            start = index * self.step_tokens
            stop = start + self.step_tokens
            windows.append({'step': step, 'start': start, 'stop': stop,
                            'gain': self._gains[step], 'entered': step in self._entered,
                            'scheduled_positions': max(0, min(self._furthest + 1, stop) - start),
                            'injected_positions': len(self._injected[step])})
        return {'diagnostic_only': True, 'process_version': PROCESS_VERSION,
                'schedule': 'absolute output-position windows',
                'step_order': list(STEP_ORDER), 'step_tokens': self.step_tokens,
                'budget_positions': self.budget_positions,
                'decision_position': self._decision_position,
                'furthest_output_position': self._furthest,
                'observed_output_positions': max(0, self._furthest + 1),
                'forward_count': self._epoch, 'windows': windows,
                'events': [dict(event) for event in self._events]}


def build_definition_process_handles(model, subject, step_bank, *, step_gains=None,
                                     step_tokens=None, cap_fraction=None,
                                     subject_fraction=.02, process=None):
    """Return removable root and residual hooks for four scheduled operations.

    The original assistant decision position is output position zero. The
    fixed probe subject pulses only there. Exactly one scheduled operation
    contributes per output position, and all contributions share one live cap.
    A supplied controller remains available for diagnostics after hook removal.
    """
    from invariants.engine import _cap_steer

    if not math.isfinite(subject_fraction) or not 0 <= subject_fraction <= 1:
        raise ValueError('Definition subject fraction must be finite and between zero and one')
    if process is None:
        process = DefinitionStepProcess(DEFAULT_STEP_TOKENS if step_tokens is None else step_tokens,
                                        step_gains=step_gains)
    elif not isinstance(process, DefinitionStepProcess):
        raise ValueError('Definition process must be a DefinitionStepProcess controller')
    elif step_tokens is not None and _token_window(step_tokens) != process.step_tokens:
        raise ValueError('Definition step tokens must match the supplied process controller')
    if process._claimed or process._epoch:
        raise ValueError('Definition process controller must be fresh for each invocation')
    gains = _gains(step_gains, process.gains)
    if not subject:
        raise ValueError('Definition process requires subject layers')
    layers = tuple(sorted(subject))
    for step in STEP_ORDER:
        if step not in step_bank:
            raise ValueError(f'Definition process step {step} requires a matching axis')
        if set(step_bank[step]) != set(subject):
            raise ValueError(f'Definition process step {step} must exactly match subject layers')

    # Validate every requested vector before registering any hook, including
    # zero-gain stages, so the stored bank remains a complete four-step plan.
    vectors = {}
    for layer in layers:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError('Definition process subject layers must be nonnegative integer indices')
        marker = subject[layer].detach().to(model.device).float()
        norm = marker.norm()
        if (marker.ndim != 1 or not torch.isfinite(marker).all() or
                not torch.isfinite(norm) or norm.item() <= 1e-8):
            raise ValueError(f'Definition process subject must be finite and nonzero at layer {layer}')
        marker = marker / norm
        directions = {}
        for step in STEP_ORDER:
            vector = step_bank[step][layer].detach().to(model.device).float()
            norm = vector.norm()
            if (vector.shape != marker.shape or not torch.isfinite(vector).all() or
                    not torch.isfinite(norm) or norm.item() <= 1e-8):
                raise ValueError(f'Definition process step {step} must match the subject and be finite and nonzero at layer {layer}')
            directions[step] = vector / norm
        vectors[layer] = (marker, directions)

    root = model.model
    if not hasattr(root, 'register_forward_pre_hook'):
        raise ValueError('Definition process requires a model root with forward prehooks')
    handles = []
    try:
        handles.append(root.register_forward_pre_hook(lambda _module, _inputs: process.begin_forward()))
        for layer in layers:
            marker, directions = vectors[layer]

            def hook(_module, _inputs, output, marker=marker, directions=directions):
                h = output[0] if isinstance(output, tuple) else output
                if h.ndim < 2:
                    raise ValueError('Definition process residuals need a token or batch dimension')
                sequence_length = h.shape[-2] if h.ndim >= 3 else 1
                positions = process.positions(sequence_length)
                with torch.no_grad():
                    scale = h.float().norm(dim=-1, keepdim=True)
                    add = torch.zeros_like(h)
                    for step, begin, end in process.ranges():
                        gain = gains[step]
                        if not gain:
                            continue
                        direction = directions[step].to(h.device)
                        if h.ndim >= 3:
                            add[..., begin:end, :] += (scale[..., begin:end, :] * gain * direction).to(h.dtype)
                        else:
                            add += (scale * gain * direction).to(h.dtype)
                    if 0 in positions:
                        decision = positions.index(0)
                        if h.ndim >= 3:
                            add[..., decision, :] += (subject_fraction * scale[..., decision, :] * marker.to(h.device)).to(h.dtype)
                        else:
                            add += (subject_fraction * scale * marker.to(h.device)).to(h.dtype)
                    capped = _cap_steer(add, h, cap_fraction=cap_fraction)
                    # One small, bounded mask per layer records actual additions;
                    # original evidence positions never enter the diagnostic mask.
                    begin = max(0, -positions[0])
                    end = min(sequence_length, process.budget_positions - positions[0])
                    if begin < end:
                        if h.ndim >= 3:
                            present = capped[..., begin:end, :].float().norm(dim=-1) > 0
                            mask = present.reshape(-1, end - begin).any(dim=0).tolist()
                        else:
                            mask = [bool((capped.float().norm(dim=-1) > 0).any().item())]
                        process.record_additions(positions[0] + begin, mask)
                    changed = h + capped
                return (changed,) + tuple(output[1:]) if isinstance(output, tuple) else changed

            handles.append(root.model.layers[layer].register_forward_hook(hook))
        process._gains = gains
        process._claimed = True
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles
