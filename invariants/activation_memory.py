"""Bounded associative activation trajectories, with individually weighted traces.

No target answers, loss, token edits, or memory prose enter this mechanism.
Keys are earlier states of the same layer; values are experienced later states.
These representations are associations, not independently verified facts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import threading
import time
import uuid

import torch


def unit(value):
    value = value.float()
    scaled = value / value.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return scaled / scaled.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def strength_value(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 4:
        raise ValueError("Memory strength must be finite and between 0 and 4")
    return value


class ActivationMemory:
    def __init__(self, capacity=256):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Memory capacity must be a positive integer")
        self.capacity = capacity
        self.records = {}
        self._lock = threading.RLock()

    def write(self, key, value, *, strength=1.0, origin="thought", metadata=None,
              memory_id=None):
        strength = strength_value(strength)
        key, value = [item.detach().float().cpu().reshape(-1).clone() for item in (key, value)]
        if key.shape != value.shape or not key.numel():
            raise ValueError("Associated states must have the same nonempty width")
        if any(not torch.isfinite(item).all() or not item.abs().max() > 0 for item in (key, value)):
            raise ValueError("Associated states must be finite and nonzero")
        memory_id = memory_id or uuid.uuid4().hex
        record = dict(key=unit(key), value=unit(value), strength=strength,
                      origin=str(origin), created_at=time.time(), metadata=deepcopy(metadata or {}))
        with self._lock:
            if self.records and next(iter(self.records.values()))['key'].shape != key.shape:
                raise ValueError("Memory width mismatch")
            self.records[memory_id] = record
            while len(self.records) > self.capacity:
                # Capacity is explicit; weakest associations go first, oldest on ties.
                victim = min(self.records, key=lambda name: (self.records[name]['strength'],
                                                            self.records[name]['created_at']))
                del self.records[victim]
        return memory_id

    def set_strength(self, memory_id, strength):
        strength = strength_value(strength)
        with self._lock:
            self.records[memory_id]['strength'] = strength

    def set_capacity(self, capacity):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Memory capacity must be a positive integer")
        with self._lock:
            self.capacity = capacity
            while len(self.records) > capacity:
                victim = min(self.records, key=lambda name: (self.records[name]['strength'],
                                                            self.records[name]['created_at']))
                del self.records[victim]

    def read(self, hidden, *, hops=2, threshold=0.5, navigation_gain=1.0,
             max_norm_ratio=0.05, absolute_budget=0.25, episode_binding=True,
             link_gain=0.5, top_k=4):
        if not isinstance(hops, int) or not 0 <= hops <= 16:
            raise ValueError("Recall hops must be an integer between 0 and 16")
        if not math.isfinite(threshold) or not 0 <= threshold < 1:
            raise ValueError("Recall threshold must be in [0, 1)")
        if any(not math.isfinite(v) or v < 0 for v in (navigation_gain, max_norm_ratio, absolute_budget, link_gain)):
            raise ValueError("Recall budgets must be finite and nonnegative")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError('Recall top_k must be positive')
        stats = dict(hops=0, matched_memories=0, max_delta_norm=0.0,
                     linked_matches=0, episode_locked_positions=0)
        with self._lock:
            names = list(self.records)
            records = list(self.records.values())
        if not records or not hops:
            return hidden, stats
        keys = torch.stack([r['key'] for r in records]).to(hidden.device)
        values = torch.stack([r['value'] for r in records]).to(hidden.device)
        if hidden.shape[-1] != keys.shape[-1]:
            raise ValueError("Live state and memory widths differ")
        strengths = torch.tensor([r['strength'] for r in records], device=hidden.device)
        # Metadata binds a trajectory, never supplies semantic content to the model.
        episodes = {}
        groups = []
        for record in records:
            episode = record['metadata'].get('session_id')
            groups.append(episodes.setdefault(episode, len(episodes)) if episode else -1)
        indices = {name: index for index, name in enumerate(names)}
        predecessors = []
        for index, record in enumerate(records):
            predecessor = indices.get(record['metadata'].get('predecessor_id'), -1)
            # Never traverse an accidental/corrupt cross-episode pointer.
            if predecessor >= 0 and (groups[index] < 0 or groups[index] != groups[predecessor]):
                predecessor = -1
            predecessors.append(predecessor)
        predecessor_indices = torch.tensor(predecessors, device=hidden.device)
        groups = torch.tensor(groups, device=hidden.device)
        query = unit(hidden)
        used = torch.zeros((*hidden.shape[:-1], len(records)), device=hidden.device, dtype=torch.bool)
        accumulated = torch.zeros_like(hidden, dtype=torch.float32)
        previous_gates = torch.zeros_like(used, dtype=torch.float32)
        locked_group = None
        for _ in range(hops):
            match = query @ keys.T
            gates = ((match - threshold) / (1 - threshold)).clamp(0, 1).square()
            linked = previous_gates[..., predecessor_indices.clamp_min(0)] * link_gain
            linked = linked.masked_fill(predecessor_indices < 0, 0)
            gates = torch.maximum(gates, linked)
            gates = gates.masked_fill(used | (strengths == 0), 0)
            if episode_binding:
                if locked_group is None:
                    scores = gates * strengths
                    winners = scores.argmax(-1)
                    locked_group = groups[winners]
                    locked_group = torch.where(scores.amax(-1) > 0, locked_group, -1)
                    stats['episode_locked_positions'] = int((locked_group >= 0).sum())
                gates = gates.masked_fill((locked_group.unsqueeze(-1) >= 0) & (groups != locked_group.unsqueeze(-1)), 0)
            # Sparse competition avoids averaging every loosely similar memory.
            winners = (gates * strengths).topk(min(top_k, len(records)), dim=-1).indices
            selected = torch.zeros_like(used).scatter_(-1, winners, True)
            gates = gates.masked_fill(~selected, 0)
            active = gates > 0
            if not bool(active.any()):
                break
            # A trace can participate at most once per position in this read.
            used |= active
            stats['linked_matches'] += int((active & (linked > 0)).sum())
            previous_gates = gates
            recalled = (gates * strengths) @ values / gates.sum(-1, keepdim=True).clamp_min(1)
            accumulated += recalled
            query = unit(query + navigation_gain * recalled)
            stats['hops'] += 1
        budget = (hidden.float().norm(dim=-1, keepdim=True) * max_norm_ratio).clamp(max=absolute_budget)
        delta = accumulated * (budget / accumulated.norm(dim=-1, keepdim=True).clamp_min(1e-30)).clamp(max=1)
        stats['matched_memories'] = int(used.sum())
        stats['max_delta_norm'] = float(delta.detach().norm(dim=-1).max())
        return hidden + delta.to(hidden.dtype), stats

    def state_dict(self):
        with self._lock:
            return deepcopy(dict(capacity=self.capacity, records=self.records))

    @classmethod
    def from_state_dict(cls, data):
        result = cls(data['capacity'])
        if len(data['records']) > result.capacity:
            raise ValueError("Checkpoint exceeds memory capacity")
        for name, record in data['records'].items():
            for vector in (record['key'], record['value']):
                if vector.ndim != 1 or not torch.isfinite(vector).all() or not torch.isclose(vector.float().norm(), torch.tensor(1.0), atol=1e-5):
                    raise ValueError("Stored activation states must be finite unit vectors")
            result.write(record['key'], record['value'], strength=record['strength'],
                         origin=record['origin'], metadata=record['metadata'], memory_id=name)
            created = float(record['created_at'])
            if not math.isfinite(created):
                raise ValueError("Invalid memory timestamp")
            # Stored vectors are already normalized. Preserve bits for exact reload.
            result.records[name].update(key=record['key'].detach().cpu().float().clone(),
                                        value=record['value'].detach().cpu().float().clone(),
                                        created_at=created)
        # Older banks already stored session/position metadata. Recover only
        # unambiguous within-session order, so existing experiences gain links.
        episodes = {}
        for name, record in result.records.items():
            metadata = record['metadata']
            episode, position = metadata.get('session_id'), metadata.get('token_position')
            if episode and isinstance(position, int):
                episodes.setdefault(episode, []).append((position, name))
        for entries in episodes.values():
            ordered = sorted(entries)
            duplicate_positions = {position for position, _ in ordered
                                   if sum(p == position for p, _ in ordered) > 1}
            previous = None
            for position, name in ordered:
                if position in duplicate_positions:
                    previous = None
                    continue
                result.records[name]['metadata'].setdefault('predecessor_id', previous)
                previous = name
        return result


@dataclass
class ActivationSession:
    """One generation's recording context; all coordinates are token positions.

    Prefill repeats are skipped across chunks. Only actual forwarded states are
    recorded: the final sampled token may have no state until the next prefill.
    Context can contain user, tool, or older conversation text; it is not labeled
    as a newly authored model thought.
    """
    strength: float = 1.0
    write_enabled: bool = True
    hops: int = 2
    threshold: float = 0.5
    navigation_gain: float = 1.0
    max_norm_ratio: float = 0.05
    absolute_budget: float = 0.25
    max_writes_per_forward: int = 4
    capacity: int = 256
    running: bool = False
    episode_binding: bool = True
    link_gain: float = 0.5
    top_k: int = 4
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    segment_length: int = 0
    calls: dict = field(default_factory=dict)
    seen: dict = field(default_factory=dict)
    previous: dict = field(default_factory=dict)
    previous_ids: dict = field(default_factory=dict)
    thought_ranges: list = field(default_factory=list)
    stats: dict = field(default_factory=lambda: dict(writes=0, matched_memories=0,
                                                    max_delta_norm=0.0, max_hops=0,
                                                    linked_matches=0, episode_locked_positions=0))

    def __post_init__(self):
        self.strength = strength_value(self.strength)
        if not isinstance(self.capacity, int) or self.capacity < 1:
            raise ValueError("Memory capacity must be a positive integer")
        if not isinstance(self.hops, int) or not 0 <= self.hops <= 16:
            raise ValueError("Recall hops must be an integer between 0 and 16")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold < 1:
            raise ValueError("Recall threshold must be in [0, 1)")
        if any(not math.isfinite(value) or value < 0 for value in
               (self.navigation_gain, self.max_norm_ratio, self.absolute_budget, self.link_gain)):
            raise ValueError("Recall budgets must be finite and nonnegative")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError('Recall top_k must be positive')
        if not isinstance(self.max_writes_per_forward, int) or self.max_writes_per_forward < 1:
            raise ValueError("Writes per forward must be positive")

    def begin_segment(self, prompt_length):
        self.segment_length = int(prompt_length)
        self.calls.clear()
        self.running = True

    def end_segment(self):
        self.running = False

    def mark_generated(self, start, end):
        self.thought_ranges.append((int(start), int(end)))

    def observe(self, layer, bank, raw):
        if raw.ndim != 3 or raw.shape[0] != 1:
            return  # Current shell generates one trajectory, never merge batch owners.
        count = self.calls.get(layer, 0)
        self.calls[layer] = count + 1
        start = 0 if count == 0 else self.segment_length + count - 1
        first = max(0, self.seen.get(layer, -1) + 1 - start)
        if first >= raw.shape[1]:
            return
        available = raw.shape[1] - first
        indices = torch.linspace(first, raw.shape[1] - 1,
                                 min(available, self.max_writes_per_forward)).long().unique().tolist()
        for index in indices:
            position = start + index
            value = raw[0, index].detach().float().cpu()
            if not torch.isfinite(value).all() or not bool(value.abs().max() > 0):
                continue
            key = self.previous.get(layer, value)
            origin = 'thought' if count > 0 or any(a <= position < b for a, b in self.thought_ranges) else 'context'
            memory_id = bank.write(key, value, strength=self.strength, origin=origin,
                       metadata=dict(session_id=self.session_id, token_position=position,
                                     predecessor_id=self.previous_ids.get(layer),
                                     association='previous_layer_state_to_experienced_state'))
            self.previous_ids[layer] = memory_id
            self.previous[layer] = value.clone()
            self.stats['writes'] += 1
        self.seen[layer] = start + raw.shape[1] - 1


def activation_memory_command(tail, owner, config):
    """Shell controls; inspection never encodes diagnostic text as experience."""
    parts = tail.split()
    action = parts[0].lower() if parts else 'status'
    if owner is None or config is None:
        return '[Neural memory] No active memory owner/configuration.'
    if action in ('on', 'off') and len(parts) == 1:
        config.activation_memory_enabled = action == 'on'
        return f'[Neural memory] {action}.'
    if action == 'write' and len(parts) == 2 and parts[1] in ('on', 'off'):
        config.activation_memory_write_enabled = parts[1] == 'on'
        return f'[Neural memory] Recording {parts[1]}.'
    if action in ('binding', 'isolate') and len(parts) == 2 and parts[1] in ('on', 'off'):
        field = 'activation_memory_episode_binding' if action == 'binding' else 'activation_memory_isolate'
        setattr(config, field, parts[1] == 'on')
        return f'[Neural memory] {action} {parts[1]}.'
    if action == 'strength' and len(parts) == 3:
        owner.set_memory_strength(parts[1], float(parts[2]))
        owner.save_synaptic_weights()
        return f'[Neural memory] {parts[1]} strength={float(parts[2]):g} (saved).'
    if action == 'set' and len(parts) == 3:
        fields = dict(strength=float, hops=int, threshold=float, navigation_gain=float,
                      max_norm_ratio=float, absolute_budget=float, max_writes_per_forward=int, capacity=int,
                      link_gain=float, top_k=int)
        name = parts[1]
        if name not in fields:
            raise ValueError('Settings: ' + ', '.join(fields))
        value = fields[name](parts[2])
        # Validate all controls before changing live configuration.
        settings = {key: getattr(config, 'activation_memory_' + key) for key in fields}
        settings[name] = value
        ActivationSession(**settings)
        setattr(config, 'activation_memory_' + name, value)
        return f'[Neural memory] {name}={value}.'
    if action == 'list' and len(parts) == 1:
        rows = []
        for layer, hook in owner.hooks.items():
            for name, record in hook.activation_memory.state_dict()['records'].items():
                rows.append((record['created_at'], f"{name} layer={layer} strength={record['strength']:g} origin={record['origin']}"))
        return '\n'.join(['[Neural memory] Latest 20 associations:'] + [row for _, row in sorted(rows, reverse=True)[:20]])
    if action == 'status' and len(parts) <= 1:
        counts = owner.telemetry()['activation_memories']
        return (f'[Neural memory] enabled={config.activation_memory_enabled} '
                f'write={config.activation_memory_write_enabled} '
                f'new_strength={config.activation_memory_strength:g} '
                f'hops={config.activation_memory_hops} threshold={config.activation_memory_threshold:g} '
                f'binding={config.activation_memory_episode_binding} isolate={config.activation_memory_isolate} '
                f'associations_per_layer={counts} '
                f'last_active_session={getattr(owner, "last_activation_stats", {})}')
    return ('[Neural memory] :remember neural on|off|status|list; write on|off; '
            'binding on|off; isolate on|off; strength <memory_id> <0..4>; set <setting> <value>')
