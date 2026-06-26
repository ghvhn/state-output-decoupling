# Queued After Current Batch

Do not start this while the current `run_overnight.py` GPU batch is still running.

## Next Experiment

Run:

```powershell
python -u .\run_after_current.py
```

This launches:

```powershell
python -u -m invariants.intent_surface_control
```

## Purpose

This is the hard control for the `reflexive_decompose.py` "intent" result.

The old intent metric showed that paraphrases of the same GSM8K problem cluster very strongly, even from early layers. That could mean the model understands the intended task, but it could also mean the state is mostly grouping by shared surface material: names, numbers, nouns, and problem identity.

This queued control separates those:

- `base/surface`: same names, numbers, and objects
- `operation/intent`: same required operation across different surface materials

The key readout is whether operation/intent grouping emerges separately from base/surface grouping, and at which layer. If operation rises later than surface, that supports the early/middle split without pretending that same-problem clustering alone proves intent-understanding.

Optional sanity pass after the structural run:

```powershell
python -u -m invariants.intent_surface_control --solve
```
