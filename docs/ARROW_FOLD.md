# Arrow Fold Probe

This is the next probe for the claim:

> Late layers are not merely later processing stages. They are the inverse/output arm of early layers.

Run:

```powershell
python -u .\run_arrow_fold.py
```

Quick pilot:

```powershell
python -u -m invariants.arrow_fold --max-items 32 --n-shuffle 50
```

## What It Measures

The script uses the same controlled arithmetic grid as `intent_surface_control.py`, where:

- `operation/intent` is add, subtract, multiply, or divide
- `base/surface` is the same names, objects, and numbers

It extracts two states for each prompt:

- `pre`: prompt-final state before an answer token
- `render`: mean state over generated answer tokens

Then it compares `pre` layer `L` to `render` layer `31-L`, asking whether the intake side and output side share homologous label subspaces.

## How To Read It

Evidence for the arrow/fold picture would look like:

- operation signal strong on the intake side and aligned with late render states
- surface/base signal weak through the middle but returning on the output side
- mirrored pre-to-render overlap beating same-depth pre-to-render overlap for at least one label family

Evidence against the picture would look like:

- no mirrored homology above null
- best render partners unrelated to mirrored layers
- all structure explained by same-depth residual similarity or lexical prompt artifacts
