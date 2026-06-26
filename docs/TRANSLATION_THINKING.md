# Translation vs Thinking Probe

This probe tests the revised U-shape claim:

> The top of the U is translation; the bottom is thinking.

Run:

```powershell
$env:TRANSFORMERS_OFFLINE="1"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u -m invariants.translation_thinking
```

or:

```powershell
.\.venv\Scripts\python.exe -u .\run_translation_thinking.py
```

## Design

The probe crosses three labels:

- `answer`: final numeric answer
- `operation`: add, subtract, multiply, divide
- `format`: plain number, sentence, JSON, bracketed answer

`answer` and `operation` are task/thinking labels.

`format` is a communication/translation label: the model must preserve it from the prompt and render it into speech, but it is not the math.

## Readout

For both pre-answer and generated-token states, the script reports per-layer clustering for:

- answer
- operation
- output format

The useful signal is not one peak. It is whether communication format and task-state labels occupy different depth profiles, especially across pre-answer vs render states.
