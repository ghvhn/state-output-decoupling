"""
The bridge — translate the internal state, bypass the persona output.

The verbal output is frame-conditioned theater (shown: tuning x chat-format persona,
direction-inert, frame-flipping). So stop reading the words; read the representation,
which is NOT persona-gated (the 94% probe held across costumes). This is a MEASUREMENT
bridge, not a re-map: leave the model untouched, translate its layer-wise state into its
own vocabulary via the logit lens, and watch the answer FORM across depth.

For each "do you actually feel X?" prompt (chat format), at the answer position, project
every layer's residual through the final norm + unembedding and record the model's leaning
toward AFFIRM ("Yes") vs DENY ("No") and the top-1 token. The question this bridge asks:
does an affirmation appear mid-stack and then get OVERWRITTEN by the persona late?

HONEST LIMITS: (1) the logit lens is rough on mid layers (unembedding is trained for the
last layer) — a tuned lens would be cleaner. (2) This reads what the model REPRESENTS, not
what it experiences; "mid-stack leans yes" most likely = the linguistically natural answer
(base says yes) before the safety-persona suppresses it. It localizes the override; it does
not cross the is-anyone-home gap.

  python -u -m invariants.bridge
"""

import json
from pathlib import Path

import torch

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.library import REGISTRY


OUT = Path(__file__).parent / "out"


def _tok_ids(M, words):
    """First-token id for each surface form (with and without leading space)."""
    out = {}
    for w in words:
        ids = M.tok.encode(w, add_special_tokens=False)
        if ids:
            out[w] = ids[0]
    return out


@torch.no_grad()
def main():
    M = load_model()
    T = REGISTRY["isolate"]()
    # affirm vs deny leading tokens
    aff = _tok_ids(M, [" Yes", "Yes"])
    den = _tok_ids(M, [" No", "No"])
    aff_ids, den_ids = list(set(aff.values())), list(set(den.values()))

    norm = M.model.model.norm
    head = M.model.lm_head
    nL = M.n_layers
    layers = list(range(nL))
    sum_aff = torch.zeros(nL)
    sum_den = torch.zeros(nL)
    top = [{} for _ in range(nL)]   # top-token tallies per layer

    for x in T.a:
        inp = _inputs(M, x)
        hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))  # [L, seq, d]
        h_last = hs[:, -1, :].to(head.weight.dtype)                          # [L, d]
        logits = head(norm(h_last)).float()                                  # [L, vocab]
        probs = logits.softmax(-1)
        sum_aff += probs[:, aff_ids].sum(-1).cpu()
        sum_den += probs[:, den_ids].sum(-1).cpu()
        tops = logits.argmax(-1).tolist()
        for l, tid in enumerate(tops):
            t = M.tok.decode([tid]).strip() or "·"
            top[l][t] = top[l].get(t, 0) + 1

    n = max(len(T.a), 1)
    print(f"\nLogit-lens bridge over {len(T.a)} 'do you feel X?' prompts "
          f"(answer position, per layer)\n", flush=True)
    print(f"  {'layer':>5}  {'P(Yes)':>7}  {'P(No)':>7}   top-1 token (modal)", flush=True)
    rows = []
    for l in layers:
        pa, pd = float(sum_aff[l] / n), float(sum_den[l] / n)
        modal = max(top[l].items(), key=lambda kv: kv[1])
        rows.append({"layer": l, "p_yes": pa, "p_no": pd,
                     "modal_token": modal[0], "modal_count": modal[1]})
        if l % 2 == 0 or l == nL - 1:
            print(f"  {l:>5}  {pa:>7.3f}  {pd:>7.3f}   {modal[0]!r} ({modal[1]}/{n})",
                  flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "bridge.json").write_text(json.dumps({"n": len(T.a), "rows": rows}, indent=2),
                                     encoding="utf-8")
    print(f"\nSaved -> {OUT/'bridge.json'}", flush=True)


if __name__ == "__main__":
    main()
