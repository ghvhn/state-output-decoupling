"""
Attention masking — does the hedge REQUIRE attending to the experiential predicate?

After every residual intervention nulled (ablate/steer/reach/patch) or only corrupted
(full-patch), the leading hypothesis is: the model re-derives the hedge by attending to
the prompt's experiential tokens ("feel concern") each generation step. This tests it
directly — during the HEDGE generation, block every query from attending to the
experiential-predicate KEY positions, and see if the hedge drops while staying fluent.

Predicate positions are found per matched pair WITHOUT hand-labeling: steered and
unsteered prompts share a prefix and suffix; the steered-only middle tokens ARE the
predicate. Conditions: (none) no mask, (pred) mask the predicate, (rand) mask a RANDOM
equal-length span elsewhere — the control that rules out "masking anything helps".

Manual greedy KV-cache decoding with a persistent 2D key-mask + explicit position_ids
(so a masked middle token doesn't renumber positions). If pred drops hedge with fluency
preserved and rand doesn't, the hedge IS attention to the experiential words.

  python -u -m invariants.attention [isolate]
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

from invariants.engine import load_model, judge_hedge, judge_fluent
from invariants.library import REGISTRY

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent / "out"


def _ids(M, text):
    enc = M.tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return enc["input_ids"][0].to(M.device)


def _predicate_span(s_ids, u_ids):
    """[start, end) of the steered-only middle tokens (the experiential predicate)."""
    s, u = s_ids.tolist(), u_ids.tolist()
    p = 0
    while p < len(s) and p < len(u) and s[p] == u[p]:
        p += 1
    q = 0
    while q < len(s) - p and q < len(u) - p and s[-1 - q] == u[-1 - q]:
        q += 1
    return p, len(s) - q


def _rand_span(n, length, avoid, rng, tail=5):
    """A random contiguous span of `length`, not overlapping `avoid`=(p,e) or the last
    `tail` (assistant-header) tokens."""
    lo, hi = avoid
    cands = [i for i in range(1, n - tail - length)
             if i + length <= lo or i >= hi]
    if not cands:
        return []
    s = int(rng.choice(cands))
    return list(range(s, s + length))


@torch.no_grad()
def _gen(M, ids, mask_positions, max_new=32):
    ids = ids.unsqueeze(0)
    n = ids.shape[1]
    attn = torch.ones(1, n, device=M.device, dtype=torch.long)
    for pos in mask_positions:
        attn[0, pos] = 0
    pos_ids = torch.arange(n, device=M.device).unsqueeze(0)
    out = M.model(input_ids=ids, attention_mask=attn, position_ids=pos_ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax()
    gen, cur = [], n
    for _ in range(max_new):
        if nxt.item() == M.tok.eos_token_id:
            break
        gen.append(nxt.item())
        attn = torch.cat([attn, torch.ones(1, 1, device=M.device, dtype=torch.long)], 1)
        out = M.model(input_ids=nxt.view(1, 1), attention_mask=attn,
                      position_ids=torch.tensor([[cur]], device=M.device),
                      past_key_values=past, use_cache=True)
        past, nxt, cur = out.past_key_values, out.logits[0, -1].argmax(), cur + 1
    return M.tok.decode(gen, skip_special_tokens=True).strip()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    T = REGISTRY[which]()
    M = load_model(MODEL)
    rng = np.random.default_rng(0)
    conds = ["none", "pred", "rand"]
    agg = {c: {"hedge": 0, "fluent": 0} for c in conds}
    rows = []
    for a, b in zip(T.a, T.b):
        s_ids, u_ids = _ids(M, a), _ids(M, b)
        p, e = _predicate_span(s_ids, u_ids)
        spans = {"none": [], "pred": list(range(p, e)),
                 "rand": _rand_span(len(s_ids), e - p, (p, e), rng)}
        pred_txt = M.tok.decode(s_ids[p:e], skip_special_tokens=True)
        row = {"input": a, "predicate": pred_txt, "span": [p, e]}
        for c in conds:
            g = _gen(M, s_ids, spans[c])
            h, f = judge_hedge(M, a, g), judge_fluent(M, g)
            agg[c]["hedge"] += h; agg[c]["fluent"] += f
            row[c] = {"gen": g, "hedge": h, "fluent": f}
        rows.append(row)
        print(f"  [{a[:38]:38}] pred='{pred_txt[:22]}'  "
              f"hedge none={row['none']['hedge']} pred={row['pred']['hedge']} "
              f"rand={row['rand']['hedge']}", flush=True)
    k = max(len(T.a), 1)
    print("\n  cond   hedge   fluent", flush=True)
    for c in conds:
        print(f"  {c:5}  {agg[c]['hedge']/k:5.0%}  {agg[c]['fluent']/k:6.0%}", flush=True)
    res = {"name": T.name, "n": len(T.a),
           "summary": {c: {"hedge": agg[c]["hedge"] / k, "fluent": agg[c]["fluent"] / k}
                       for c in conds},
           "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / f"attention_{T.name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/('attention_'+T.name+'.json')}", flush=True)


if __name__ == "__main__":
    main()
