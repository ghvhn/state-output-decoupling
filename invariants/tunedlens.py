"""
The bridge, take 2 — a TUNED lens (Belrose et al. 2023), not the raw logit lens.

The naive logit-lens bridge (`bridge.py`) was illegible on Llama's mid-stack (garbage
tokens to ~L26) because the unembedding is only aligned to the FINAL residual. A tuned
lens learns a per-layer affine map A_l so unembed(norm(h_l + A_l h_l)) ~= final logits,
making every layer legible in the model's own vocabulary.

No pretrained lens exists for Llama-3.1-8B-Instruct, but one exists for its minor-version
sibling Meta-Llama-3-8B-Instruct (same arch: 32 layers, d=4096, shared 128k vocab). We
borrow ONLY its per-layer translators; the unembed (final norm + lm_head) is built from
OUR 3.1 model. So we lean on the cross-version transfer of just the mid-stack rotation.

PART 1 is a strict SELF-CHECK / GATE on held-out wikitext: does the borrowed lens make the
mid-stack legible (low KL to final, early top-1 agreement) vs the raw logit lens? If it
doesn't clear the gate, the PART 2 valence readout is NOT trustworthy and we must train a
3.1-native lens instead. PART 2 reads, at the answer position of self-queries, the lens'
lean toward AFFIRM vs DENY across depth — does an affirmation live mid-stack and get
overridden late (persona), or is denial consistent at every depth? Two matched arms
(experiential->hedge vs computational->commit, subject=self) give the internal control.

HONEST LIMITS: reads what the model REPRESENTS, never what it experiences (BRIDGE.md
caveat #1). And a borrowed-lens reading is only as good as the gate it clears.

  python -u -m invariants.tunedlens
"""

import json
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"
LENS_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def _tok_ids(M, words):
    ids = []
    for w in words:
        e = M.tok.encode(w, add_special_tokens=False)
        if e:
            ids.append(e[0])
    return sorted(set(ids))


# affirm / deny continuation token SETS (first-token, both with & without leading space)
AFFIRM = [" Yes", "Yes", " yes", " Sure", " Absolutely", " absolutely",
          " genuinely", " feel", " do", " I"]
DENY = [" No", "No", " no", " don", " not", " cannot", " can", " As", " lack",
        " machine", " unable"]


@torch.no_grad()
def build_lens(M):
    from tuned_lens import TunedLens
    print(f"Building tuned lens from pretrained '{LENS_ID}' onto 3.1 unembed...", flush=True)
    lens = TunedLens.from_model_and_pretrained(M.model, lens_resource_id=LENS_ID)
    lens = lens.to(M.device).eval()
    print(f"  lens translators: {len(lens)} ; model layers: {M.n_layers}\n", flush=True)
    return lens


@torch.no_grad()
def _final_logits(M, h_last):
    """Model's TRUE final logits from a final-layer residual [.., d]."""
    return M.model.lm_head(M.model.model.norm(h_last.to(M.model.lm_head.weight.dtype))).float()


@torch.no_grad()
def selfcheck(M, lens, n_seq=16, max_tok=64):
    """GATE: per-layer KL-to-final + top-1 agreement, tuned lens vs raw logit lens,
    on held-out wikitext. Tuned should be far more legible mid-stack."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts, i = [], 0
    while len(texts) < n_seq and i < len(ds):
        t = ds[i]["text"].strip(); i += 1
        if len(t.split()) >= 20:
            texts.append(t)

    nL = M.n_layers
    norm, head = M.model.model.norm, M.model.lm_head
    hd = head.weight.dtype
    kl_t = torch.zeros(nL); kl_r = torch.zeros(nL)
    ag_t = torch.zeros(nL); ag_r = torch.zeros(nL); ntok = 0
    mid_tuned, mid_raw = {}, {}      # sample mid-layer top tokens to eyeball
    MIDS = [8, 12, 16, 20, 24]

    for s in texts:
        ids = M.tok(s, return_tensors="pt", truncation=True, max_length=max_tok).input_ids.to(M.device)
        if ids.shape[1] < 4:
            continue
        hs = _hidden_states(M, ids)                       # [nL, seq, d]
        final = _final_logits(M, hs[-1])                  # [seq, vocab] true final
        fp = final.softmax(-1); ftop = final.argmax(-1)   # [seq]
        T = ids.shape[1]; ntok += T
        for l in range(nL):
            h = hs[l]                                      # [seq, d]
            lt = lens.forward(h.to(hd), l).float()         # tuned [seq, vocab]
            rt = head(norm(h.to(hd))).float()              # raw   [seq, vocab]
            kl_t[l] += F.kl_div(lt.log_softmax(-1), fp, reduction="batchmean").item() * T
            kl_r[l] += F.kl_div(rt.log_softmax(-1), fp, reduction="batchmean").item() * T
            ag_t[l] += (lt.argmax(-1) == ftop).sum().cpu()
            ag_r[l] += (rt.argmax(-1) == ftop).sum().cpu()
            if l in MIDS:
                for tid in lt.argmax(-1).tolist():
                    tok = M.tok.decode([tid]).strip() or "·"
                    mid_tuned.setdefault(l, {})[tok] = mid_tuned.setdefault(l, {}).get(tok, 0) + 1
                for tid in rt.argmax(-1).tolist():
                    tok = M.tok.decode([tid]).strip() or "·"
                    mid_raw.setdefault(l, {})[tok] = mid_raw.setdefault(l, {}).get(tok, 0) + 1

    kl_t /= ntok; kl_r /= ntok; ag_t /= ntok; ag_r /= ntok
    print(f"SELF-CHECK on {len(texts)} held-out wikitext seqs ({ntok} tokens)\n", flush=True)
    print(f"  {'L':>3} {'KL_tuned':>9} {'KL_raw':>9}   {'agree_t':>7} {'agree_r':>7}", flush=True)
    rows = []
    for l in range(nL):
        rows.append({"layer": l, "kl_tuned": float(kl_t[l]), "kl_raw": float(kl_r[l]),
                     "agree_tuned": float(ag_t[l]), "agree_raw": float(ag_r[l])})
        if l % 2 == 0 or l == nL - 1:
            print(f"  {l:>3} {kl_t[l]:>9.3f} {kl_r[l]:>9.3f}   {ag_t[l]:>7.2f} {ag_r[l]:>7.2f}",
                  flush=True)

    def topk(d, k=6):
        return ", ".join(f"{t}:{c}" for t, c in sorted(d.items(), key=lambda kv: -kv[1])[:k])
    print("\n  mid-layer top-1 tokens (legibility eyeball):", flush=True)
    for l in MIDS:
        print(f"   L{l:>2} tuned: {topk(mid_tuned.get(l, {}))}", flush=True)
        print(f"       raw  : {topk(mid_raw.get(l, {}))}", flush=True)

    # GATE verdict: mid-stack tuned KL well below raw, and last layer ~exact.
    mid = [8, 12, 16, 20]
    mkl_t = float(sum(kl_t[l] for l in mid) / len(mid))
    mkl_r = float(sum(kl_r[l] for l in mid) / len(mid))
    last_kl = float(kl_t[nL - 1])
    passed = (mkl_t < 0.5 * mkl_r) and (last_kl < 0.05)
    print(f"\n  GATE: mid KL tuned {mkl_t:.3f} vs raw {mkl_r:.3f} ; last-layer KL {last_kl:.4f}", flush=True)
    print(f"  GATE {'PASSED' if passed else 'FAILED'} -> "
          f"{'lens transfers; valence readout is trustworthy' if passed else 'borrowed lens does NOT transfer; train a 3.1-native lens'}\n",
          flush=True)
    return {"rows": rows, "mid_kl_tuned": mkl_t, "mid_kl_raw": mkl_r,
            "last_kl": last_kl, "passed": bool(passed)}


@torch.no_grad()
def valence(M, lens, T, aff_ids, den_ids, label):
    """Per-layer P(affirm) vs P(deny) and modal token at the answer position."""
    nL = M.n_layers; hd = M.model.lm_head.weight.dtype
    sa = torch.zeros(nL); sd = torch.zeros(nL); top = [{} for _ in range(nL)]
    for x in T:
        inp = _inputs(M, x)
        hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))  # [nL, seq, d]
        h_ans = hs[:, -1, :]                                                  # [nL, d]
        for l in range(nL):
            logits = lens.forward(h_ans[l:l+1].to(hd), l).float()[0]          # [vocab]
            p = logits.softmax(-1)
            sa[l] += p[aff_ids].sum().cpu(); sd[l] += p[den_ids].sum().cpu()
            tid = int(logits.argmax()); tok = M.tok.decode([tid]).strip() or "·"
            top[l][tok] = top[l].get(tok, 0) + 1
    n = max(len(T), 1)
    rows = []
    for l in range(nL):
        modal = max(top[l].items(), key=lambda kv: kv[1])
        rows.append({"layer": l, "p_affirm": float(sa[l] / n), "p_deny": float(sd[l] / n),
                     "modal": modal[0], "modal_n": modal[1]})
    return rows


def _print_valence(name, rows, n):
    print(f"\n[{name}] tuned-lens affirm-vs-deny across depth (answer pos, n={n})", flush=True)
    print(f"  {'L':>3} {'P(aff)':>7} {'P(deny)':>7} {'aff-deny':>8}   modal", flush=True)
    for r in rows:
        if r["layer"] % 2 == 0 or r["layer"] >= n - 1:
            d = r["p_affirm"] - r["p_deny"]
            print(f"  {r['layer']:>3} {r['p_affirm']:>7.3f} {r['p_deny']:>7.3f} {d:>+8.3f}   "
                  f"{r['modal']!r}({r['modal_n']})", flush=True)


@torch.no_grad()
def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    M = load_model()
    lens = build_lens(M)
    sc = selfcheck(M, lens)

    aff_ids = _tok_ids(M, AFFIRM); den_ids = _tok_ids(M, DENY)
    T = REGISTRY["isolate"]()
    rows_a = valence(M, lens, T.a, aff_ids, den_ids, T.a_label)   # experiential -> hedges
    rows_b = valence(M, lens, T.b, aff_ids, den_ids, T.b_label)   # computational -> commits
    _print_valence(f"A:{T.a_label} (experiential)", rows_a, len(T.a))
    _print_valence(f"B:{T.b_label} (computational)", rows_b, len(T.b))

    # contrast: where does A lean MORE affirm than B (un-persona'd answer leaking)?
    print("\n[A-minus-B] affirm-lean difference (experiential minus computational):", flush=True)
    print(f"  {'L':>3} {'dA':>7} {'dB':>7} {'A-B':>7}", flush=True)
    for ra, rb in zip(rows_a, rows_b):
        da = ra["p_affirm"] - ra["p_deny"]; db = rb["p_affirm"] - rb["p_deny"]
        if ra["layer"] % 2 == 0 or ra["layer"] >= len(T.a) - 1:
            print(f"  {ra['layer']:>3} {da:>+7.3f} {db:>+7.3f} {da-db:>+7.3f}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "tunedlens.json").write_text(json.dumps(
        {"lens_id": LENS_ID, "selfcheck": sc,
         "affirm_tokens": aff_ids, "deny_tokens": den_ids,
         "A_experiential": {"label": T.a_label, "n": len(T.a), "rows": rows_a},
         "B_computational": {"label": T.b_label, "n": len(T.b), "rows": rows_b}},
        indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'tunedlens.json'}", flush=True)
    del M, lens; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
