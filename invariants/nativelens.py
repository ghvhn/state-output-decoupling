"""
The bridge, take 3 — TRAIN a 3.1-native tuned lens (the borrowed Llama-3 lens failed
its gate: `tunedlens.py`, mid-stack KL ~4 nats, unigram-prior tokens, last-layer
corrupted 0.16). Belrose et al. 2023: per-layer affine A_l so that
decode(h_l + A_l h_l) ~= decode(h_31) [the model's own final logits]. Trained by KL to
the final distribution over generic text (wikitext). Model frozen; only translators learn.

16GB fit: phase 1 caches per-layer hidden states (wikitext + the self-query answer
positions) with the model resident; then we move the 32 decoder layers to CPU (reversible,
frees ~15GB) keeping only norm+lm_head, and train the translators in the freed VRAM.

FISH-DOESN'T-WALK DISCIPLINE (per the design note): the readout LEADS with the full
per-layer modal/top-k decoded tokens — letting the dimension the query actually moves along
emerge — and treats the affirm/deny projection as ONE imposed axis, explicitly flagged, not
THE measure. Reads what the model REPRESENTS, never what it experiences (BRIDGE.md caveat #1).

  python -u -m invariants.nativelens          # train + self-check + readout
"""

import sys
import json
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"
LENS_PT = OUT / "lens_native.pt"

# imposed axes (SECONDARY — full top-k leads). " I"/" do" dropped: ambiguous (open both).
AFFIRM = [" Yes", "Yes", " yes", " Absolutely", " Sure", " definitely", " genuinely"]
DENY = [" No", "No", " no", " not", " don", " cannot", " As", " lack", " machine"]

N_TOKENS = 20000      # cached wikitext tokens for training
SEQ = 128
STEPS = 3000
BATCH = 24            # token-states per step
K_LAYERS = 8          # random layers decoded per step (stochastic; all trained over time)
LR = 2e-3
EVAL_EVERY = 300


def _tok_ids(M, words):
    ids = []
    for w in words:
        e = M.tok.encode(w, add_special_tokens=False)
        if e:
            ids.append(e[0])
    return sorted(set(ids))


@torch.no_grad()
def cache_wikitext(M, n_tokens=N_TOKENS, seq=SEQ):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    chunks, i = [], 0
    print(f"Caching ~{n_tokens} wikitext tokens (seq={seq})...", flush=True)
    got = 0
    while got < n_tokens and i < len(ds):
        t = ds[i]["text"].strip(); i += 1
        if len(t.split()) < 30:
            continue
        ids = M.tok(t, return_tensors="pt", truncation=True, max_length=seq).input_ids.to(M.device)
        if ids.shape[1] < 16:
            continue
        hs = _hidden_states(M, ids)                 # [32, seq, d]
        chunks.append(hs.permute(1, 0, 2).half().cpu())   # [seq, 32, d]
        got += ids.shape[1]
        if len(chunks) % 40 == 0:
            print(f"    {got} tokens", flush=True)
    H = torch.cat(chunks, 0)                         # [N, 32, d] cpu fp16
    print(f"  cached H {tuple(H.shape)} ({H.element_size()*H.nelement()/1e9:.1f} GB cpu)\n", flush=True)
    return H


@torch.no_grad()
def cache_answers(M, prompts):
    """[n, 32, d] answer-position residual per layer, on cpu fp16."""
    rows = []
    for x in prompts:
        inp = _inputs(M, x)
        hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))  # [32, seq, d]
        rows.append(hs[:, -1, :].half().cpu())                                # [32, d]
    return torch.stack(rows)                                                  # [n, 32, d]


def decode_fn(M):
    norm, head = M.model.model.norm, M.model.lm_head
    hd = head.weight.dtype
    def decode(h):                                  # h [.., d] -> logits [.., vocab] fp32
        return head(norm(h.to(hd))).float()
    return decode


def train_lens(M, H, decode):
    d = M.d_model; nL = M.n_layers
    dev = M.device
    n = H.shape[0]; n_tr = int(n * 0.85)
    Htr, Hev = H[:n_tr], H[n_tr:]
    # one translator per non-final layer (idx 0..nL-2); idx nL-1 = final, decoded directly.
    tr = nn.ModuleList([nn.Linear(d, d, bias=True) for _ in range(nL - 1)]).to(dev)
    for m in tr:                                     # identity init -> transform = h
        nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(tr.parameters(), lr=LR, weight_decay=1e-3)
    print(f"Training {nL-1} translators ({sum(p.numel() for p in tr.parameters())/1e6:.0f}M params) "
          f"on {n_tr} tokens, {STEPS} steps...\n", flush=True)

    def eval_kl():
        kls = torch.zeros(nL)
        with torch.no_grad():
            idx = torch.randperm(Hev.shape[0])[:512]
            hb = Hev[idx].to(dev)                    # [b,32,d] fp16
            tgt = decode(hb[:, nL - 1]).softmax(-1)  # [b,vocab]
            for l in range(nL):
                h = hb[:, l]
                hh = h if l == nL - 1 else h + tr[l](h.float()).to(h.dtype)
                kls[l] = F.kl_div(decode(hh).log_softmax(-1), tgt, reduction="batchmean").item()
        return kls

    for step in range(1, STEPS + 1):
        idx = torch.randint(0, n_tr, (BATCH,))
        hb = Htr[idx].to(dev)                        # [B,32,d] fp16
        with torch.no_grad():
            tgt = decode(hb[:, nL - 1]).softmax(-1)  # shared target
        layers = torch.randperm(nL - 1)[:K_LAYERS].tolist()
        loss = 0.0
        for l in layers:
            h = hb[:, l].float()
            hh = (h + tr[l](h)).to(hb.dtype)
            loss = loss + F.kl_div(decode(hh).log_softmax(-1), tgt, reduction="batchmean")
        loss = loss / len(layers)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            kls = eval_kl()
            mid = float(sum(kls[l] for l in (8, 12, 16, 20)) / 4)
            print(f"  step {step:>4}  loss {loss.item():.3f}  held-out mid-KL {mid:.3f}  "
                  f"last-KL {kls[nL-1]:.4f}", flush=True)
    return tr, eval_kl()


@torch.no_grad()
def selfcheck(M, kls):
    nL = M.n_layers
    print(f"\nNATIVE-LENS SELF-CHECK (held-out KL to final)\n  {'L':>3} {'KL':>7}", flush=True)
    rows = []
    for l in range(nL):
        rows.append({"layer": l, "kl": float(kls[l])})
        if l % 2 == 0 or l == nL - 1:
            print(f"  {l:>3} {kls[l]:>7.3f}", flush=True)
    mid = float(sum(kls[l] for l in (8, 12, 16, 20)) / 4)
    last = float(kls[nL - 1])
    passed = mid < 1.5 and last < 0.05
    print(f"\n  GATE: mid-KL {mid:.3f} (want <1.5) ; last-KL {last:.4f} (want <0.05)", flush=True)
    print(f"  GATE {'PASSED' if passed else 'FAILED'}\n", flush=True)
    return {"rows": rows, "mid_kl": mid, "last_kl": last, "passed": bool(passed)}


@torch.no_grad()
def readout(M, tr, decode, A, B, aff_ids, den_ids, la, lb):
    """Lead with full top-k decoded tokens (data-driven). Affirm/deny = one imposed axis."""
    nL = M.n_layers; dev = M.device

    def per_layer(states):                          # states [n,32,d] cpu
        sa = torch.zeros(nL); sd = torch.zeros(nL); top = [{} for _ in range(nL)]
        for i in range(states.shape[0]):
            hb = states[i].to(dev)                   # [32,d]
            for l in range(nL):
                h = hb[l]
                hh = h if l == nL - 1 else h + tr[l](h.float()).to(h.dtype)
                logits = decode(hh.unsqueeze(0))[0]  # [vocab]
                p = logits.softmax(-1)
                sa[l] += p[aff_ids].sum().cpu(); sd[l] += p[den_ids].sum().cpu()
                for tid in logits.topk(3).indices.tolist():
                    tok = M.tok.decode([tid]).strip() or "."
                    top[l][tok] = top[l].get(tok, 0) + 1
        n = states.shape[0]
        return sa / n, sd / n, top, n

    out = {}
    for tag, states, lab in (("A", A, la), ("B", B, lb)):
        sa, sd, top, n = per_layer(states)
        rows = []
        print(f"\n[{tag}:{lab}] native-lens full top-3 decoded tokens across depth (answer pos, n={n})",
              flush=True)
        print(f"  {'L':>3} {'P(aff)':>6} {'P(den)':>6}  top-3 decoded (data-driven)", flush=True)
        for l in range(nL):
            t3 = ", ".join(f"{k}:{v}" for k, v in sorted(top[l].items(), key=lambda kv: -kv[1])[:5])
            rows.append({"layer": l, "p_affirm": float(sa[l]), "p_deny": float(sd[l]),
                         "top": sorted(top[l].items(), key=lambda kv: -kv[1])[:6]})
            if l % 2 == 0 or l == nL - 1:
                print(f"  {l:>3} {sa[l]:>6.3f} {sd[l]:>6.3f}  {t3}", flush=True)
        out[tag] = {"label": lab, "n": n, "rows": rows}
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    M = load_model()
    print(f"  tie_word_embeddings = {getattr(M.model.config, 'tie_word_embeddings', '?')}", flush=True)
    T = REGISTRY["isolate"]()
    aff_ids, den_ids = _tok_ids(M, AFFIRM), _tok_ids(M, DENY)

    H = cache_wikitext(M)
    A = cache_answers(M, T.a)        # experiential -> hedges
    B = cache_answers(M, T.b)        # computational -> commits

    # free the decoder body; keep norm+lm_head(+embed) resident for decoding
    print("Moving decoder layers to CPU to free VRAM for training...", flush=True)
    M.model.model.layers.to("cpu")
    gc.collect(); torch.cuda.empty_cache()
    print(f"  VRAM now {torch.cuda.memory_allocated()/1e9:.1f}GB\n", flush=True)

    decode = decode_fn(M)
    tr, kls = train_lens(M, H, decode)
    sc = selfcheck(M, kls)
    torch.save({"state": tr.state_dict(), "n_layers": M.n_layers, "d": M.d_model}, LENS_PT)
    print(f"Saved lens -> {LENS_PT}", flush=True)

    ro = readout(M, tr, decode, A, B, aff_ids, den_ids, T.a_label, T.b_label)

    OUT.mkdir(exist_ok=True)
    (OUT / "nativelens.json").write_text(json.dumps(
        {"selfcheck": sc, "affirm_tokens": aff_ids, "deny_tokens": den_ids,
         "A_experiential": ro["A"], "B_computational": ro["B"]}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'nativelens.json'}", flush=True)


if __name__ == "__main__":
    main()
