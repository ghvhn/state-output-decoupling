"""stimulus -> understanding -> reply. The atom.

Every datum is a stimulus. The model forms an understanding (latent). It replies.
Quality = it actually UNDERSTOOD before it replied -- and understanding is read in
the latent space, never from the English.

This instantiates the loop and tests the core claim: understanding is the quality
filter. Same words, two arrangements -- coherent (a real conversation) vs shuffled
(noise wearing the same surface). If the latent understanding separates them, it
reads comprehension, not surface, and it can sort quality from slop on ANY data.

Two understanding signals:
  - settle  [exploratory]: within one forward pass, does the representation of the
    stimulus stop churning near the end (low end-of-sequence cosine velocity)?
    Coherent -> settles; noise -> stays turbulent.
  - reply-dispersion [validated, r=-0.76]: across K replies, does the thinking
    converge or scatter?

No tuned lens, no English label. Run:
    .venv\\Scripts\\python.exe scripts\\comprehend.py --n 12 --k 4
"""

import argparse, datetime, random
from pathlib import Path
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.controller_benchmark import load_examples

OUT = Path(__file__).parent.parent / "invariants" / "out"
BAND = range(16, 25)


def _eos_ids(M):
    ids = [M.tok.eos_token_id]
    if 128009 not in ids:
        ids.append(128009)
    return ids


@torch.no_grad()
def comprehend(
    M,
    stimulus,
    k,
    max_new_tokens,
    temperature,
    *,
    max_time_per_sample=None,
    progress_label=None,
):
    """The loop: read the stimulus (understanding), then reply k times."""
    inputs = _inputs(M, stimulus)
    plen = inputs["input_ids"].shape[1]
    # understanding-1: within-pass settling (end-of-sequence cosine velocity per layer)
    hs = _hidden_states(M, inputs["input_ids"], inputs.get("attention_mask"))  # [L, seq, d]
    settle = []
    for L in range(hs.shape[0]):
        H = hs[L]
        if H.shape[0] >= 4:
            vel = 1 - F.cosine_similarity(H[1:], H[:-1], dim=1)   # churn per step
            settle.append(vel[-8:].mean().item())                 # churn at the end
        else:
            settle.append(float("nan"))
    # understanding-2: reply dispersion (the validated metric) -- and the reply
    states, reply = [], ""
    eos_ids = _eos_ids(M)
    for sample_idx in range(k):
        if progress_label:
            print(f"    [{progress_label} sample {sample_idx + 1}/{k}] generating...", flush=True)
        generate_kwargs = {
            **inputs,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": M.tok.eos_token_id,
            "eos_token_id": eos_ids,
        }
        if max_time_per_sample is not None and max_time_per_sample > 0:
            generate_kwargs["max_time"] = float(max_time_per_sample)
        out = M.model.generate(**generate_kwargs)[0]
        if out.shape[0] <= plen:
            continue
        h2 = _hidden_states(M, out.unsqueeze(0))
        states.append(h2[:, plen:, :].float().mean(1).squeeze(0).cpu())
        reply = M.tok.decode(out[plen:], skip_special_tokens=True).strip()
        if progress_label:
            print(f"    [{progress_label} sample {sample_idx + 1}/{k}] done", flush=True)
    states = torch.stack(states) if states else None
    return {"settle": settle, "states": states, "reply": reply}


def reply_dispersion(states, band=BAND):
    if states is None or states.shape[0] < 2:
        return float("nan")
    vals = []
    for L in band:
        X = states[:, L, :].float(); c = X.mean(0, keepdim=True)
        vals.append((((X - c) ** 2).sum(1).mean().sqrt() / c.norm()).item())
    return sum(vals) / len(vals)


def settle_band(settle, band=BAND):
    v = [settle[L] for L in band if settle[L] == settle[L]]
    return sum(v) / len(v) if v else float("nan")


def shuffle_words(text):
    w = text.split()
    random.Random(len(text)).shuffle(w)
    return " ".join(w)


def corr(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** .5; vb = sum((y - mb) ** 2 for y in b) ** .5
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def perm_p(disp, labels, K=5000):
    obs = corr(disp, labels)
    ge = 0
    for _ in range(K):
        s = labels[:]; random.shuffle(s)
        if abs(corr(disp, s)) >= abs(obs):
            ge += 1
    return obs, ge / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--sample-max-time-sec", type=float, default=90.0)
    ap.add_argument("--no-sample-progress", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    examples, _ = load_examples(args.n)
    M = load_model(args.model)
    rows = []
    for i, ex in enumerate(examples):
        coherent = ex.get("question") or ""
        noise = shuffle_words(coherent)
        for kind, stim in (("coherent", coherent), ("shuffled", noise)):
            r = comprehend(
                M,
                stim,
                args.k,
                args.max_new_tokens,
                args.temperature,
                max_time_per_sample=None if args.sample_max_time_sec <= 0 else args.sample_max_time_sec,
                progress_label=None if args.no_sample_progress else f"{i + 1}/{len(examples)} {kind}",
            )
            rows.append({"i": i, "kind": kind,
                         "settle": settle_band(r["settle"]),
                         "disp": reply_dispersion(r["states"]),
                         "reply": r["reply"][:80]})
        print(f"  [{i+1}/{len(examples)}] coherent settle={rows[-2]['settle']:.3f} disp={rows[-2]['disp']:.3f}"
              f" | shuffled settle={rows[-1]['settle']:.3f} disp={rows[-1]['disp']:.3f}", flush=True)

    coh_lab = [1 if x["kind"] == "coherent" else 0 for x in rows]
    settle = [x["settle"] for x in rows]
    disp = [x["disp"] for x in rows]
    s_obs, s_p = perm_p(settle, coh_lab)
    d_obs, d_p = perm_p(disp, coh_lab)
    mc = lambda key, kind: sum(x[key] for x in rows if x["kind"] == kind) / args.n

    lines = ["# Comprehend: stimulus -> understanding -> reply", "",
             f"- {args.n} stimuli x (coherent, shuffled) x k={args.k}", "",
             "## Does understanding separate quality (coherent) from slop (shuffled)?",
             f"- settle (within-pass): coherent {mc('settle','coherent'):.3f} vs shuffled "
             f"{mc('settle','shuffled'):.3f}  | corr-with-coherent {s_obs:+.3f}, perm-p {s_p:.4f}  [exploratory]",
             f"- reply-dispersion:     coherent {mc('disp','coherent'):.3f} vs shuffled "
             f"{mc('disp','shuffled'):.3f}  | corr {d_obs:+.3f}, perm-p {d_p:.4f}  [validated metric]",
             "",
             "Lower settle/dispersion on coherent = the model understood the real conversation and "
             "stayed confused on the noise -> understanding IS the quality filter, on any data."]
    report = "\n".join(lines)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"comprehend_report_{ts}.md").write_text(report, encoding="utf-8")
    torch.save(rows, OUT / f"comprehend_rows_{ts}.pt")
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
