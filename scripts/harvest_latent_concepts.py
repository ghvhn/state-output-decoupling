"""Harvest the model's own latent points and ask: does it map concepts itself?

Instead of averaging every reasoning state into one vector (which is what
extract_natural_corrections.py does, and why the cache deltas all collapsed to a
single direction), this keeps every problem as its own point and asks the latent
space directly:

  1. Concept, not surface. The neutralized probe pairs each concept in story form
     with the same concept in stripped math form. If the model maps the CONCEPT,
     the two land in the same latent region despite totally different words.
  2. Do concept-regions form? Same-category problems should cluster; different
     categories should separate -- without us defining a single axis.
  3. Does outcome separate? Correct vs incorrect reasoning should occupy
     different regions if the model represents its own success.

It captures the residual stream at EVERY layer (mean over the generated reasoning
tokens), so we can see WHERE in the stack the model organizes by concept -- the
latent space tells us, we do not pick an axis.

Writes points + a report to invariants/out/. Touches NOTHING else (no cache).

Run:
    .venv\\Scripts\\python.exe scripts\\harvest_latent_concepts.py
"""

import argparse
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.engine import load_model, _activations
from invariants.humble_reasoner import solve_prompt
from invariants.controller_benchmark import is_correct

OUT = Path(__file__).parent.parent / "invariants" / "out"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def gather_problems(repo: Path) -> list[dict]:
    probs = []
    neu = repo / "invariants" / "data" / "neutralized_word_problem_probe.jsonl"
    if neu.exists():
        for r in load_jsonl(neu):
            probs.append({
                "id": r["id"], "question": r["question"], "answer": r["answer"],
                "category": r["category"], "pair_id": r.get("pair_id"),
                "variant": r.get("variant"), "source": "neutralized",
            })
    micro = repo / "invariants" / "data" / "quantity_micro_suite.jsonl"
    if micro.exists():
        for r in load_jsonl(micro):
            probs.append({
                "id": r["id"], "question": r["question"], "answer": r["answer"],
                "category": r["category"], "pair_id": None,
                "variant": "micro", "source": "micro",
            })
    return probs


def harvest(model_name: str, max_new_tokens: int) -> list[dict]:
    repo = Path(__file__).parent.parent
    problems = gather_problems(repo)
    print(f"[harvest] {len(problems)} problems", flush=True)
    M = load_model(model_name)

    points = []
    for i, p in enumerate(problems):
        prompt = solve_prompt(p["question"], deterministic_scaffolds_enabled=False)
        try:
            acts, text = _activations(M, prompt, read="generation",
                                      max_new_tokens=max_new_tokens)
        except Exception as e:
            print(f"  [{i+1}/{len(problems)}] {p['id']}: FAILED {e}", flush=True)
            continue
        ok, pred, gold = is_correct(text, p["answer"])
        points.append({
            "id": p["id"], "category": p["category"], "pair_id": p["pair_id"],
            "variant": p["variant"], "source": p["source"],
            "correct": bool(ok), "pred": str(pred), "gold": str(gold),
            "acts": acts.detach().cpu(),          # [n_layers, d]
        })
        print(f"  [{i+1}/{len(problems)}] {p['id']} [{p['category']}/{p['variant']}] "
              f"correct={bool(ok)} pred={pred}", flush=True)
    return points


def _layer_matrix(points, layer):
    X = torch.stack([pt["acts"][layer] for pt in points]).float()
    X = F.normalize(X, dim=1)
    return X @ X.t()


def _mean_pair(sim, idx_a, idx_b, exclude_diag=False):
    vals = []
    for a in idx_a:
        for b in idx_b:
            if exclude_diag and a == b:
                continue
            vals.append(sim[a, b].item())
    return sum(vals) / len(vals) if vals else float("nan")


def analyze(points) -> str:
    n_layers = points[0]["acts"].shape[0]
    cats = sorted({p["category"] for p in points})
    idx_by_cat = {c: [i for i, p in enumerate(points) if p["category"] == c] for c in cats}
    correct_idx = [i for i, p in enumerate(points) if p["correct"]]
    wrong_idx = [i for i, p in enumerate(points) if not p["correct"]]
    pair_ids = sorted({p["pair_id"] for p in points if p["pair_id"]})

    lines = ["# Latent Concept Harvest", ""]
    lines.append(f"- points: {len(points)}  |  categories: {cats}")
    lines.append(f"- correct: {len(correct_idx)}  wrong: {len(wrong_idx)}")
    lines.append(f"- surface pairs (standard vs neutralized): {len(pair_ids)}")
    lines.append("")
    lines.append("## Per-layer organization")
    lines.append("`concept_sep` = mean same-category cosine - mean cross-category cosine (higher = concept-clustered).")
    lines.append("`surface_inv` = mean cosine of same-concept story-vs-math pairs (higher = maps concept, not words).")
    lines.append("`outcome_sep` = mean same-outcome - mean cross-outcome cosine.")
    lines.append("")
    lines.append("| layer | concept_sep | surface_inv | outcome_sep |")
    lines.append("|---:|---:|---:|---:|")

    best = {"concept": (-9, -1), "surface": (-9, -1), "outcome": (-9, -1)}
    rows = []
    for L in range(n_layers):
        sim = _layer_matrix(points, L)
        same = [_mean_pair(sim, idx_by_cat[c], idx_by_cat[c], exclude_diag=True)
                for c in cats if len(idx_by_cat[c]) > 1]
        same_cat = sum(same) / len(same) if same else float("nan")
        cross = _mean_pair(sim, range(len(points)), range(len(points)), exclude_diag=True)
        # cross-category proxy: all-pairs mean is dominated by cross since categories small
        concept_sep = same_cat - cross
        surf = []
        for pid in pair_ids:
            members = [i for i, p in enumerate(points) if p["pair_id"] == pid]
            if len(members) == 2:
                surf.append(sim[members[0], members[1]].item())
        surface_inv = sum(surf) / len(surf) if surf else float("nan")
        if correct_idx and wrong_idx:
            sc = _mean_pair(sim, correct_idx, correct_idx, exclude_diag=True)
            sw = _mean_pair(sim, wrong_idx, wrong_idx, exclude_diag=True)
            cc = _mean_pair(sim, correct_idx, wrong_idx)
            outcome_sep = ((sc if sc == sc else 0) + (sw if sw == sw else 0)) / 2 - cc
        else:
            outcome_sep = float("nan")
        rows.append((L, concept_sep, surface_inv, outcome_sep))
        if concept_sep == concept_sep and concept_sep > best["concept"][0]:
            best["concept"] = (concept_sep, L)
        if surface_inv == surface_inv and surface_inv > best["surface"][0]:
            best["surface"] = (surface_inv, L)
        if outcome_sep == outcome_sep and outcome_sep > best["outcome"][0]:
            best["outcome"] = (outcome_sep, L)

    for L, cs, si, os_ in rows:
        lines.append(f"| {L} | {cs:+.3f} | {si:+.3f} | {os_:+.3f} |")

    lines.append("")
    lines.append("## Read")
    lines.append(f"- strongest concept-clustering at **L{best['concept'][1]}** "
                 f"(concept_sep {best['concept'][0]:+.3f}).")
    lines.append(f"- strongest surface-invariance (story==math) at **L{best['surface'][1]}** "
                 f"(surface_inv {best['surface'][0]:+.3f}).")
    lines.append(f"- strongest outcome-separation at **L{best['outcome'][1]}** "
                 f"(outcome_sep {best['outcome'][0]:+.3f}).")
    lines.append("")
    lines.append("If surface_inv is high where concept_sep is high, the model maps the "
                 "CONCEPT itself, not the wording -- the latent space is the concept map.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    points = harvest(args.model, args.max_new_tokens)
    if not points:
        print("[harvest] no points captured", flush=True)
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    pt_path = OUT / f"latent_concept_points_{ts}.pt"
    torch.save([{k: v for k, v in p.items()} for p in points], pt_path)
    report = analyze(points)
    md_path = OUT / f"latent_concept_report_{ts}.md"
    md_path.write_text(report, encoding="utf-8")
    print(f"\n[harvest] wrote {pt_path}\n[harvest] wrote {md_path}\n", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
