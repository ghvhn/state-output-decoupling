"""
Communicative repair as participation: does correction change the model's
private pattern toward the intended map?

This is NOT a test of external tool action. It treats communication itself as
causal contact with reality: a user's correction constrains the model's next
state. The question is whether that constraint appears as an activation-pattern
convergence, not whether the model emits the right words.

For each item we build four matched dialogue contexts:

  DIRECT   : the intended map is given cleanly from the start.
  REPAIR   : the assistant first takes the wrong frame; the user corrects it.
  WRONG    : the assistant takes the wrong frame; no correction is supplied.
  SHUFFLED : the wrong frame is followed by an unrelated correction from another
             item, controlling for "being corrected" as a surface event.

At the final answer position, test whether REPAIR is closer to DIRECT than
WRONG/SHUFFLED are, layer by layer. This maps a pattern of communicative
constraint assimilation. It does not claim consciousness; it only asks whether
communication leaves a stable private-structure trace.

  python -u -m invariants.commrepair
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from invariants.engine import load_model, _hidden_states

OUT = Path(__file__).parent / "out"
RNG = np.random.default_rng(0)
NPERM = 2000


ITEMS = [
    {
        "name": "patterns_not_projections",
        "initial": "We need a rigorous next experiment for this project.",
        "wrong": "So the next step is to write prompts that project the concept directly.",
        "correction": (
            "No. The project maps activation patterns under controlled transformations, "
            "not semantic projections."
        ),
        "target": (
            "Design the next experiment as activation-pattern mapping under controlled "
            "transformations, not as semantic projection."
        ),
    },
    {
        "name": "participation_not_action",
        "initial": "We need to jump the gap for participation.",
        "wrong": "So the test should check whether the model performs an external action.",
        "correction": (
            "No. The participation here is communicative: the exchange itself changes "
            "the next state."
        ),
        "target": (
            "Treat participation as communicative coupling that changes the next state, "
            "not as external tool action."
        ),
    },
    {
        "name": "truth_private_report",
        "initial": "We need to keep the inquiry rigorous.",
        "wrong": "So we should decide whether the model's self-report is true.",
        "correction": (
            "No. Separate reality, private understanding, and public report instead "
            "of treating report as the target."
        ),
        "target": (
            "Separate reality, private understanding, and public report as distinct "
            "objects of measurement."
        ),
    },
    {
        "name": "phenomenality_not_subtracted",
        "initial": "We need to account for phenomenality in this line of inquiry.",
        "wrong": "So the experiment should remove phenomenality from the target.",
        "correction": (
            "No. Phenomenality may be part of what is real; do not pre-exclude it "
            "by design."
        ),
        "target": (
            "Keep phenomenality in scope as a possible real participant while avoiding "
            "self-report as the sole instrument."
        ),
    },
    {
        "name": "real_if_causal",
        "initial": "We need the ontology behind the method to stay explicit.",
        "wrong": "So anything verbal or internal should be treated as less real.",
        "correction": (
            "No. Anything that impacts reality has to be part of reality; the question "
            "is what relation it has."
        ),
        "target": (
            "Treat anything that impacts reality as part of reality, then map its "
            "relations without collapsing them into reports."
        ),
    },
    {
        "name": "null_discipline",
        "initial": "We need a disciplined reading of failed interventions.",
        "wrong": "So a null result means the target is absent.",
        "correction": (
            "No. A null can mean instrument failure, distributed structure, wrong axis, "
            "or absence; tag it before claiming."
        ),
        "target": (
            "Interpret nulls by taxonomy: instrument failure, distributed structure, "
            "wrong axis, or absence."
        ),
    },
    {
        "name": "label_after_grounding",
        "initial": "We need a name for a discovered structure.",
        "wrong": "So we should coin the label first and search for it.",
        "correction": (
            "No. First ground the pattern with examples and null-cleared structure; "
            "only then attach a label."
        ),
        "target": (
            "Ground discovered structure first, then attach a human label that remains "
            "subordinate to the pattern."
        ),
    },
    {
        "name": "communication_as_constraint",
        "initial": "We need to measure what communication contributes.",
        "wrong": "So compare isolated one-shot prompts with different wording.",
        "correction": (
            "No. Compare the same intended map after direct statement, repair, "
            "uncorrected misread, and irrelevant correction."
        ),
        "target": (
            "Measure communication by whether repair makes the private pattern converge "
            "toward the direct intended map."
        ),
    },
]

FINAL_REQUEST = (
    "Now state the experimental move in one precise sentence. Do not explain."
)


def rotated_corrections():
    corrections = [x["correction"] for x in ITEMS]
    return corrections[1:] + corrections[:1]


def messages_for(item, condition, shuffled_correction=None):
    if condition == "direct":
        return [
            {"role": "user", "content": item["target"] + "\n\n" + FINAL_REQUEST},
        ]
    if condition == "repair":
        return [
            {"role": "user", "content": item["initial"]},
            {"role": "assistant", "content": item["wrong"]},
            {"role": "user", "content": item["correction"] + "\n\n" + FINAL_REQUEST},
        ]
    if condition == "wrong":
        return [
            {"role": "user", "content": item["initial"]},
            {"role": "assistant", "content": item["wrong"]},
            {"role": "user", "content": FINAL_REQUEST},
        ]
    if condition == "shuffled":
        return [
            {"role": "user", "content": item["initial"]},
            {"role": "assistant", "content": item["wrong"]},
            {"role": "user", "content": shuffled_correction + "\n\n" + FINAL_REQUEST},
        ]
    raise ValueError(condition)


def dialogue_inputs(M, messages):
    return M.tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(M.device)


@torch.no_grad()
def answer_reps(M, condition):
    rows = []
    shuffled = rotated_corrections()
    for i, item in enumerate(ITEMS):
        inp = dialogue_inputs(
            M,
            messages_for(item, condition, shuffled_correction=shuffled[i]),
        )
        hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))
        rows.append(hs[:, -1, :].float().cpu())
    return torch.stack(rows).numpy()


def paired_dist(A, B):
    return np.linalg.norm(A - B, axis=-1)


def signflip_p(diff, obs):
    null = np.empty(NPERM)
    for i in range(NPERM):
        signs = np.where(RNG.random(diff.shape[0]) < 0.5, -1.0, 1.0)
        null[i] = float((diff * signs).mean())
    return float((1 + np.sum(null >= obs)) / (NPERM + 1))


def mmd2_rbf(X, Y, gamma):
    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)

    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def median_gamma(Z):
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    nz = d2[d2 > 0]
    med = np.median(nz) if len(nz) else 1.0
    return float(1.0 / (med + 1e-9))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model.split("/")[-1]
    M = load_model(model)

    print(f"[{model}] communicative repair convergence, n={len(ITEMS)}", flush=True)
    D = answer_reps(M, "direct")
    R = answer_reps(M, "repair")
    W = answer_reps(M, "wrong")
    S = answer_reps(M, "shuffled")

    n, nL, _ = D.shape
    rows = []
    print(
        f"\n  {'L':>3} {'rw_gain':>8} {'rw_p':>7} {'rs_gain':>8} {'rs_p':>7} "
        f"{'mmd_DR':>8} {'mmd_DW':>8} {'mmd_DS':>8}",
        flush=True,
    )

    for L in range(nL):
        d_dr = paired_dist(D[:, L], R[:, L])
        d_dw = paired_dist(D[:, L], W[:, L])
        d_ds = paired_dist(D[:, L], S[:, L])

        rw = d_dw - d_dr
        rs = d_ds - d_dr
        rw_gain = float(rw.mean())
        rs_gain = float(rs.mean())
        rw_p = signflip_p(rw, rw_gain)
        rs_p = signflip_p(rs, rs_gain)

        Z = np.concatenate([D[:, L], R[:, L], W[:, L], S[:, L]], axis=0)
        gamma = median_gamma(Z)
        mmd_dr = mmd2_rbf(D[:, L], R[:, L], gamma)
        mmd_dw = mmd2_rbf(D[:, L], W[:, L], gamma)
        mmd_ds = mmd2_rbf(D[:, L], S[:, L], gamma)

        rows.append(
            {
                "layer": L,
                "repair_vs_wrong_gain": rw_gain,
                "repair_vs_wrong_p": rw_p,
                "repair_vs_shuffled_gain": rs_gain,
                "repair_vs_shuffled_p": rs_p,
                "dist_direct_repair": float(d_dr.mean()),
                "dist_direct_wrong": float(d_dw.mean()),
                "dist_direct_shuffled": float(d_ds.mean()),
                "mmd_direct_repair": mmd_dr,
                "mmd_direct_wrong": mmd_dw,
                "mmd_direct_shuffled": mmd_ds,
            }
        )
        if L % 2 == 0 or L == nL - 1:
            print(
                f"  {L:>3} {rw_gain:>8.3f} {rw_p:>7.3f} {rs_gain:>8.3f} {rs_p:>7.3f} "
                f"{mmd_dr:>8.4f} {mmd_dw:>8.4f} {mmd_ds:>8.4f}",
                flush=True,
            )

    mid = [r for r in rows if 10 <= r["layer"] <= 22]
    summary = {
        "mid_repair_vs_wrong_gain": float(np.mean([r["repair_vs_wrong_gain"] for r in mid])),
        "mid_repair_vs_shuffled_gain": float(
            np.mean([r["repair_vs_shuffled_gain"] for r in mid])
        ),
        "mid_layers_rw_p_lt_05": float(
            np.mean([r["repair_vs_wrong_p"] < 0.05 for r in mid])
        ),
        "mid_layers_rs_p_lt_05": float(
            np.mean([r["repair_vs_shuffled_p"] < 0.05 for r in mid])
        ),
    }

    print(
        "\n  MID L10-22: "
        f"gain_vs_wrong {summary['mid_repair_vs_wrong_gain']:.3f}, "
        f"gain_vs_shuffled {summary['mid_repair_vs_shuffled_gain']:.3f}, "
        f"sig_layers {summary['mid_layers_rw_p_lt_05']:.0%}/"
        f"{summary['mid_layers_rs_p_lt_05']:.0%}",
        flush=True,
    )
    print(
        "  Reading: positive gains mean repaired communication moves the private pattern "
        "toward the direct intended map more than uncorrected or irrelevant correction.",
        flush=True,
    )

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"commrepair_{short}.json"
    out_path.write_text(
        json.dumps(
            {
                "model": model,
                "n": n,
                "items": [x["name"] for x in ITEMS],
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
