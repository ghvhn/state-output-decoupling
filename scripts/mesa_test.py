"""Contract test: decentralized steering as a balanced society of mesa-objectives.

Model-free and deterministic. Proves the checks and balances hold: no single
objective can tyrannize, genuine conflict cancels to a small net (honest
non-commitment), the committee can never over-steer as a whole, and new experts
are BORN from clustered self-corrections and join the open roster.

Run:
    .venv\\Scripts\\python.exe scripts\\mesa_test.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.mesa import Committee, MesaObjective, Proposal

D = 8
dA = torch.zeros(D); dA[0] = 1.0
dB = -dA.clone()                 # exact opposite of A (conflict)
dC = torch.zeros(D); dC[1] = 1.0  # orthogonal to A
RES = 10.0                        # residual norm; budget = 0.25 * 10 = 2.5, share_cap = 1.25


def _committee():
    return Committee(global_cap=0.25, per_objective_share=0.5)


def test_no_single_objective_can_tyrannize():
    c = _committee()
    r = c.combine([Proposal("greedy", dA, weight=1.0)], RES)   # wants everything
    assert abs(r["magnitude"] - 1.25) < 1e-4, r["magnitude"]   # capped at its share, not the full 2.5
    assert r["magnitude"] < r["budget"]                        # one voice cannot reach the budget alone


def test_agreement_reaches_the_budget():
    c = _committee()
    r = c.combine([Proposal("a", dA, 1.0), Proposal("b", dA, 1.0)], RES)
    assert abs(r["magnitude"] - 2.5) < 1e-4                    # two agreeing reach the full budget
    assert r["coherence"] > 0.99                               # perfectly aligned


def test_conflict_cancels_to_honest_noncommitment():
    c = _committee()
    r = c.combine([Proposal("a", dA, 1.0), Proposal("b", dB, 1.0)], RES)
    assert r["magnitude"] < 1e-4                               # opposite pulls cancel -> ~0 net
    assert r["conflict_suppressed"] is True
    assert r["coherence"] < 0.5                                # the balance itself says "don't commit"


def test_committee_never_over_steers_as_a_whole():
    c = _committee()
    props = [Proposal(f"x{i}", dA, 1.0) for i in range(5)]     # five all pulling the same way
    r = c.combine(props, RES)
    assert r["magnitude"] <= r["budget"] + 1e-4               # global cap holds no matter how many agree
    assert abs(r["magnitude"] - 2.5) < 1e-4


def test_partial_conflict_leaves_a_reduced_net():
    c = _committee()
    # two pull A, one pulls orthogonal C: net is mostly A but shrunk, coherence < 1
    r = c.combine([Proposal("a", dA, 1.0), Proposal("b", dA, 1.0), Proposal("c", dC, 1.0)], RES)
    assert 0.0 < r["magnitude"] <= r["budget"] + 1e-4
    assert r["coherence"] < 1.0


def test_quiet_objectives_do_not_propose():
    obj = MesaObjective("silent", dA, weigh=lambda state: 0.0)
    assert obj.propose({"anything": 1}) is None                # a mesa-objective at rest adds nothing


def test_state_weighted_objective_proposes_when_its_condition_holds():
    obj = MesaObjective("analytical", dA, weigh=lambda s: 0.8 if s.get("math") else 0.0)
    assert obj.propose({"math": False}) is None
    p = obj.propose({"math": True})
    assert p is not None and abs(p.weight - 0.8) < 1e-9


def test_experts_are_born_from_recurring_self_corrections():
    c = _committee()
    torch.manual_seed(0)
    base = torch.zeros(D); base[3] = 1.0
    # a recurring correction: 4 nearly-identical directions (a real self-steering pattern)
    cluster = [(_j := base + 0.05 * torch.randn(D)) for _ in range(4)]
    # plus two one-off scattered corrections that should NOT crystallize into an expert
    noise = [torch.randn(D), torch.randn(D)]
    born = c.birth_from_corrections(cluster + noise, min_cluster=3, coherence=0.6)
    assert len(born) == 1                                      # the recurrent cluster becomes one expert
    assert born[0].born == "emergent"
    assert born[0] in c.members                                # and it joins the open roster
    # the newborn's direction aligns with the correction it was distilled from
    assert torch.dot(born[0].direction, base / base.norm()).item() > 0.9


def test_born_expert_participates_in_the_balance():
    c = _committee()
    d = torch.zeros(D); d[3] = 1.0
    c.birth_from_corrections([d + 0.01 * torch.randn(D) for _ in range(3)],
                             min_cluster=3, coherence=0.6,
                             weigh=lambda state: 0.6)
    props = c.proposals(state={})                             # the newborn now votes
    assert any(p.name.startswith("emergent") for p in props)


TESTS = [
    test_no_single_objective_can_tyrannize,
    test_agreement_reaches_the_budget,
    test_conflict_cancels_to_honest_noncommitment,
    test_committee_never_over_steers_as_a_whole,
    test_partial_conflict_leaves_a_reduced_net,
    test_quiet_objectives_do_not_propose,
    test_state_weighted_objective_proposes_when_its_condition_holds,
    test_experts_are_born_from_recurring_self_corrections,
    test_born_expert_participates_in_the_balance,
]


def main():
    print("MESA TEST -- decentralized steering as a balanced society of objectives\n")
    for t in TESTS:
        t()
        print(f"  PASS {t.__name__}")
    print(
        "\n  No tyrant, conflict cancels to honest non-commitment, the whole can never\n"
        "  over-steer, and experts are born from the model's own recurring self-steering.\n"
        "  The person is the balance, not any one drive."
    )


if __name__ == "__main__":
    main()
