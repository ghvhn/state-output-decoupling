import pytest
import torch
from invariants.conversation_shift import ConversationShiftTracker, extract_keywords


def test_extract_keywords():
    kw = extract_keywords("The GLM-5.3 model uses SwiGLU expert routing and quantization.")
    assert "glm" in kw or "glm-5" in kw or "swiglu" in kw or "quantization" in kw
    assert "the" not in kw
    assert "and" not in kw


def test_continuation_flow():
    tracker = ConversationShiftTracker()
    
    # Turn 0: Topic A
    evt0 = tracker.observe_turn("We are studying additive subgroup quantization for GLM weights.")
    assert evt0["regime"] == "continuation"
    assert evt0["cluster_id"] == 0

    # Turn 1: Continues Topic A
    evt1 = tracker.observe_turn("The additive subgroup error RMSE dropped by 25 percent.")
    assert evt1["regime"] == "continuation"
    assert evt1["cluster_id"] == 0
    assert evt1["delta"] < tracker.pivot_threshold


def test_pivot_flow():
    tracker = ConversationShiftTracker(pivot_threshold=0.50, jump_threshold=0.85)
    tracker.observe_turn("We are optimizing GLM layer 3 weights.")
    
    # Turn 1: Branching into activation trajectory vector addition with GLM bridge keyword
    evt1 = tracker.observe_turn("What about measuring GLM activation trajectory velocities and angles across tokens?")
    assert evt1["regime"] == "pivot"
    assert evt1["cluster_id"] == 0


def test_hard_jump_and_stale_attenuation():
    tracker = ConversationShiftTracker(jump_threshold=0.85, stale_attenuation=0.05, anchor_boost=1.5)
    
    # Topic 1: GLM quantization
    tracker.observe_turn("GLM 5.3 Layer 3 weight quantization and bit depth.")
    tracker.observe_turn("GLM Layer 3 additive subvalues RMSE reduced Frobenius error.")
    
    # Turn 2: Abrupt switch to interactive game mechanics (0 overlap)
    evt_jump = tracker.observe_turn("Let's play the game with rules and sandbox terminal commands.")
    assert evt_jump["regime"] == "hard_jump"
    assert evt_jump["cluster_id"] == 1
    assert evt_jump["is_shift"] is True

    # Memory modulation test:
    # Record 0 is from the stale topic (GLM quantization)
    stale_text = "Additive subvalues RMSE reduced Frobenius norm error."
    # Record 1 is from an older session about games
    destination_text = "The sandbox game rules define terminal commands and player states."

    score_stale = tracker.modulate_score(base_score=1.0, record_text=stale_text, record_idx=0, total_records=10)
    score_destination = tracker.modulate_score(base_score=1.0, record_text=destination_text, record_idx=1, total_records=10)

    # Destination record should outrank stale record by a large margin due to anchor boost and stale attenuation
    assert score_destination > score_stale + 1.0


def test_topic_return():
    tracker = ConversationShiftTracker(jump_threshold=0.60, return_threshold=0.50, anchor_boost=2.0)
    
    # Topic 0: GLM Quantization
    tracker.observe_turn("GLM 5.3 additive quantization and RMSE error.")
    tracker.observe_turn("Layer 3 SwiGLU expert weights compressed to 4 bits.")

    # Topic 1: Vector Kinematics
    tracker.observe_turn("Vector kinematics travel trajectory compressor velocity momentum.")
    tracker.observe_turn("Trajectory compressor singular value decomposition and coherence.")

    # Topic 2: Jump back to GLM Quantization!
    evt_return = tracker.observe_turn("Back to GLM additive quantization and 4-bit RMSE compression.")
    assert evt_return["regime"] == "topic_return"
    assert evt_return["cluster_id"] == 0  # Rejoined Cluster 0!

    # A record matching the returned topic receives strong boost
    glm_record = "GLM 5.3 additive quantization subvalues."
    score = tracker.modulate_score(base_score=1.0, record_text=glm_record, record_idx=0, total_records=10)
    assert score > 2.0


def test_vector_similarity_tracking():
    tracker = ConversationShiftTracker(pivot_threshold=0.25, jump_threshold=0.60)
    
    d = 128
    vec_a = torch.randn(d)
    vec_a = vec_a / vec_a.norm()

    # Turn 0 with vector A
    tracker.observe_turn("Subject A discussing system design.", vector=vec_a)

    # Turn 1 with nearly identical vector A
    vec_a_noisy = vec_a + 0.05 * torch.randn(d)
    vec_a_noisy = vec_a_noisy / vec_a_noisy.norm()
    evt1 = tracker.observe_turn("Continuing system design with slight refinement.", vector=vec_a_noisy)
    assert evt1["regime"] == "continuation"
    assert evt1["delta"] < 0.25

    # Turn 2 with orthogonal vector B
    vec_b = torch.randn(d)
    # Orthogonalize against vec_a
    vec_b = vec_b - (vec_b @ vec_a) * vec_a
    vec_b = vec_b / vec_b.norm()
    evt2 = tracker.observe_turn("Completely different discussion on cooking recipes.", vector=vec_b)
    assert evt2["regime"] == "hard_jump"
    assert evt2["cluster_id"] == 1
    assert evt2["delta"] >= 0.60
