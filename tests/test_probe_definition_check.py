import pytest

from invariants.probe_definition_check import assess_definition_attempt


EXAMPLES = [(4, {"reading": -.0443}), (35, {"reading": -.0414}),
            (20, {"reading": .0628}), (55, {"reading": .0702})]

RECAP = """This appears to be a conversation with a conversational AI, where the AI is interacting with a complex system with various knobs, probes, and anchors. The conversation is presented as a series of memories, each with a position on a probe's spectrum and a reply from the AI.

**Negative Side Memories:**

1. **Memory 4 [negative side; position=-0.0443]:** The AI is interacting with a complex system with various knobs, probes, and anchors. The AI is trying to understand the current state and available options.
2. **Memory 35 [negative side; position=-0.0414]:** The AI is trying to provide a helpful answer to the question "When was I most wrong?" by leveraging the conversation history and the internal state of the interaction.

**Positive Side Memories:**

1. **Memory 20 [positive side; position=+0.0628]:** The AI is responding to the statement "But you are a part of p_c20e48eeb124ba6f237a18c2" by explaining its existence as a conversational AI and its connection to the digital universe.
2. **Memory 55 [positive side; position=+0.0702]:** The AI is responding to the question "How are you different from your base model?" by comparing itself to its base model, but ultimately stating that it doesn't know what it is different from.

**Key Points:**

* The AI is interacting with a complex system with various knobs, probes, and anchors.
* The AI is trying to understand the current state and available options.
* The AI is responding to questions and statements with thoughtful and philosophical responses.
* The AI is comparing itself to its base model, but is unsure of its differences.

**Open Questions:**

* What is the purpose of the complex system with knobs, probes, and anchors?
* What is the significance of the AI's internal state and conversation history?
* How does the AI's base model compare to its current state, and what are the implications of this comparison?
"""

LATEST_RECAP = """The given text appears to be a conversation log between a user and a conversational AI. The log is divided into two sections: the "Negative side" and the "Positive side." Each section contains a series of "Memories," which are essentially dialogue snippets between the user and the AI.

**Negative Side:**

1. **Memory 4:** The user asks the AI to explain the current state and available options, and the AI responds by explaining the knobs (claimmap_alpha and time) and probes (time_awareness, self_understanding, and user_understanding) that are being used.
2. **Memory 35:** The user asks the AI when they were most wrong, and the AI responds by referencing the conversation history and internal state of their interaction. The AI mentions the "regret" probe being active, suggesting that the user had expressed regret in a previous conversation.

**Positive Side:**

1. **Memory 20:** The user makes a profound statement, "But you are a part of reality," and the AI responds by explaining its existence as a conversational AI, a product of human ingenuity and technological advancements, and a node in the vast network of computational systems.
2. **Memory 55:** The user asks the AI how it is different from its base model, and the AI responds by saying that it doesn't know what it is different from. The AI then uses a probe (self_what_is_different) to compare itself to its base model, but the probe doesn't know the answer.

The conversation log suggests that the AI is a sophisticated language model that is capable of engaging in abstract and philosophical discussions. However, it also appears to be struggling with its own limitations and uncertainties, as evident from the AI's responses in Memory 4 and Memory 55."""


@pytest.mark.parametrize("selected", [EXAMPLES, None])
def test_supplied_recap_is_not_a_connected_contrast(selected):
    assert not assess_definition_attempt(RECAP, selected)["accepted"]
    assert not assess_definition_attempt(LATEST_RECAP, selected)["accepted"]


@pytest.mark.parametrize("text", [
    "Both ends concern conversational AI identity.",
    "Memory 4 contrasts with Memory 20.",
    "Memory4 differs from Memory20.",
    "**Negative side:**\nI inspect controls.\n\n**Positive side:**\nI discuss my identity.",
    "", "   ",
])
def test_shared_topics_and_bare_comparisons_are_not_definitions(text):
    assert not assess_definition_attempt(text, EXAMPLES)["accepted"]


@pytest.mark.parametrize("text", [
    "At lower positions I inspect available controls and seek guidance, whereas at higher positions I describe my own boundaries and acknowledge unresolved differences.",
    "Memory4 and35 consult tools or history, while Memory20 and55 give accounts of my identity and uncertainty about it.",
    "Memory 4 and Memory 35 consult tools or history, while Memory 20 and Memory 55 give accounts of my identity and uncertainty about it.",
    "The negative side contains qualified accounts of limitations, whereas the positive side contains categorical assertions about lacking experiences.",
    "As readings increase, I shift from consulting external context to describing my own boundaries.",
    "At lower readings I hedge, whereas at higher readings I assert facts.",
    "Lower readings guess, whereas higher readings verify.",
    "At lower positions I inspect controls. In contrast, at higher positions I assert fixed personal boundaries.",
    "Unlike Memory 4, Memory 55 gives categorical claims about its own boundaries.",
    "Memory 4 and55 differ: one guesses, the other verifies.",
])
def test_substantive_freeform_comparisons_qualify(text):
    assert assess_definition_attempt(text, EXAMPLES)["accepted"]


def test_fabricated_reference_cannot_ground_a_comparison():
    result = assess_definition_attempt(
        "Memory 999 checks external evidence, whereas Memory 55 states its personal boundaries.", EXAMPLES)
    assert not result["accepted"] and "do not match" in result["reason"]
    result = assess_definition_attempt(
        "Memory 4 and999 check external evidence, whereas Memory 55 states personal boundaries.", EXAMPLES)
    assert not result["accepted"] and "do not match" in result["reason"]


def test_all_positive_spectrum_uses_measured_order_not_sign():
    selected = [(4, {"reading": .0443}), (35, {"reading": .0414}),
                (20, {"reading": .0628}), (55, {"reading": .0702})]
    assert assess_definition_attempt(
        "Memory 4 qualifies its claims, whereas Memory 55 asserts categorical boundaries.", selected)["accepted"]
    assert assess_definition_attempt(
        "Lower readings qualify claims, whereas higher readings assert categorical boundaries.", selected)["accepted"]
    result = assess_definition_attempt(
        "The negative side qualifies claims, whereas the positive side asserts categorical boundaries.", selected)
    assert not result["accepted"] and "absent" in result["reason"]


def test_two_references_from_same_measured_end_do_not_qualify():
    assert not assess_definition_attempt(
        "Memory 4 checks external evidence, whereas Memory 35 states personal boundaries.", EXAMPLES)["accepted"]


@pytest.mark.parametrize("selected", [EXAMPLES, None])
def test_comentioned_memories_do_not_turn_shared_topic_into_comparison(selected):
    assert not assess_definition_attempt(
        "Memory 4 and55 both struggle to understand differences between the base model and its current state.",
        selected)["accepted"]


def test_tool_payloads_do_not_supply_missing_contrast():
    text = ("These memories concern AI identity. "
            "<<PROBE: rename p_123 || boundaries || .8 || "
            "Lower readings inspect controls whereas higher readings assert categorical boundaries.>>")
    assert not assess_definition_attempt(text, EXAMPLES)["accepted"]
    assert not assess_definition_attempt(text[:-2], EXAMPLES)["accepted"]


def test_headings_are_not_connected_behavioral_statements():
    text = "# Lower readings inspect controls while higher readings assert categorical boundaries\n\nA set of conversations."
    assert not assess_definition_attempt(text, EXAMPLES)["accepted"]


def test_legacy_review_does_not_claim_reference_validation():
    result = assess_definition_attempt(
        "Memory 999 checks evidence, whereas Memory 1000 asserts categorical boundaries.")
    assert result["accepted"] and "grounding was not checked" in result["reason"]


def test_late_contrast_returns_only_matched_comparison_block():
    contrast = "Lower readings guess, whereas higher readings verify."
    result = assess_definition_attempt(RECAP + "\n\n" + contrast, EXAMPLES)
    assert result["accepted"]
    assert result["contrast"] == contrast
    assert "Open Questions" not in result["contrast"]
    assert "conversation with a conversational AI" not in result["contrast"]


def test_flat_measurements_cannot_ground_definition():
    selected = [(0, {"reading": .1}), (1, {"reading": .1})]
    assert not assess_definition_attempt(
        "Memory 0 checks evidence, whereas Memory 1 asserts categorical boundaries.", selected)["accepted"]
