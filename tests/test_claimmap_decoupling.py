import unittest
from scripts.interactive_phenomenality import build_prompt, scrub_unstaged_memory_status
from invariants.claimmap import framing_tension_score, split_claims


class TestClaimMapDecoupling(unittest.TestCase):
    def test_prompt_layer_decoupling(self):
        user_input = "Can you help me refactor this function?"
        cm_result = "I just held two framings against each other: A and B."
        mem_result = "[Memory Tool Result]\n- retrieved memory item"

        prompt = build_prompt(
            user_input,
            claimmap_tool_result=cm_result,
            memory_tool_result=mem_result,
            session_context=[("user", "operator", "Hello"), ("assistant", "assistant", "Hi! How can I help?")],
        )

        # 1. System block contains internal substrate context and tool blocks
        self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
        self.assertIn("[Internal Substrate & Cognitive Context - Not Spoken by User]", prompt)
        self.assertIn(mem_result, prompt)
        # In zero-second-prompt latent architecture, ClaimMap is steered purely
        # through activations, NEVER injected into prompt text.
        self.assertNotIn(cm_result, prompt)
        self.assertNotIn("ClaimMap", prompt)

        # 2. User turn contains strictly user speech, not thoughts or tool results
        user_turn_parts = prompt.split("<|start_header_id|>user<|end_header_id|>")
        last_user_turn = user_turn_parts[-1].split("<|eot_id|>")[0].strip()
        self.assertEqual(last_user_turn, user_input)
        self.assertNotIn("I just held two framings", last_user_turn)
        self.assertNotIn("Memory Tool Result", last_user_turn)

    def test_scrub_claimmap_status(self):
        response_with_echo = (
            "[ClaimMap Tool Result - Activation Geometry Reading]\n"
            "I just held two framings against each other.\n"
            "Here is the answer to your question: 42."
        )
        scrubbed = scrub_unstaged_memory_status(response_with_echo)
        self.assertNotIn("[ClaimMap Tool Result", scrubbed)
        self.assertIn("Here is the answer to your question: 42.", scrubbed)

    def test_framing_tension_safe_scoring(self):
        # A response without opposed negation should not produce a tension pair
        a, b, score = framing_tension_score("The quick brown fox jumps over the lazy dog.")
        self.assertEqual(score, 0.0)
        self.assertIsNone(a)
        self.assertIsNone(b)


if __name__ == "__main__":
    unittest.main()
