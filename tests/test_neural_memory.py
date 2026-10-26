import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from invariants.neural_memory import NeuralMemory


class NeuralMemoryTests(unittest.TestCase):
    def test_language_learning_base_isolation_and_reload(self):
        torch.manual_seed(31)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=24, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        )).eval()
        ids = torch.tensor([[1, 4, 5, 9, 10], [1, 6, 7, 11, 12]])
        labels = ids.clone()
        labels[:, :3] = -100
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            baseline = model(ids).logits.clone()
        memory = NeuralMemory(16, rank=8).attach(model.model.layers[-1])
        self.addCleanup(memory.detach)
        with torch.no_grad():
            torch.testing.assert_close(model(ids).logits, baseline)
        report = memory.learn(model, ids, labels, steps=60, learning_rate=0.03)
        self.assertLess(report['final_loss'], report['initial_loss'] - 0.1)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.training)
        with torch.no_grad():
            learned = model(ids).logits.clone()
            with memory.suspended():
                torch.testing.assert_close(model(ids).logits, baseline)
        checkpoint = memory.state_dict()
        memory.detach()
        restored = NeuralMemory(16, rank=8)
        restored.load_state_dict(checkpoint)
        restored.attach(model.model.layers[-1])
        self.addCleanup(restored.detach)
        with torch.no_grad():
            torch.testing.assert_close(model(ids).logits, learned)
        with self.assertRaises(ValueError):
            restored.learn(model, ids, torch.full_like(ids, -100))


if __name__ == '__main__':
    unittest.main()
