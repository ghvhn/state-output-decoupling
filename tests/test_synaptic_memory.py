"""Unit tests for SynapticMemoryEngine and PlasticWeightHook."""

import unittest
import torch
import torch.nn as nn
from invariants.synaptic_memory import PlasticWeightHook, SynapticMemoryEngine


class TestSynapticMemory(unittest.TestCase):
    def test_plastic_weight_hook_alters_output(self):
        # A simple linear layer representing mlp.down_proj: 64 -> 64
        linear = nn.Linear(64, 64, bias=False)
        hook = PlasticWeightHook(module=linear, layer_idx=16, module_name="test_mlp")

        x_test = torch.randn(1, 4, 64)
        out_base = linear(x_test).clone()

        # Add associative memory: key is direction k, target shift is direction v
        key = torch.randn(64)
        target = torch.randn(64)
        hook.add_associative_trace("trace_01", key_vector=key, value_vector=target, alpha=0.1)

        # 1. Forward pass with an input that aligns with key
        x_aligned = key.unsqueeze(0).unsqueeze(0)  # [1, 1, 64]
        out_with_memory = linear(x_aligned)

        # Without memory, output would be:
        with torch.no_grad():
            hook.keys.clear()  # Temporarily disable
            out_unmodified = linear(x_aligned).clone()
            hook.keys.append(key / key.norm())

        # The difference should align strongly with target direction
        delta = (out_with_memory - out_unmodified).squeeze()
        cos_sim = torch.cosine_similarity(delta, target, dim=0).item()
        self.assertGreater(cos_sim, 0.95)

        hook.remove()

    def test_orthogonal_input_does_not_fire_memory(self):
        linear = nn.Linear(32, 32, bias=False)
        hook = PlasticWeightHook(module=linear, layer_idx=16)

        key = torch.zeros(32)
        key[0] = 1.0  # Key points along dim 0
        target = torch.zeros(32)
        target[1] = 1.0

        hook.add_associative_trace("t1", key_vector=key, value_vector=target, alpha=0.1)

        # Input strictly orthogonal to key
        x_ortho = torch.zeros(1, 1, 32)
        x_ortho[0, 0, 5] = 2.0  # Points along dim 5

        out_with_mem = linear(x_ortho)
        with torch.no_grad():
            hook.keys.clear()
            out_without_mem = linear(x_ortho)

        diff_norm = (out_with_mem - out_without_mem).norm().item()
        self.assertAlmostEqual(diff_norm, 0.0, places=5)
        hook.remove()

    def test_synaptic_decay(self):
        linear = nn.Linear(16, 16)
        hook = PlasticWeightHook(module=linear, layer_idx=16)
        k, v = torch.randn(16), torch.randn(16)
        hook.add_associative_trace("decay_test", k, v, alpha=0.1)

        self.assertAlmostEqual(hook.alphas[0], 0.1)
        hook.decay(rate=0.5)
        self.assertAlmostEqual(hook.alphas[0], 0.05)
        hook.remove()


if __name__ == "__main__":
    unittest.main()
