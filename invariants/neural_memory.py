"""Experimental language-trained memory weights, separate from base weights.

Unlike a retrieved text prefix or an arbitrary activation nudge, this adapter
learns continuations using the host causal language model's prediction loss.
Attach one owner at a time to a model; detach before sharing it across agents.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn


class NeuralMemory(nn.Module):
    def __init__(self, width: int, rank: int = 16, strength: float = 1.0):
        super().__init__()
        if width < 1 or rank < 1 or not math.isfinite(strength) or strength < 0:
            raise ValueError("Invalid memory dimensions or strength")
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.register_buffer("strength", torch.tensor(float(strength)))
        self.enabled = True
        self._handle = None

    def forward(self, hidden):
        # Keep trainable memory in float32 even when the host uses half precision.
        delta = self.up(torch.tanh(self.down(hidden.float())))
        return hidden + (delta * self.strength).to(hidden.dtype)

    def attach(self, module):
        """Attach to a decoder layer returning hidden states or a hidden tuple."""
        self.detach()

        def apply(_module, _inputs, output):
            if not self.enabled:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            changed = self(hidden)
            return (changed,) + output[1:] if isinstance(output, tuple) else changed

        self._handle = module.register_forward_hook(apply)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @contextmanager
    def suspended(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def learn(self, model, input_ids, labels, *, steps=50, learning_rate=0.01,
              attention_mask=None):
        """Learn token batches; cue/padding labels must be -100.

        Supply complete cue+continuation tokens. Only continuation labels should
        be supervised. Replay older examples in the same batch to limit forgetting.
        Returns measured loss, not a claim of exact or paraphrased recall.
        """
        if self._handle is None or not self.enabled:
            raise ValueError("Attach and enable memory before learning")
        if steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Invalid training budget")
        if input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise ValueError("Expected equally shaped [batch, tokens] inputs and labels")
        if not (labels[:, 1:] != -100).any():
            raise ValueError("At least one continuation token must be supervised")
        base_parameters = list(model.parameters())
        flags = [p.requires_grad for p in base_parameters]
        modes = [(module, module.training) for module in model.modules()]
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best = {k: v.detach().clone() for k, v in self.state_dict().items()}
        history = []
        best_loss = float("inf")
        try:
            model.eval()
            for parameter in base_parameters:
                parameter.requires_grad_(False)
            with torch.enable_grad():
                for step in range(steps + 1):
                    optimizer.zero_grad(set_to_none=True)
                    result = model(input_ids=input_ids, labels=labels,
                                   attention_mask=attention_mask, use_cache=False)
                    loss = result.loss
                    score = float(loss.detach())
                    if not math.isfinite(score):
                        raise ValueError("Non-finite memory training loss")
                    history.append(score)
                    if score < best_loss:
                        best_loss = score
                        best = {k: v.detach().clone() for k, v in self.state_dict().items()}
                    if step < steps:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                        optimizer.step()
        finally:
            self.load_state_dict(best)
            optimizer.zero_grad(set_to_none=True)
            for parameter, flag in zip(base_parameters, flags):
                parameter.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode
        return {"initial_loss": history[0], "final_loss": best_loss,
                "steps": steps, "loss_history": history}
