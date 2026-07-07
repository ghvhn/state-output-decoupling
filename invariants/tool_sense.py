"""Tool-sensing seam: any tool can be called whenever its state trigger crosses.

Tools stop being turn-scoped or tag-invoked. Each registers a detector that
reads the deliberation (the text so far, or later the residual state) into a
scalar signal, plus an action that fires when the signal clears the tool's
live-tuned threshold. `ToolSense` is passed to `generate_agentic_text` as the
mid_chunk_hook: it runs at every chunk seam, so a tool fires MID-THOUGHT and its
steering bends the remaining chunks of the same answer.

Firing consults the tuner's threshold (`tuner.get`) rather than logging a sample
every chunk -- the per-turn distribution + credit stay owned by the post-turn
path, so the tuner's lift semantics remain one-sample-per-turn. Adding a tool is
just `register(Tool(name, detect, act))`; ClaimMap is the first through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Tool:
    name: str                                   # tuner trigger name (born tunable)
    detect: Callable[..., tuple[float, Any]]    # (text, state) -> (signal, payload or None)
                                                # state = the live activation dict
                                                # (last_phenomenality, ...). A tool fires
                                                # from STATE, not a text tag.
    act: Callable[[Any, Any], Optional[list]]   # (payload, model) -> steer handles to
                                                # apply to the remaining generation
    comparator: str = ">="                      # fire when signal >= threshold (or <=)
    activation_criteria: str = ""               # human-readable condition for when it should fire
    steer_magnitude: str = ""                   # human-readable alpha/strength surface


class ToolSense:
    def __init__(self, model, tuner):
        self.model = model
        self.tuner = tuner
        self.tools: list[Tool] = []
        self._handles: list = []
        self._fired: set[str] = set()  # fire each tool at most once per generation
        self.injected_text: str = ""
        # Optional (text, state) -> str, wired by the shell to drain any lines
        # the operator typed mid-generation. Runs every seam; its return is
        # appended to the live stream (the model then redirects or folds in).
        self.input_drain = None
        # RELEASE-DEPENDENCY: {tool_name: probability}. When set, that fraction
        # of fire decisions is DECOUPLED from the signal (a coin flip), so the
        # trigger and the action decorrelate -- the only way credit lift can
        # separate "the trigger caused a good turn" from "acting helped anyway".
        self.release_probs: dict[str, float] = {}
        self.release_total = 0  # decoupled decisions this session (observability)

    def register(self, tool: Tool):
        self.tools.append(tool)
        return self

    def _crosses(self, tool: Tool, signal: float) -> bool:
        threshold = self.tuner.get(tool.name)
        if tool.comparator == "<=":
            return signal <= threshold
        return signal >= threshold

    def _decide(self, tool: Tool, signal: float) -> tuple[bool, bool]:
        """(fire, decoupled). Normally fire == signal crosses the bar. If this
        tool is released, `prob` of the time the decision is a coin flip
        independent of the signal -- reported so the fire can be labeled."""
        import random
        prob = float(self.release_probs.get(tool.name, 0.0) or 0.0)
        if prob > 0.0 and random.random() < prob:
            return (random.random() < 0.5, True)
        return (self._crosses(tool, signal), False)

    def __call__(self, text: str, state: Optional[dict] = None):
        """Run every detector on the text-so-far AND the live activation state;
        fire the ones that cross. `state` is the engine's activation dict (e.g.
        state["last_phenomenality"]) so a tool triggers from the model's own
        internal signature, not a text tag."""
        for tool in self.tools:
            if tool.name in self._fired:
                continue
            try:
                signal, payload = tool.detect(text, state)
            except TypeError:
                # Back-compat: a detector that only accepts text.
                signal, payload = tool.detect(text)
            if payload is None:
                continue
            fire, decoupled = self._decide(tool, signal)
            if decoupled:
                self.release_total += 1
            if fire:
                handles = tool.act(payload, self.model) or []
                self._handles.extend(handles)
                self._fired.add(tool.name)
        # Always-ingest operator input: drain last so a fresh interjection lands
        # on top of any tool result this seam produced.
        if self.input_drain is not None:
            try:
                drained = self.input_drain(text, state)
            except Exception:
                drained = ""
            if drained:
                self.injected_text = (self.injected_text or "") + drained

    def cleanup(self):
        """Remove any steering registered this generation and reset per-gen state.
        Called by generate_agentic_text in its finally, so hooks never leak."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self._fired = set()
        self.injected_text = ""
