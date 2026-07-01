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
    detect: Callable[[str], tuple[float, Any]]  # text -> (signal, payload or None)
    act: Callable[[Any, Any], Optional[list]]   # (payload, model) -> steer handles to
                                                # apply to the remaining generation
    comparator: str = ">="                      # fire when signal >= threshold (or <=)


class ToolSense:
    def __init__(self, model, tuner):
        self.model = model
        self.tuner = tuner
        self.tools: list[Tool] = []
        self._handles: list = []
        self._fired: set[str] = set()  # fire each tool at most once per generation

    def register(self, tool: Tool):
        self.tools.append(tool)
        return self

    def _crosses(self, tool: Tool, signal: float) -> bool:
        threshold = self.tuner.get(tool.name)
        if tool.comparator == "<=":
            return signal <= threshold
        return signal >= threshold

    def __call__(self, text: str):
        """Run every detector on the text so far; fire the ones that cross."""
        for tool in self.tools:
            if tool.name in self._fired:
                continue
            signal, payload = tool.detect(text)
            if payload is None:
                continue
            if self._crosses(tool, signal):
                handles = tool.act(payload, self.model) or []
                self._handles.extend(handles)
                self._fired.add(tool.name)

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
