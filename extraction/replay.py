"""
Replays a conversation turn by turn through the model, capturing residual
stream activations on the specified turns.

Context is built with the model's native CHAT TEMPLATE (special role tokens),
not a raw "role: text" transcript. This matters: an instruct model only enters
the dialogue regime when it sees the tokens it was trained to read as turns.
Feeding a flat transcript makes it *continue* text rather than *participate* in
an exchange — so activations captured that way reflect the wrong cognitive mode.
If the tokenizer exposes no chat template, we fall back to the raw transcript.

Two modes:
- replay: feeds pre-written turns as context, captures activations while the
  model reads them. Fast, good for scale.
- live: feeds user turns, generates assistant responses token by token,
  capturing activations at each generated token — the model in its active state.

Each model's activations are evaluated against itself only. What passes between
models is plain text — the bridge is the conversation, not the geometry.
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer
from extraction.hooks import extract_bands


def _chat_tokens(model, messages, add_generation_prompt=False):
    """
    Tokenize [{role, content}, ...] with the model's chat template so the model
    reads genuine dialogue turns. Falls back to a raw transcript if no template
    is available. Returns a [1, seq] LongTensor on the model's device.
    """
    tok = getattr(model, "tokenizer", None)
    if tok is not None and getattr(tok, "chat_template", None):
        try:
            enc = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
                return_dict=True,
            )
            return enc["input_ids"].to(model.cfg.device)
        except Exception:
            pass  # malformed turn order etc. — fall through to raw
    raw = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
    if add_generation_prompt:
        raw += "assistant: "
    return model.to_tokens(raw)


def _eos_ids(model):
    """End-of-turn ids: eos plus Llama's <|eot_id|> when present."""
    ids = set()
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return ids
    eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(x) for x in eos)
    try:
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            ids.add(eot)
    except Exception:
        pass
    return ids


def replay(
    model: HookedTransformer,
    conversation: list[dict],
    bands: dict[str, list[int]],
    capture: str = "generated_turns",
    quantize: bool = True,
) -> list[dict]:
    """
    Feeds the full conversation as chat-formatted context. Turns are read, not
    generated. capture: "generated_turns" | "all_turns" | "both"
    """
    records = []
    is_self_dialogue = all(
        t["role"] in ("user", "assistant") for t in conversation
    ) and sum(1 for t in conversation if t["role"] == "user") > 1

    messages = []
    for i, turn in enumerate(conversation):
        role, content = turn["role"], turn["content"]
        messages.append({"role": role, "content": content})

        should_capture = (
            capture == "all_turns"
            or (capture == "generated_turns" and role == "assistant")
            or capture == "both"
            or is_self_dialogue  # both sides generated — capture everything
        )
        if not should_capture:
            continue

        tokens = _chat_tokens(model, messages, add_generation_prompt=False)
        band_activations = extract_bands(model, tokens, bands, quantize=quantize)
        records.append({
            "turn_index": i,
            "role": role,
            "content": content,
            "mode": "replay",
            "bands": band_activations,
        })
    return records


def live(
    model: HookedTransformer,
    conversation: list[dict],
    bands: dict[str, list[int]],
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    quantize: bool = True,
) -> list[dict]:
    """
    Feeds user turns, generates assistant responses token by token, capturing
    residual-stream activations at every generated token. Context uses the chat
    template; generation stops at the model's end-of-turn token.

    Returns one record per assistant turn, with per-token activation
    trajectories stacked along the token axis.
    """
    records = []
    eos_ids = _eos_ids(model)
    messages = []

    for i, turn in enumerate(conversation):
        role, content = turn["role"], turn["content"]
        if role == "user":
            messages.append({"role": "user", "content": content})
            continue

        # Assistant turn: generate from a chat-formatted generation prompt.
        tokens = _chat_tokens(model, messages, add_generation_prompt=True)
        token_band_activations = {band: [] for band in bands}
        generated_ids = []

        for _ in range(max_new_tokens):
            # Capture activations at the current last position — the one
            # resolving the next token.
            step_bands = extract_bands(model, tokens, bands, quantize=quantize)
            for band in bands:
                token_band_activations[band].append(step_bands[band][:, -1:, :])

            with torch.no_grad():
                logits = model(tokens)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1]
            tid = int(next_token.item())
            generated_ids.append(tid)
            if tid in eos_ids:
                break
            tokens = torch.cat(
                [tokens, next_token.view(1, 1).to(tokens.device)], dim=1
            )

        stacked_bands = {}
        for band in bands:
            if token_band_activations[band]:
                stacked_bands[band] = np.concatenate(
                    token_band_activations[band], axis=1
                )

        tok = getattr(model, "tokenizer", None)
        if tok is not None:
            text = tok.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            text = model.to_string(
                torch.tensor(generated_ids, dtype=torch.long)
            ).strip()

        messages.append({"role": "assistant", "content": text})
        records.append({
            "turn_index": i,
            "role": "assistant",
            "content": text,
            "mode": "live",
            "bands": stacked_bands,
        })
    return records
