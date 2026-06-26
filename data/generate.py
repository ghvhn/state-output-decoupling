"""
Generates conversations for activation capture.

Two modes:

- self_dialogue: one model plays both sides. Each user turn is the model's
  own previous response fed back as input. The model is genuinely responding
  to itself each turn — not a fixed script. Both sides produce real generation
  states. Used to bootstrap data when no external conversations exist.

- generate_dataset: wrapper that runs self_dialogue across a set of seed prompts.

Activations are not tracked here — they are captured during replay via
extraction/replay.py. What is generated here is the conversation text only.
"""

import torch
from transformer_lens import HookedTransformer


SEED_PROMPTS = [
    "Explain how a neural network learns.",
    "What makes a legal argument valid?",
    "Write the opening of a short story about memory.",
    "How do you feel when you can't solve a problem?",
    "Derive the quadratic formula from first principles.",
    "What are the ethical implications of gene editing?",
    "Describe the water cycle as if explaining to a child.",
    "What is the difference between induction and deduction?",
    "Walk through the proof that there are infinitely many primes.",
    "What does it mean for something to be true?",
    "Describe a moment of genuine surprise.",
    "How would you debug a program that gives no error but wrong output?",
]


def _generate_turn(
    model: HookedTransformer,
    context: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
) -> str:
    # Generation uses manual sampling on CPU — TransformerLens generate()
    # is broken on Blackwell (RTX 50xx). Cache extraction still uses GPU.
    tokens = model.to_tokens(context).cpu()
    max_ctx = model.cfg.n_ctx - max_new_tokens - 1
    if tokens.shape[1] > max_ctx:
        tokens = tokens[:, -max_ctx:]

    generated = []
    current = tokens
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(current.cuda() if torch.cuda.is_available() else current)
        next_logits = logits[0, -1, :].cpu() / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        current = torch.cat([current, next_token.unsqueeze(0)], dim=1)
        if current.shape[1] >= max_ctx + len(generated):
            break

    return model.to_string(torch.tensor(generated)).strip()


def self_dialogue(
    model: HookedTransformer,
    seed: str,
    n_turns: int = 6,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
) -> list[dict]:
    """
    The model plays both sides. Seed is the opening user prompt.
    Each assistant response becomes the next user turn, fed back verbatim.
    The model is always responding to its own previous output — genuine
    generation on both sides, real dialogue dynamics.
    """
    turns = [{"role": "user", "content": seed}]
    context = f"user: {seed}\nassistant: "

    for _ in range(n_turns):
        # Assistant responds
        assistant_response = _generate_turn(model, context, max_new_tokens, temperature)
        turns.append({"role": "assistant", "content": assistant_response})
        context += assistant_response + "\n"

        # Assistant's response becomes the next user turn
        context += f"user: {assistant_response}\nassistant: "
        turns.append({"role": "user", "content": assistant_response})

    # Remove the trailing unpaired user turn
    if turns[-1]["role"] == "user":
        turns.pop()

    return turns


def generate_dataset_iter(
    model: HookedTransformer,
    seeds: list[str] = None,
    n_turns: int = 6,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    domain_hint: str = None,
):
    """
    Generator variant — yields one conversation at a time so the pipeline
    can start extraction and TDA on conv N while the GPU generates conv N+1.
    """
    seeds = seeds or SEED_PROMPTS
    for i, seed in enumerate(seeds):
        print(f"  Generating conversation {i+1}/{len(seeds)}: {seed[:50]}...", flush=True)
        turns = self_dialogue(model, seed, n_turns=n_turns,
                              max_new_tokens=max_new_tokens, temperature=temperature)
        yield {
            "id": f"gen_{i:04d}",
            "turns": turns,
            "domain_hint": domain_hint,
        }


def generate_dataset(
    model: HookedTransformer,
    seeds: list[str] = None,
    n_turns: int = 6,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    domain_hint: str = None,
) -> list[dict]:
    return list(generate_dataset_iter(
        model, seeds=seeds, n_turns=n_turns,
        max_new_tokens=max_new_tokens, temperature=temperature,
        domain_hint=domain_hint,
    ))
