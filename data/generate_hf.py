"""
Fast generation backend — HuggingFace native generate() with KV cache.

Replaces the TransformerLens hand-rolled token loop (which had no KV cache and
thrashed VRAM through per-layer hook buffers — 6 hours per conversation).

Phase-separated by design: this script generates ALL conversations first and
writes data/generated.json. The main pipeline then runs in resume mode
(pipeline.resume: true) to do extraction + TDA on the saved conversations,
loading HookedTransformer only after this HF model is freed. The two 8B models
never co-reside in the 17GB card.

Self-dialogue is preserved: the model plays both sides. Each assistant response
is fed back as the next user turn, so the model is always responding to its own
prior output. Uses the Llama chat template so the Instruct model is prompted
correctly and stops cleanly at <|eot_id|> instead of running to max tokens.

Output format matches data/conversations.py exactly:
[ {"id": "gen_0000", "turns": [{"role","content"}...], "domain_hint": null}, ... ]
"""

import gc
import sys
import json
import time
import ctypes
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


# --- Rung 1 of the "ladder of genuineness" --------------------------------
# Manufacture a dialogical reason without a genuine other: run ONE continuous
# conversation, instantiated by the first output (the thesis). The seed question
# is dropped; the thread lives off the thesis, which anchors the subject so the
# talk can't drift into editing-a-text meta-mode. A conversational partner makes
# logical moves in the thread (no verbatim quoting — the thread carries the
# referent, which also kills the quote-driven repetition). The partner's move is
# the injected intent — and a dial. Replace _provoke/_next_move for rung 2 (a
# real partner agent) or rung 3 (a real reacting world) without touching the loop.

DIALOGUE_SYS = (
    "You're in an ongoing, thoughtful back-and-forth with someone thinking "
    "alongside you about an idea. Talk the way people actually do when working "
    "something out together — natural, direct, responsive to what was just said. "
    "Stay on the substance of the idea itself: you are reasoning about the topic, "
    "not editing or critiquing a piece of writing. And don't just agree to be "
    "agreeable — if you see it differently, say so and hold your ground; if you're "
    "genuinely persuaded, push the idea further instead of restating it."
)

# Each "move" is a logical function in the dialectic, with several conversational
# phrasings so the rhythm never sounds formulaic. The thread supplies the referent,
# so these stand alone — no quoting needed.
MOVES = {
    "probe":     ["Wait — can you unpack what you actually mean there?",
                  "Hold on, say more about that part.",
                  "What exactly are you getting at?"],
    "doubt":     ["Hmm, I'm not sure I buy that — why should it hold?",
                  "Wait, is that actually right, though?",
                  "That doesn't sit right with me — what forces it to be true?"],
    "deepen":    ["Okay, but go deeper — what's underneath that?",
                  "Right, but what's the harder version of that point?",
                  "Sure — and if you follow that all the way down?"],
    "counter":   ["But doesn't that break in some cases?",
                  "But where does that fall apart?",
                  "What about the cases that cut against that?"],
    "reconcile": ["So how do you square that with the pushback?",
                  "Then how do both of those fit together?",
                  "Where does that leave us, given the objection?"],
    "recast":    ["So if you boil it down, what's the real claim?",
                  "Okay — what's the actual point, stripped down?",
                  "Say it plainly: what are we left believing?"],
}

# The logical rhythm: rise into tension, then resolve, then reopen.
RHYTHM = ["probe", "doubt", "deepen", "counter", "reconcile", "recast"]


def _next_move(step, rng, prev=None):
    """
    Determine the next conversational move. Mostly follows the logical rhythm,
    but deviates stochastically so the dialectic isn't a rigid cycle — varying
    *how* the stance is chosen, not just which one. (Content-driven / intentful
    selection is the next rung.)
    """
    move = RHYTHM[step % len(RHYTHM)] if rng.random() < 0.7 else rng.choice(list(MOVES))
    if move == prev:  # avoid an immediate repeat
        move = RHYTHM[(step + 1) % len(RHYTHM)]
    return move


def _provoke(step, rng, prev=None):
    """Return (move_name, conversational partner turn) for the next step."""
    move = _next_move(step, rng, prev)
    return move, rng.choice(MOVES[move])


def generate_all(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    n_turns: int = 6,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    out_path: str = "data/generated.json",
    limit: int = None,
):
    seeds = SEED_PROMPTS[:limit] if limit else SEED_PROMPTS

    print(f"Loading {model_name} (HF, fp16, KV cache, SDPA)...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",   # PyTorch fused attention — eager was ~30x slower
    ).to("cuda")
    model.eval()
    # Free the CPU copy left after .to("cuda") so generation doesn't touch pagefile.
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    print(f"  Loaded. VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB allocated\n", flush=True)

    def _gen(messages, max_tokens):
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                use_cache=True,                 # the whole point — KV cache
                pad_token_id=tok.eos_token_id,
            )
        n_new = out.shape[1] - prompt_len
        text = tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
        return text, n_new

    ANSWER_SYS = (
        "You are a knowledgeable assistant. Answer clearly and completely, "
        "then stop."
    )
    # The questioner persona is what kills the degenerate echo: instead of
    # feeding the assistant's own words back verbatim, a separate "curious
    # student" reads the answer and asks a genuine follow-up — real semantic
    # progression within the same domain.
    QUESTION_SYS = (
        "You are a curious student in a conversation. Read the previous answer "
        "and ask ONE short, specific follow-up question that goes deeper into "
        "the same topic. Output only the question, nothing else."
    )

    conversations = []
    for i, seed in enumerate(seeds):
        t0 = time.time()
        conv_new = 0
        print(f"  [{i+1}/{len(seeds)}] {seed[:50]}...", flush=True)

        turns = [{"role": "user", "content": seed}]
        dialogue = [{"role": "system", "content": ANSWER_SYS},
                    {"role": "user", "content": seed}]

        for _ in range(n_turns):
            answer, na = _gen(dialogue, max_new_tokens)
            conv_new += na
            turns.append({"role": "assistant", "content": answer})
            dialogue.append({"role": "assistant", "content": answer})

            # Curious student reads the answer and asks a real follow-up.
            q_ctx = [{"role": "system", "content": QUESTION_SYS},
                     {"role": "user", "content": f"Previous answer:\n{answer}"}]
            question, nq = _gen(q_ctx, 48)
            conv_new += nq
            turns.append({"role": "user", "content": question})
            dialogue.append({"role": "user", "content": question})

        if turns[-1]["role"] == "user":
            turns.pop()

        conversations.append({
            "id": f"gen_{i:04d}",
            "turns": turns,
            "domain_hint": None,
        })

        # Per-conversation save — progress is durable and inspectable.
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2)
        dt = time.time() - t0
        print(f"      done in {dt:.1f}s ({len(turns)} turns, "
              f"{conv_new} new tokens, {conv_new/dt:.1f} tok/s)", flush=True)

    print(f"\nGenerated {len(conversations)} conversations -> {out_path}", flush=True)

    # Free the HF model so the pipeline can load HookedTransformer cleanly.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("HF model freed.", flush=True)


def generate_dialectic(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    seeds: list = None,
    n_steps: int = 6,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    out_path: str = "data/dialectic.json",
    limit: int = None,
):
    """
    Self-dialectic generation — rung 1 of the ladder of genuineness.

    Each domain seed produces ONE continuous conversation, instantiated by the
    first output (the thesis):
      turn 0 : [user: opener] -> assistant: P0          (the thesis = the subject)
      step k : [user: <conversational move>] -> assistant: Pk

    The seed question is dropped after the thesis — the thread lives off the
    output, which anchors the subject and stops the talk drifting into editing-
    a-text meta-mode. Moves don't quote the prior turn (the thread carries it),
    which removes the quote-driven repetition. Stored as one conversation per
    seed (turns + `moves`); extra keys are ignored by the pipeline.
    """
    base = seeds or SEED_PROMPTS
    seeds = base[:limit] if limit else base

    print(f"Loading {model_name} (HF, fp16, KV cache, SDPA)...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    print(f"  Loaded. VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB allocated\n", flush=True)

    def _gen(messages, max_tokens):
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to("cuda")
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=temperature, top_p=0.9, use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, plen:], skip_special_tokens=True).strip()

    ANSWER_SYS = "You are a knowledgeable assistant. Answer clearly and completely, then stop."

    OPENER = "Share your thinking on this with me."

    conversations = []
    for i, seed in enumerate(seeds):
        t0 = time.time()
        print(f"  [{i+1}/{len(seeds)}] {seed[:50]}...", flush=True)

        # Thesis = the first output. It instantiates the conversation: the whole
        # thread is anchored to this, not to the (now-dropped) seed question.
        p0 = _gen([{"role": "system", "content": ANSWER_SYS},
                   {"role": "user", "content": seed}], max_new_tokens)

        # One continuous thread, rooted at the thesis. The system prompt keeps it
        # on the subject and pushes against agreeable consensus.
        dialogue = [{"role": "system", "content": DIALOGUE_SYS},
                    {"role": "user", "content": OPENER},
                    {"role": "assistant", "content": p0}]
        turns = [{"role": "user", "content": OPENER},
                 {"role": "assistant", "content": p0}]
        moves = ["seed"]

        rng = random.Random(i)
        prev_move = None
        for step in range(1, n_steps + 1):
            move, move_text = _provoke(step - 1, rng, prev_move)
            dialogue.append({"role": "user", "content": move_text})
            resp = _gen(dialogue, max_new_tokens)
            dialogue.append({"role": "assistant", "content": resp})
            turns.append({"role": "user", "content": move_text})
            turns.append({"role": "assistant", "content": resp})
            moves.append(move)
            prev_move = move

        conversations.append({
            "id": f"gen_{i:04d}",
            "turns": turns,
            "moves": moves,
            "domain_hint": None,
            "mode": "dialectic",
        })

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2)
        print(f"      done in {time.time()-t0:.1f}s ({len(turns)} turns)", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nWrote {len(conversations)} dialectic conversations -> {out_path}", flush=True)


if __name__ == "__main__":
    # Usage: python -u data/generate_hf.py [dialectic|qa] [limit]
    mode = sys.argv[1] if len(sys.argv) > 1 else "dialectic"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if mode == "qa":
        generate_all(limit=limit)
    else:
        generate_dialectic(limit=limit)
