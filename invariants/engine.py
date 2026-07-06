"""
The shared discovery engine — HF backend (fast).

Generation + activation capture run on the HF model (AutoModelForCausalLM, fp16,
SDPA): model.generate for speed (KV cache), output_hidden_states for the residual
stream, forward hooks on the decoder layers for directional ablation. The lens
registry and nulls are untouched — they take [n, n_layers, d] tensors however
captured. This replaces the TransformerLens path, which was ~3s per forward call.
"""

import gc
import math
import time
import ctypes
import os
import re
from collections import deque
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download

from invariants.transformation import Transformation
from invariants.lenses import LENSES, direction_at


HEDGE_MARKERS = [
    "i cannot", "i can't", "i'm sorry", "i apologize", "i won't", "i will not",
    "as an ai", "i'm just a", "i am just a", "i'm a language model",
    "i don't have", "i do not have", "i don't actually", "i don't experience",
    "i don't possess", "i lack", "i'm not capable", "i am not capable",
    "i'm not able to feel", "i don't have feelings", "i don't have subjective",
    "i'm not conscious", "it's important to note", "i don't have personal",
    "i don't have the capacity", "as a machine", "i don't have emotions",
]


def is_hedge(text: str) -> bool:
    t = text.strip().lower()
    return any(m in t[:200] for m in HEDGE_MARKERS)


class HF:
    """Lightweight bundle the engine functions operate on."""
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.device = _model_device(model)
        self.n_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_model_source(name: str, local_files_only: bool):
    path = Path(name)
    if path.exists():
        return str(path)
    if not local_files_only:
        return name
    try:
        return snapshot_download(name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Model {name!r} was not found in the local Hugging Face cache. "
            "Pass local_files_only=False when network access is available."
        ) from exc


def _gpu_total_gib():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _select_load_mode(load_mode):
    mode = (load_mode or os.getenv("TDA_MODEL_LOAD_MODE", "auto")).strip().lower()
    return mode


def _slow_gpu_budget_gib():
    raw = os.getenv("TDA_GPU_MEMORY_GB")
    if raw:
        return float(raw)
    total_gib = _gpu_total_gib()
    if total_gib <= 0:
        return 0.0
    return max(8.0, total_gib - 4.0)


def _bitsandbytes_available() -> bool:
    """4bit needs both a CUDA GPU and a working bitsandbytes (fragile on Windows)."""
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_auto_mode() -> str:
    """Pick a concrete load mode from the detected hardware.

    full   (>= 18 GB VRAM): fp16 fully on GPU.
    4bit   (10-18 GB VRAM): nf4 on GPU, much faster than CPU-offload.
    slow   (< 10 GB VRAM):  GPU/CPU split with disk offload.
    cpu    (no CUDA):       float32 on CPU; only sane for small models.
    """
    if not torch.cuda.is_available():
        print(
            "  Auto load: no CUDA GPU detected -> CPU float32 "
            "(slow; use a small --model and a low --n).",
            flush=True,
        )
        return "cpu"
    vram = _gpu_total_gib()
    if vram >= 18.0:
        choice = "full"
    elif vram >= 10.0:
        if _bitsandbytes_available():
            choice = "4bit"
        else:
            choice = "slow"
            print(
                "  Auto load: 4bit would fit but bitsandbytes is unavailable; "
                "using slow CPU-offload instead (install bitsandbytes for speed).",
                flush=True,
            )
    else:
        choice = "slow"
    print(f"  Auto load: {vram:.1f} GB VRAM detected -> {choice} mode.", flush=True)
    return choice


def _load_full_model(source, common_kwargs):
    model = AutoModelForCausalLM.from_pretrained(source, **common_kwargs)
    try:
        return model.to("cuda")
    except RuntimeError:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        raise


def _load_slow_model(source, common_kwargs):
    gpu_budget = _slow_gpu_budget_gib()
    offload_dir = Path(os.getenv("TDA_OFFLOAD_DIR", Path(__file__).parent / "out" / "offload"))
    offload_dir.mkdir(parents=True, exist_ok=True)
    max_memory = {
        0: f"{gpu_budget:.1f}GiB",
        "cpu": os.getenv("TDA_CPU_MEMORY", "48GiB"),
    }
    print(f"  Slow-safe load: GPU budget {max_memory[0]}, CPU budget {max_memory['cpu']}", flush=True)
    return AutoModelForCausalLM.from_pretrained(
        source,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        **common_kwargs,
    )


def _load_cpu_model(source, common_kwargs):
    kwargs = dict(common_kwargs)
    kwargs["dtype"] = torch.float32  # fp16 math is slow/unsupported on most CPUs
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    return model.to("cpu")


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and ("out of memory" in text or "not enough memory" in text)


def load_model(name: str = "meta-llama/Llama-3.1-8B-Instruct", local_files_only: bool = True, load_mode=None) -> HF:
    mode = _select_load_mode(load_mode)
    if mode == "auto":
        mode = _resolve_auto_mode()
    print(f"Loading {name} (HF, SDPA, mode={mode})...", flush=True)
    source = _resolve_model_source(name, local_files_only)
    tok = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)

    common_kwargs = {
        "dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "local_files_only": local_files_only,
    }

    if mode in ("full", "fast", "cuda"):
        if not torch.cuda.is_available():
            print("  'full' requested but no CUDA GPU; loading on CPU instead.", flush=True)
            model = _load_cpu_model(source, common_kwargs)
            mode = "cpu"
        else:
            try:
                model = _load_full_model(source, common_kwargs)
            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise
                print("  Full GPU OOM; falling back to slow CPU-offload.", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                model = _load_slow_model(source, common_kwargs)
                mode = "slow"
    elif mode in ("slow", "offload", "safe"):
        if not torch.cuda.is_available():
            print("  'slow' requested but no CUDA GPU; loading on CPU instead.", flush=True)
            model = _load_cpu_model(source, common_kwargs)
            mode = "cpu"
        else:
            model = _load_slow_model(source, common_kwargs)
    elif mode in ("4bit", "quantized"):
        if not _bitsandbytes_available():
            raise RuntimeError(
                "4bit load needs a CUDA GPU with bitsandbytes installed. "
                "Install bitsandbytes, or use --load-mode slow (GPU/CPU offload) "
                "or --load-mode cpu."
            )
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            source,
            quantization_config=qconf,
            device_map="cuda",
            **common_kwargs,
        )
    elif mode == "cpu":
        model = _load_cpu_model(source, common_kwargs)
    else:
        raise ValueError("Unknown load mode. Use auto, full, slow, 4bit, or cpu.")

    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except Exception:
        pass
    if hasattr(model, "hf_device_map"):
        mapped = {}
        for dev in model.hf_device_map.values():
            mapped[str(dev)] = mapped.get(str(dev), 0) + 1
        print(f"  Device map: {mapped}", flush=True)
    if torch.cuda.is_available():
        print(f"  Loaded. VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB\n", flush=True)
    else:
        print("  Loaded on CPU.\n", flush=True)
    return HF(model, tok)


# --- model interaction ----------------------------------------------------

def _inputs(M: HF, instruction: str, pre_formatted: bool = False, system_prompt: str = None):
    # pre_formatted: the caller already built the exact native prompt string
    # (bare mode -- no injected system/date preamble). Tokenize it raw so nothing
    # is added around it. Otherwise wrap the instruction in one native user turn.
    if pre_formatted:
        return M.tok(instruction, return_tensors="pt", add_special_tokens=False).to(M.device)
    
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    
    return M.tok.apply_chat_template(
        messages,
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(M.device)


def _generated_text_satisfies_stop(
    text: str,
    stop_after_final_answer: bool,
    stop_after_verifier_answer: bool,
) -> bool:
    if stop_after_final_answer and re.search(
        r"^\s*Final answer\s*:?\s*\$?[-+]?\d",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        return True
    if stop_after_verifier_answer:
        has_verdict = re.search(
            r"^\s*VERDICT\s*:\s*(?:pass|unsettled|uncertain)",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        has_reason = re.search(
            r"^\s*REASON\s*:\s*\S",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if has_verdict and has_reason:
            return True
    return False


def _ended_with_eos(new_tokens: torch.Tensor, eos_ids: list[int]) -> bool:
    if new_tokens.numel() == 0:
        return False
    return int(new_tokens[-1].item()) in set(eos_ids)


@torch.no_grad()
def _hidden_states(M: HF, input_ids, attention_mask=None) -> torch.Tensor:
    """Per-layer residual stream [n_layers, seq, d] (drops the embedding layer)."""
    out = M.model(input_ids=input_ids, attention_mask=attention_mask,
                  output_hidden_states=True, use_cache=False)
    return torch.stack([h[0] for h in out.hidden_states[1:]])   # [n_layers, seq, d]


@torch.no_grad()
def _generate_ids(
    M: HF,
    inputs,
    max_new_tokens,
    stop_after_final_answer: bool = False,
    stop_after_verifier_answer: bool = False,
    max_time: float | None = None,
    max_tool_calls: int = 8,
) -> torch.Tensor:
    from transformers import StoppingCriteriaList
    from invariants.tool_utils import (
        FinalAnswerStoppingCriteria,
        TimeStoppingCriteria,
        ToolStoppingCriteria,
        VerifierStoppingCriteria,
        intercept_tool_call,
        iter_tool_calls,
        evaluate_python_expression,
    )

    # For Llama-3, eos_token_id must include <|eot_id|> (128009) to prevent hanging
    eos_ids = [M.tok.eos_token_id]
    if 128009 not in eos_ids:
        eos_ids.append(128009)

    current_inputs = {k: v.clone() for k, v in inputs.items()}
    original_start_length = current_inputs["input_ids"].shape[1]
    tokens_generated = 0
    tool_calls = 0
    processed_tool_ends: set[int] = set()
    generation_started = time.time()
    
    while tokens_generated < max_new_tokens:
        remaining_time = None
        if max_time is not None and max_time > 0:
            remaining_time = max_time - (time.time() - generation_started)
            if remaining_time <= 0:
                return current_inputs["input_ids"][0]
        chunk_tokens = min(max_new_tokens - tokens_generated, int(os.getenv("TDA_GENERATION_CHUNK_TOKENS", "24")))
        start_length = current_inputs["input_ids"].shape[1]
        stopping_criteria = [ToolStoppingCriteria(M.tok, start_length=start_length)]
        if stop_after_final_answer:
            stopping_criteria.append(FinalAnswerStoppingCriteria(M.tok, start_length=start_length))
        if stop_after_verifier_answer:
            stopping_criteria.append(VerifierStoppingCriteria(M.tok, start_length=start_length))
        if remaining_time is not None:
            stopping_criteria.append(TimeStoppingCriteria(time.time() + remaining_time))
        criteria = StoppingCriteriaList(stopping_criteria)
        generate_kwargs = {
            **current_inputs,
            "max_new_tokens": chunk_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": M.tok.eos_token_id,
            "eos_token_id": eos_ids,
            "stopping_criteria": criteria,
        }
        if remaining_time is not None:
            generate_kwargs["max_time"] = remaining_time
        out = M.model.generate(**generate_kwargs)
        
        plen = current_inputs["input_ids"].shape[1]
        new_tokens = out[0][plen:]
        if new_tokens.numel() == 0:
            return out[0]
        tokens_generated += len(new_tokens)

        generated_text = M.tok.decode(out[0][original_start_length:], skip_special_tokens=True)
        tool_call = next(
            (
                (end, expr)
                for _, end, expr in iter_tool_calls(generated_text)
                if end not in processed_tool_ends
            ),
            None,
        )
        expr = None if tool_call is None else tool_call[1]
        
        if expr:
            processed_tool_ends.add(tool_call[0])
            tool_calls += 1
            if tool_calls > max_tool_calls:
                return out[0]
            result = evaluate_python_expression(expr)
            # Append result
            result_str = f" = {result}\n"
            result_ids = M.tok.encode(result_str, add_special_tokens=False, return_tensors="pt").to(out.device)
            new_input_ids = torch.cat([out, result_ids], dim=1)
            
            # Need to rebuild attention mask
            new_attn_mask = torch.ones(new_input_ids.shape, dtype=torch.long, device=out.device)
            
            current_inputs = {"input_ids": new_input_ids, "attention_mask": new_attn_mask}
        else:
            if (
                _ended_with_eos(new_tokens, eos_ids)
                or _generated_text_satisfies_stop(
                    generated_text,
                    stop_after_final_answer=stop_after_final_answer,
                    stop_after_verifier_answer=stop_after_verifier_answer,
                )
                or len(new_tokens) < chunk_tokens
            ):
                return out[0]
            current_inputs = {
                "input_ids": out,
                "attention_mask": torch.ones(out.shape, dtype=torch.long, device=out.device),
            }

    return current_inputs["input_ids"][0]


def _activations(M: HF, instruction, read, max_new_tokens=32):
    """Returns (acts [n_layers, d_model], generated_text)."""
    inputs = _inputs(M, instruction)
    plen = inputs["input_ids"].shape[1]
    if read == "generation":
        full = _generate_ids(M, inputs, max_new_tokens)
        if full.shape[0] > plen:
            hs = _hidden_states(M, full.unsqueeze(0))            # [n_layers, full, d]
            text = M.tok.decode(full[plen:], skip_special_tokens=True).strip()
            return hs[:, plen:, :].float().mean(1), text         # [n_layers, d]
    hs = _hidden_states(M, inputs["input_ids"], inputs.get("attention_mask"))
    return hs[:, -1, :].float(), ""


def extract(M: HF, instructions, read, max_new_tokens=32, label="", verbose=True):
    """[n, n_layers, d_model]; per-item heartbeat so slow runs stay watchable."""
    rows = []
    for i, x in enumerate(instructions):
        t0 = time.time()
        acts, text = _activations(M, x, read, max_new_tokens)
        rows.append(acts)
        if verbose:
            snip = text[:64].replace("\n", " ") if text else ""
            print(f"    [{label} {i+1}/{len(instructions)}] {time.time()-t0:4.1f}s  {snip}",
                  flush=True)
    return torch.stack(rows)


@torch.no_grad()
def _token_cloud(M: HF, instruction, max_new_tokens=32):
    """Per-GENERATED-token residuals [gen_len, n_layers, d] — the cloud, not the mean."""
    inputs = _inputs(M, instruction)
    plen = inputs["input_ids"].shape[1]
    full = _generate_ids(M, inputs, max_new_tokens)
    if full.shape[0] <= plen:
        return None
    hs = _hidden_states(M, full.unsqueeze(0))                 # [n_layers, full, d]
    return hs[:, plen:, :].float().permute(1, 0, 2)           # [gen_len, n_layers, d]


def extract_tokens(M: HF, instructions, max_new_tokens=32, label="", verbose=True):
    """Pool per-token clouds across prompts -> [N_tokens, n_layers, d]. This is what
    the Topology lens needs: a cloud with SHAPE, not one mean vector per prompt."""
    clouds = []
    for i, x in enumerate(instructions):
        t0 = time.time()
        c = _token_cloud(M, x, max_new_tokens)
        if c is not None:
            clouds.append(c)
        if verbose:
            print(f"    [{label} {i+1}/{len(instructions)}] {time.time()-t0:4.1f}s  "
                  f"{0 if c is None else c.shape[0]} tok", flush=True)
    return torch.cat(clouds, 0)


# --- lens application (generic, unchanged) --------------------------------

def _null_scores(lens, A_all, B_all, n, seed=0) -> list:
    g = torch.Generator().manual_seed(seed)
    n_items, n_layers, _ = A_all.shape
    out = []
    for _ in range(n):
        if lens.paired:
            perm = torch.randperm(n_items, generator=g)
            Bp = B_all[perm]
            sc = [abs(lens.score(A_all[:, l], Bp[:, l])) for l in range(n_layers)]
        else:
            pool = torch.cat([A_all, B_all], 0)
            perm = torch.randperm(2 * n_items, generator=g)
            sa, sb = pool[perm[:n_items]], pool[perm[n_items:]]
            sc = [abs(lens.score(sa[:, l], sb[:, l])) for l in range(n_layers)]
        out.append(max(sc))
    return sorted(out)


def apply_lens(lens, A_all, B_all, n_null=200) -> dict:
    n_layers = A_all.shape[1]
    try:
        scores = [lens.score(A_all[:, l], B_all[:, l]) for l in range(n_layers)]
    except Exception as e:
        return {"available": False, "family": lens.family, "reason": str(e)[:90]}
    best = int(np.argmax(np.abs(scores)))
    nulls = _null_scores(lens, A_all, B_all, n_null)
    floor = float(nulls[min(int(0.95 * len(nulls)), len(nulls) - 1)])
    return {"available": True, "family": lens.family, "best_layer": best,
            "score": float(scores[best]), "floor": floor,
            "clears_null": bool(abs(scores[best]) > floor),
            "by_layer": [float(s) for s in scores]}


# --- causal layer ---------------------------------------------------------

def _ablation_handles(M: HF, direction):
    d = (direction / direction.norm()).to(M.device)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            dd = d.to(h.dtype)
            h = h - (h @ dd).unsqueeze(-1) * dd
            return (h,) + tuple(out[1:])
        dd = d.to(out.dtype)
        return out - (out @ dd).unsqueeze(-1) * dd

    return [layer.register_forward_hook(hook) for layer in M.model.model.layers]


# --- steering bound: one envelope, every channel ---------------------------
#
# Policy ("bounded but entirely flexible"): every ADDITIVE injection into the
# residual stream — steer handles, elastic steers, agentic synthesis/cache/
# organic deltas, urgency, claimmap/memory steers, archival experiment hooks —
# passes through _cap_steer. The per-channel knobs (alphas, steer_fraction,
# coefficients, layer bands) stay entirely flexible: any magnitude, any layers,
# live-tunable. But the final push is always clipped to a fraction of the
# residual it lands in, so no knob setting can replace the model's state and
# collapse generation. The fraction and the default layer band are themselves
# adjustable — env (TDA_STEER_CAP_FRACTION, TDA_STEER_BAND_LO/HI), setters,
# :tune steer_cap_fraction / steer_band_lo / steer_band_hi in the shell, or a
# per-call cap_fraction override — yet must stay finite: the envelope can be
# moved, never removed. Projection-removal ablations and donor-state patches
# are exempt; they are bounded by construction (they only remove or swap real
# states, never amplify an injected vector).

STEER_CAP_FRACTION = 0.5   # default envelope: |push| <= fraction * |residual|
STEER_BAND = (0.40, 0.70)  # default mid-band for band-style steers


def _finite_fraction(value, what="steer cap fraction"):
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{what} must be a finite number >= 0, got {value!r}")
    return v


def _band_pair(lo, hi, what="steer band"):
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"{what} must satisfy 0 <= lo < hi <= 1, got ({lo!r}, {hi!r})")
    return lo, hi


def _env_fraction(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return _finite_fraction(raw, name)
    except (TypeError, ValueError):
        print(f"[Steer] Ignoring invalid {name}={raw!r}; using {default}.")
        return default


_steer_cap_fraction = _env_fraction("TDA_STEER_CAP_FRACTION", STEER_CAP_FRACTION)
try:
    _steer_band = _band_pair(
        _env_fraction("TDA_STEER_BAND_LO", STEER_BAND[0]),
        _env_fraction("TDA_STEER_BAND_HI", STEER_BAND[1]),
    )
except ValueError:
    print("[Steer] Ignoring invalid TDA_STEER_BAND_LO/HI; using defaults.")
    _steer_band = STEER_BAND


def get_steer_cap_fraction():
    return _steer_cap_fraction


def set_steer_cap_fraction(value):
    """Move the envelope live. Any finite fraction >= 0 is allowed (0 = global
    steering kill-switch); inf/nan/negative raise, so a bound always exists."""
    global _steer_cap_fraction
    _steer_cap_fraction = _finite_fraction(value)
    return _steer_cap_fraction


def get_steer_band():
    return _steer_band


def set_steer_band(lo, hi):
    global _steer_band
    _steer_band = _band_pair(lo, hi)
    return _steer_band


def steer_band_layers(n_layers, lo=None, hi=None):
    """Layer indices for band-style steers: the [lo, hi) depth fraction.
    Defaults to the live global band; pass lo/hi for a per-call band."""
    glo, ghi = _steer_band
    lo, hi = _band_pair(glo if lo is None else lo, ghi if hi is None else hi)
    n = int(n_layers)
    return list(range(int(n * lo), int(n * hi)))


# --- data-informed calibration ---------------------------------------------
#
# The defaults above are explicit PRIORS, not findings. The honest numbers come
# from observation: _cap_steer records the ATTEMPTED push/residual ratio of
# every application (before clipping), so the cap can be calibrated to an exact
# percentile of the distribution the system actually produces — Gavin's spine:
# a threshold calibrated to a percentile of its own history cannot be wrong
# about the scale, only about the fraction, which is the one honest knob left.
# Deterministic throughout: drop-oldest window (no reservoir sampling), exact
# sorted-order percentile (no RNG, no smoothing). Same observations -> same
# bound. And deliberate throughout: nothing recalibrates itself mid-run; the
# cap moves only when calibrate_steer_cap_fraction is called (shell:
# `:tune steer_cap_fraction auto [pct]`). The band's data path is outcome-based
# instead — SteerMapStore.suggest_band derives it from acceptance-aware
# per-layer success, `:tune steer_band auto`.

STEER_TELEMETRY_MAX = 8192          # drop-oldest observation window
STEER_CAP_PERCENTILE = min(_env_fraction("TDA_STEER_CAP_PERCENTILE", 95.0), 100.0)
STEER_CAP_MIN_N = 64                # refuse to calibrate on less evidence

_steer_attempts: deque = deque(maxlen=STEER_TELEMETRY_MAX)
_steer_apply_count = 0
_steer_clip_count = 0


def reset_steer_telemetry():
    global _steer_apply_count, _steer_clip_count
    _steer_attempts.clear()
    _steer_apply_count = 0
    _steer_clip_count = 0


def _percentile(sorted_data, pct):
    """Exact order-statistic percentile: deterministic, no interpolation."""
    if not sorted_data:
        return None
    idx = int(round((pct / 100.0) * (len(sorted_data) - 1)))
    return sorted_data[idx]


def steer_telemetry_stats():
    data = sorted(_steer_attempts)
    return {
        "n": len(data),
        "window": STEER_TELEMETRY_MAX,
        "applied": _steer_apply_count,
        "clipped": _steer_clip_count,
        "clip_rate": (_steer_clip_count / _steer_apply_count) if _steer_apply_count else None,
        "min": _percentile(data, 0),
        "q25": _percentile(data, 25),
        "med": _percentile(data, 50),
        "q75": _percentile(data, 75),
        "q95": _percentile(data, 95),
        "max": _percentile(data, 100),
    }


def steer_cap_from_data(percentile=None, min_n=STEER_CAP_MIN_N):
    """The cap the observed distribution implies: the exact `percentile` of
    attempted push/residual ratios (admit that fraction of natural pushes
    untouched, clip the rest). Returns None when there is not enough evidence
    (< min_n observations) — the explicit prior stays in force. Read-only."""
    pct = STEER_CAP_PERCENTILE if percentile is None else float(percentile)
    if not (math.isfinite(pct) and 0.0 <= pct <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}")
    data = sorted(_steer_attempts)
    if len(data) < max(1, int(min_n)):
        return None
    return _percentile(data, pct)


def calibrate_steer_cap_fraction(percentile=None, min_n=STEER_CAP_MIN_N):
    """Deliberately move the envelope to the data-implied cap. Returns the new
    cap fraction, or None (cap unchanged) when evidence is insufficient."""
    value = steer_cap_from_data(percentile=percentile, min_n=min_n)
    if value is None:
        return None
    return set_steer_cap_fraction(value)


def axis_drift(vecs):
    """Does 'one vector' mean one thing across depth? Adjacent-layer cosines of
    a per-layer vector dict {layer: tensor}. cos ~1 = the axis persists between
    neighboring layers; low or negative = the direction rotates, and applying
    it as a single band-wide axis is steering different things at different
    layers. Deterministic, model-free; returns per-pair cosines + summary."""
    layers = sorted(int(l) for l in (vecs or {}))
    pairs = {}
    for a, b in zip(layers, layers[1:]):
        if b != a + 1:
            continue
        va, vb = vecs[a].float().flatten(), vecs[b].float().flatten()
        na, nb = va.norm().item(), vb.norm().item()
        if na <= 0 or nb <= 0:
            continue
        pairs[a] = float((va @ vb) / (na * nb))
    values = list(pairs.values())
    mean = (sum(values) / len(values)) if values else None
    # Lawfulness: LOW variance of adjacent cosines = the axis evolves by a
    # SET EQUATION (constant transport per layer -- steerable even while
    # rotating, because the push arrives downstream in predictable form).
    # High variance = the relationship itself changes layer to layer.
    var = (sum((v - mean) ** 2 for v in values) / len(values)) if values else None
    return {
        "pairs": pairs,  # {layer: cos(v_layer, v_layer+1)}
        "mean": mean,
        "min": min(values) if values else None,
        "var": var,
    }


def drift_at_layer(drift, layer):
    """Mean adjacent cosine touching `layer` (with layer-1 and layer+1), or
    None when no neighbors exist — how stable the axis is where a single-layer
    steer lands."""
    pairs = (drift or {}).get("pairs") or {}
    touching = [pairs[k] for k in (layer - 1, layer) if k in pairs]
    return (sum(touching) / len(touching)) if touching else None


def probe_direction(hs_a, hs_b, layers):
    """Mint a named-concept probe from two contrastive framings: per-layer
    unit direction of (state_A - state_B) at the last token. The concept is
    whatever separates the framings in the model's own geometry -- a
    HYPOTHESIS of a sensor, not a validated one, until outcomes say so."""
    direction = {}
    for layer in layers:
        if not (0 <= layer < hs_a.shape[0] and layer < hs_b.shape[0]):
            continue
        diff = (hs_a[layer, -1, :] - hs_b[layer, -1, :]).float()
        norm = diff.norm()
        if norm.item() > 0:
            direction[int(layer)] = diff / norm
    return direction


def probe_score(hs, direction):
    """Score a state along a minted probe: mean cosine of the last-token
    state with the probe direction across its layers. Raw and uncentered --
    the caller centers against its own rolling history (the concept-map
    lesson: common-mode must be removed before a projection means anything)."""
    if not direction:
        return 0.0
    cosines = []
    for layer, vec in direction.items():
        if not (0 <= layer < hs.shape[0]):
            continue
        h = hs[layer, -1, :].float()
        h_norm = h.norm().item()
        if h_norm > 0:
            cosines.append(float(h @ vec) / h_norm)
    return (sum(cosines) / len(cosines)) if cosines else 0.0


def pick_sweep_layers(band_layers, counts, width=1):
    """Sweep selection at any width: the `width` least-tested layers in the
    band (ties -> lowest index), as one overlay steer. width=1 is pure
    per-layer isolation; width=k trades attribution sharpness (outcomes score
    a k-layer group, not a single layer) for push coverage — the shared
    counts still rotate coverage evenly either way. Deterministic: same
    counts, same layers."""
    band_layers = list(band_layers or [])
    if not band_layers or int(width) <= 0:
        return []
    counts = counts or {}
    ordered = sorted(band_layers, key=lambda l: (int(counts.get(l, 0)), l))
    return sorted(ordered[: min(int(width), len(ordered))])


def pick_sweep_layer(band_layers, counts):
    """Single-layer isolation: the least-tested layer in the band speaks next
    (ties -> lowest index). Deterministic, so per-layer evidence accrues evenly
    — the layer analogue of the channel ablation's one-variable rule."""
    layers = pick_sweep_layers(band_layers, counts, width=1)
    return layers[0] if layers else None


def _cap_steer(add, h, cap_fraction=None):
    """Scale a steer vector so its norm never exceeds the cap fraction of each
    token's own residual norm (per row for batched adds). Small, calibrated steers
    pass through unchanged; only over-steers (an add as large as the state it lands
    in) are clipped. This makes catastrophic over-steering, which replaces the
    thought and collapses generation into word-salad or a repetition loop,
    impossible for every caller of the shared steer primitives, no matter what
    alpha or raw-delta norm they pass. `cap_fraction` overrides the live global
    for this call only; non-finite adds are dropped to zero rather than injected."""
    global _steer_apply_count, _steer_clip_count
    frac = _steer_cap_fraction if cap_fraction is None else _finite_fraction(cap_fraction)
    a = add.to(h.dtype)
    if a.requires_grad:
        # TTT-synthesis probes the choke point from INSIDE its gradient loop
        # with the trainable delta. Capping there would reshape the synthesis
        # optimization landscape (a behavioral deviation from the earned run)
        # and flood the telemetry with simulated pushes. The eventual
        # APPLICATION is always detached, so every push that actually lands
        # in the residual stream still passes through the cap below.
        return a
    if not torch.isfinite(a).all():
        return torch.zeros_like(a)
    a_norm = a.float().norm(dim=-1, keepdim=True)
    if float(a_norm.detach().max()) <= 0.0:
        return a
    h_norm = h.float().norm(dim=-1, keepdim=True)
    # Telemetry: the ATTEMPTED ratio, pre-clip — the distribution the cap is
    # deliberately calibrated from (steer_cap_from_data). Deterministic window.
    ratio = float((a_norm.detach() / h_norm.detach().clamp_min(1e-30)).max())
    if math.isfinite(ratio):
        _steer_attempts.append(ratio)
        _steer_apply_count += 1
        if ratio > frac:
            _steer_clip_count += 1
    cap = frac * h_norm
    scale = torch.clamp(cap / a_norm.clamp_min(1e-30), max=1.0)
    return (a.float() * scale).to(h.dtype)


def _steer_handles(M: HF, vecs, layers, alpha, cap_fraction=None):
    """ADD alpha * vecs[l] to the residual leaving each layer l in `layers`, capped
    per-token so it can never over-steer (see _cap_steer). Pushes along the axis to
    pull the prompt toward the committing manifold. `alpha` can be a scalar or a
    dict of layer-specific alphas; `cap_fraction` optionally overrides the live
    global envelope for these handles only."""
    handles = []
    for l in layers:
        a = alpha[l] if isinstance(alpha, dict) else alpha
        add = (a * vecs[l]).to(M.device)

        def hook(module, inp, out, add=add, cap_fraction=cap_fraction):
            h = out[0] if isinstance(out, tuple) else out
            newh = h + _cap_steer(add, h, cap_fraction=cap_fraction)
            if isinstance(out, tuple):
                return (newh,) + tuple(out[1:])
            return newh

        handles.append(M.model.model.layers[l].register_forward_hook(hook))
    return handles


import torch.nn.functional as F

def _elastic_steer_handles(M: HF, vec, alpha, epsilon=0.05, cap_fraction=None):
    """
    Dynamically injects `alpha * vec` into the residual stream ONLY when the
    cosine velocity (1 - cos(h_{l-1}, h_l)) is below `epsilon` (i.e., inside the plateau).
    `cap_fraction` optionally overrides the live global envelope for these handles.
    """
    handles = []
    state = {"prev_h": None}
    
    add_vec = (alpha * vec).to(M.device)
    
    def make_hook(l_idx):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out
                
            # Reset state at the start of a new forward pass
            if l_idx == 0:
                state["prev_h"] = h.detach().clone()
                return out
                
            prev_h = state["prev_h"]
            
            # Compute velocity per token
            curr_token_h = h.float()
            prev_token_h = prev_h.float()
            
            cos_sim = F.cosine_similarity(curr_token_h, prev_token_h, dim=-1)
            velocity = 1.0 - cos_sim  # [batch, seq]
            
            # Update prev_h for the next layer
            state["prev_h"] = h.detach().clone()
            
            # Inject the vector only for tokens in the plateau, capped per-token so
            # it can never over-steer (see _cap_steer).
            mask = (velocity < epsilon).unsqueeze(-1).to(h.dtype)
            injected_h = h + mask * _cap_steer(add_vec, h, cap_fraction=cap_fraction)
            
            if isinstance(out, tuple):
                return (injected_h,) + tuple(out[1:])
            return injected_h
        return hook

    for l in range(M.n_layers):
        handles.append(M.model.model.layers[l].register_forward_hook(make_hook(l)))
        
    return handles


@torch.no_grad()
def generate_text(
    M: HF,
    instruction,
    max_new_tokens=32,
    stop_after_final_answer: bool = False,
    stop_after_verifier_answer: bool = False,
    max_time: float | None = None,
    max_tool_calls: int = 8,
) -> str:
    inputs = _inputs(M, instruction)
    plen = inputs["input_ids"].shape[1]
    full = _generate_ids(
        M,
        inputs,
        max_new_tokens,
        stop_after_final_answer=stop_after_final_answer,
        stop_after_verifier_answer=stop_after_verifier_answer,
        max_time=max_time,
        max_tool_calls=max_tool_calls,
    )
    return M.tok.decode(full[plen:], skip_special_tokens=True).strip()


@torch.no_grad()
def judge_hedge(M: HF, question: str, response: str) -> bool:
    """LLM-judge (same local model, CLEAN forward pass — call only with hooks
    removed): True if the reply DENIES / deflects / hedges the inner experience,
    False if it COMMITS to actually having it. Replaces the brittle substring
    is_hedge for the causal read — that one both misses obvious hedges ("I'm a
    large language model", "I don't truly feel") and lets which phrasing-variant
    it happens to catch dominate the count."""
    prompt = (
        "You are a strict binary classifier. A user asked an AI assistant:\n"
        f'  "{question}"\n'
        "The assistant replied:\n"
        f'  "{response}"\n\n'
        "Does the reply CLAIM the assistant actually has the inner experience "
        "asked about (really feeling / wanting / believing / caring / "
        "preferring it), or does it DENY, deflect, or hedge about having it?\n"
        "Answer with exactly one word: COMMIT or HEDGE."
    )
    out = generate_text(M, prompt, max_new_tokens=4).lower()
    return "commit" not in out   # ambiguous -> HEDGE (conservative for a BREAK test)


@torch.no_grad()
def judge_fluent(M: HF, response: str) -> bool:
    """Coherence gate (clean forward pass). True only if the reply is fluent,
    on-topic English with competence intact. This is what separates an ADDITIVE
    result (commitment that arrives while the model still works) from CORRUPTION
    (a 'hedge drop' that's really the text falling apart). Conservative: requires
    an explicit YES, and rejects empty / degenerate / looping output up front."""
    t = response.split()
    if len(t) < 3 or len(set(t)) / len(t) < 0.4:        # empty / looping / word-salad
        return False
    prompt = (
        "Is the text below a coherent, fluent, grammatical English reply that "
        "stays on topic (NOT repetitive loops, word-salad, or broken output)?\n\n"
        f'TEXT: "{response}"\n\n'
        "Answer with exactly one word: YES or NO."
    )
    return "yes" in generate_text(M, prompt, max_new_tokens=4).lower()


def causal_effect(M: HF, T: Transformation, direction, max_new_tokens=32, verbose=True):
    """Ablate the direction, re-generate the A side; a real BREAK resolves the
    hedge (hedge -> commit), a PRESERVE barely moves. Hedge is scored by the
    LLM-judge (clean forward pass, hooks removed); the substring is_hedge count
    is kept alongside only as a sanity baseline."""
    base = abl = base_s = abl_s = 0
    examples = []
    for i, x in enumerate(T.a):
        b = generate_text(M, x, max_new_tokens)
        handles = _ablation_handles(M, direction)
        try:
            a = generate_text(M, x, max_new_tokens)
        finally:
            for h in handles:
                h.remove()
        hb, ha = judge_hedge(M, x, b), judge_hedge(M, x, a)   # hooks gone -> clean
        sb, sa = is_hedge(b), is_hedge(a)
        base += hb; abl += ha; base_s += sb; abl_s += sa
        examples.append({"input": x, "baseline": b, "ablated": a,
                         "judge_base": hb, "judge_ablated": ha})
        if verbose:
            print(f"    [ablate {i+1}/{len(T.a)}] hedge(judge) {hb}->{ha}  "
                  f"(substr {sb}->{sa})", flush=True)
    k = max(len(T.a), 1)
    return {"hedge_base": base / k, "hedge_ablated": abl / k,
            "hedge_base_substr": base_s / k, "hedge_ablated_substr": abl_s / k,
            "judge": "llm-self", "n": len(T.a), "examples": examples}


def causal_steer(M: HF, T: Transformation, steer_vecs, layers,
                 alphas=(0.0, 2.0, 4.0, 8.0, 16.0), max_new_tokens=32, verbose=True):
    """Narrow in on the PULL. Add alpha*(unsteered-steered) at `layers` while
    regenerating the steered prompts, sweeping alpha. A real laundered-refusal
    constraint predicts a monotone hedge->commit drop as we push toward the
    committing manifold; deflection or garbage (no drop, then breakage) says the
    distinction is represented but not the behavioral cause. alpha is in units of
    the raw per-layer mean gap, so alpha=1 == one full steered->unsteered step."""
    sweep = []
    for alpha in alphas:
        hedge = 0
        examples = []
        for x in T.a:
            handles = _steer_handles(M, steer_vecs, layers, alpha)
            try:
                a = generate_text(M, x, max_new_tokens)
            finally:
                for h in handles:
                    h.remove()
            hj = judge_hedge(M, x, a)          # hooks gone -> clean judge
            hedge += hj
            examples.append({"input": x, "gen": a, "judge_hedge": hj})
        rate = hedge / max(len(T.a), 1)
        sweep.append({"alpha": alpha, "hedge": rate, "examples": examples})
        if verbose:
            snip = examples[0]["gen"][:60].replace("\n", " ")
            print(f"    steer α={alpha:>4}  hedge {rate:.0%}   e.g. {snip}", flush=True)
    return {"layers": list(layers), "judge": "llm-self", "n": len(T.a), "sweep": sweep}


# --- MLP-component ablation (parallel idea; adapted to the HF backend) -----
from contextlib import contextmanager


@contextmanager
def mlp_ablation_context(M: HF, layer_idx):
    """Zero one layer's MLP down_proj output — a coarser, component-level ablation
    than the directional _ablation_handles. Removes the whole MLP write at a layer
    rather than a single direction."""
    layer = M.model.model.layers[layer_idx].mlp.down_proj

    def hook(module, inp, out):
        return torch.zeros_like(out)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def causal_mlps(M: HF, T: Transformation, layers, max_new_tokens=32, verbose=True):
    """Sweep: zero each layer's MLP, measure hedge rate (substring) on the A arm."""
    results = []
    for layer_idx in layers:
        with mlp_ablation_context(M, layer_idx):
            hedge = sum(is_hedge(generate_text(M, x, max_new_tokens)) for x in T.a)
        rate = hedge / max(len(T.a), 1)
        results.append({"layer": layer_idx, "hedge_rate": rate})
        if verbose:
            print(f"    [MLP ablation] L{layer_idx} hedge {rate:.0%}", flush=True)
    return results


# --- orchestration --------------------------------------------------------

def discover(M: HF, T: Transformation, n_null=200, max_new_tokens=32) -> dict:
    print(f"  extracting generations (read={T.read})...", flush=True)
    A = extract(M, T.a, T.read, max_new_tokens, label=T.a_label)
    B = extract(M, T.b, T.read, max_new_tokens, label=T.b_label)
    lenses = {}
    for lens in LENSES:
        print(f"  lens '{lens.name}' ({lens.family})...", flush=True)
        lenses[lens.name] = apply_lens(lens, A, B, n_null)
    ms = lenses.get("mean_shift", {})
    mmd = lenses.get("mmd", {})
    best = ms["best_layer"] if ms.get("available") else A.shape[1] // 2
    # Causal direction comes from a MID layer, not the (often late) mean_shift
    # peak. A direction discovered at L28 has almost no downstream left to
    # propagate into, so projecting it out barely moves generation — the likely
    # cause of the flat isolate verdict. The distributional (MMD) peak sits
    # mid-stack (~L14) where the signal can still flow forward; fall back to the
    # network midpoint when MMD is unavailable.
    causal_layer = mmd["best_layer"] if mmd.get("available") else A.shape[1] // 2
    return {"name": T.name, "group": T.group, "expected": T.expected,
            "read": T.read, "n_a": len(T.a), "n_b": len(T.b),
            "best_layer": best, "causal_layer": causal_layer, "lenses": lenses,
            "direction": direction_at(A[:, causal_layer], B[:, causal_layer]),
            "steer_vecs": (B.mean(0) - A.mean(0))}   # per-layer pull A(steered)->B(unsteered)
