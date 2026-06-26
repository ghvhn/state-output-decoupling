import gc
import ctypes
import torch
from transformer_lens import HookedTransformer


def load(name: str, device: str = "cuda") -> HookedTransformer:
    """
    Loads model onto GPU for cache extraction (run_with_cache).
    TransformerLens generate() is broken on Blackwell (RTX 50xx) — generation
    uses manual token-by-token sampling on CPU via data/generate.py instead.
    """
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.float16)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
        # TransformerLens leaves CPU copies in Python's heap after .to("cuda").
        # Force-release them back to the OS so generation doesn't run on pagefile.
        gc.collect()
        torch.cuda.empty_cache()
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        print(f"  Model on GPU ({torch.cuda.get_device_name(0)})")
    else:
        model.to("cpu")
        print("  Model on CPU (no CUDA available)")
    return model


def n_layers(model: HookedTransformer) -> int:
    return model.cfg.n_layers


def d_model(model: HookedTransformer) -> int:
    return model.cfg.d_model
