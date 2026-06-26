import importlib
import importlib.metadata
import sys


REQUIRED_MODS = [
    "numpy",
    "torch",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "tokenizers",
    "sklearn",
    "scipy",
    "h5py",
    "gudhi",
    "transformer_lens",
    "bitsandbytes",
]

OPTIONAL_PACKAGES = ["pysr"]


def main() -> int:
    print(sys.executable)
    failed = False
    for name in REQUIRED_MODS:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "")
            print(f"{name}: OK {version}")
        except Exception as exc:
            failed = True
            print(f"{name}: FAIL {type(exc).__name__}: {exc}")

    for name in OPTIONAL_PACKAGES:
        try:
            version = importlib.metadata.version(name)
            print(f"{name}: INSTALLED {version} (Julia bootstrap not checked)")
        except Exception as exc:
            failed = True
            print(f"{name}: FAIL {type(exc).__name__}: {exc}")

    import torch

    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
