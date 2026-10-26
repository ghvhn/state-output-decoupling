"""Bounded native-FP8 GLM teacher capture, launched by torchrun.

The production adapter is source-backed against Transformers 5.15.0. CPU
fixture tests validate the capture contract, not the 753B GPU runtime. Loading
and a CUDA forward can only be forcibly bounded by the caller's process timeout.
No cloud API, model download, generation, fitting, or training happens here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import torch


MODEL_ID = "zai-org/GLM-5.3"
REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
FP8_KERNEL_COMMIT = "8c178950e8710e26a2210e4e909cd02dc16c8715"
IDENTITY_KEYS = {"model_id", "revision", "hidden_size", "vocab_size", "num_layers",
                 "source_precision", "tokenizer_sha256", "chat_template_sha256"}


def _positive_int(value: Any, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def validate_identity(identity: dict, *, production: bool = False) -> dict:
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        raise ValueError("capture identity must contain exactly the eight scope fields")
    result = dict(identity)
    for key in ("model_id", "revision", "source_precision"):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError(f"identity {key} must be nonempty")
    for key in ("hidden_size", "vocab_size", "num_layers"):
        _positive_int(result[key], key)
    for key in ("tokenizer_sha256", "chat_template_sha256"):
        _sha256(result[key], key)
    if production:
        expected = {"model_id": MODEL_ID, "revision": REVISION, "hidden_size": 6144,
                    "vocab_size": 154880, "num_layers": 78,
                    "source_precision": "native_fp8_e4m3fn"}
        if any(result[key] != value for key, value in expected.items()):
            raise ValueError("production capture requires the pinned original GLM-5.3 FP8 identity")
    return result


@dataclass(frozen=True)
class CaptureExample:
    trajectory_id: str
    environment_id: str
    split: str
    token_ids: tuple[int, ...]
    selected_positions: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    metadata: dict


def validate_examples(corpus: dict, identity: dict, *, max_seq_len: int = 1024) -> list[CaptureExample]:
    """Validate pretokenized records before model loading or GPU allocation."""
    validate_identity(identity)
    _positive_int(max_seq_len, "max_seq_len", 4096)
    if not isinstance(corpus, dict) or corpus.get("model_id") != identity["model_id"] or corpus.get("revision") != identity["revision"]:
        raise ValueError("corpus model and revision must match capture identity")
    tokenizer_identity = corpus.get("tokenizer_identity", corpus)
    for key in ("tokenizer_sha256", "chat_template_sha256"):
        if tokenizer_identity.get(key) != identity[key]:
            raise ValueError(f"corpus {key} differs from capture identity")
    records = corpus.get("examples")
    if not isinstance(records, list) or not records:
        raise ValueError("corpus requires at least one example")
    if len(records) > 100000:
        raise ValueError("corpus exceeds bounded example count")
    result, trajectory_scope = [], {}
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError("each corpus example must be an object")
        trajectory, environment = row.get("trajectory_id"), row.get("environment_id")
        split = row.get("split")
        if not isinstance(trajectory, str) or not trajectory or not isinstance(environment, str) or not environment:
            raise ValueError("examples require nonempty trajectory and environment IDs")
        if split not in ("calibration", "heldout"):
            raise ValueError("split must be calibration or heldout")
        if trajectory in trajectory_scope:
            raise ValueError("a trajectory must appear in exactly one corpus example")
        trajectory_scope[trajectory] = (environment, split)
        tokens, positions = row.get("token_ids"), row.get("selected_positions")
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= max_seq_len:
            raise ValueError("token count must be between two and max_seq_len; runtime never truncates context")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < identity["vocab_size"] for v in tokens):
            raise ValueError("token IDs must be integers inside the pinned vocabulary")
        if not isinstance(positions, list) or not positions:
            raise ValueError("selected_positions must be nonempty")
        is_framing = row.get("kind") == "probe_framing"
        upper = len(tokens) if is_framing else len(tokens) - 1
        if is_framing and (not isinstance(row.get("probe_id"), str) or not row["probe_id"] or row.get("polarity") not in (-1, 1)):
            raise ValueError("probe framings require an explicit probe ID and signed polarity")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < upper for v in positions):
            raise ValueError("selected positions require an actual next token in the same example")
        if positions != sorted(set(positions)):
            raise ValueError("selected positions must be sorted and unique")
        pairs = row.get("pairs", [])
        if not isinstance(pairs, list):
            raise ValueError("pairs must be an explicit list of token-position endpoints")
        checked_pairs = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in pair):
                raise ValueError("pair endpoints must be two integer token positions")
            if pair[0] not in positions or pair[1] not in positions or pair[0] >= pair[1]:
                raise ValueError("pairs require increasing selected endpoints inside the same trajectory")
            checked_pairs.append(tuple(pair))
        if len(set(checked_pairs)) != len(checked_pairs):
            raise ValueError("pair endpoints must be unique")
        for key in ("provenance_weight", "stimulation_weight"):
            weight = row.get(key, 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError(f"{key} must be finite and between zero and one")
        metadata = {key: value for key, value in row.items()
                    if key not in ("token_ids", "selected_positions", "pairs")}
        metadata["corpus_example_index"] = index
        result.append(CaptureExample(trajectory, environment, split, tuple(tokens), tuple(positions),
                                     tuple(checked_pairs), metadata))
    return result


def prepare_batch(examples: list[CaptureExample], *, pad_token_id: int, device: Any = "cpu") -> dict:
    """Right pad separate sequences. Never concatenate independent conversations."""
    if not examples:
        raise ValueError("cannot prepare an empty batch")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    width = max(len(row.token_ids) for row in examples)
    ids = torch.full((len(examples), width), pad_token_id, dtype=torch.int64, device=device)
    mask = torch.zeros_like(ids)
    selected_batch, selected_position, rows, pair_rows = [], [], [], []
    offset = 0
    for batch_index, row in enumerate(examples):
        ids[batch_index, :len(row.token_ids)] = torch.tensor(row.token_ids, dtype=torch.int64, device=device)
        mask[batch_index, :len(row.token_ids)] = 1
        lookup = {}
        for position in row.selected_positions:
            lookup[position] = offset
            # Context/source provenance is stored once per example in each
            # chunk. Repeating it for every selected token inflates the index.
            record = {key: row.metadata[key] for key in ("conversation_id", "kind", "probe_id", "polarity",
                "provenance_weight", "stimulation_weight", "selection_bucket") if key in row.metadata}
            record.update(id=f"{row.trajectory_id}:{position}", trajectory_id=row.trajectory_id,
                          environment_id=row.environment_id, split=row.split,
                          token_position=position, token_id=row.token_ids[position],
                          next_token_id=row.token_ids[position + 1] if position + 1 < len(row.token_ids) else -1,
                          next_token_observed=position + 1 < len(row.token_ids))
            rows.append(record)
            selected_batch.append(batch_index)
            selected_position.append(position)
            offset += 1
        pair_rows.extend([[lookup[left], lookup[right]] for left, right in row.pairs])
    return {"input_ids": ids, "attention_mask": mask,
            "selected_batch": torch.tensor(selected_batch, dtype=torch.int64, device=device),
            "selected_position": torch.tensor(selected_position, dtype=torch.int64, device=device),
            "rows": rows, "pair_rows": pair_rows}


def estimate_chunk_bytes(examples: list[CaptureExample], identity: dict, selected_layers: list[int],
                         *, spotcheck_rows: int = 0) -> int:
    """Native BF16 state payload plus conservative serializer reserve."""
    rows = sum(len(example.selected_positions) for example in examples)
    state_bytes = rows * identity["hidden_size"] * 2 * (len(selected_layers) + 2)
    # Eight selected router IDs (int64), weights (fp32), and three int64 timelines.
    timeline_router_bytes = rows * (8 * (8 + 4) + 3 * 8)
    anchors = min(rows, spotcheck_rows) * (identity["vocab_size"] * 2 + 8)
    return state_bytes + timeline_router_bytes + anchors + 128 * 1024


def validate_capture_budget(examples: list[CaptureExample], identity: dict, selected_layers: list[int],
                            *, batch_size: int, max_chunk_bytes: int, max_total_bytes: int,
                            spotcheck_rows: int = 2) -> int:
    """Reject an impossible first capture before loading the model."""
    _positive_int(batch_size, "batch_size", 8)
    _positive_int(max_chunk_bytes, "max_chunk_bytes")
    _positive_int(max_total_bytes, "max_total_bytes")
    head_bytes = identity["hidden_size"] * identity["vocab_size"] * 2
    head_reserve = head_bytes + math.ceil(identity["vocab_size"] / 1024) * 128 * 1024
    first_bytes = 0
    for offset in range(0, len(examples), batch_size):
        expected = estimate_chunk_bytes(examples[offset:offset + batch_size], identity, selected_layers,
                                        spotcheck_rows=spotcheck_rows if offset == 0 else 0)
        if expected > max_chunk_bytes:
            raise ValueError("selected-position batch exceeds archive chunk budget; reduce batch size or preselect fewer positions")
        if offset == 0:
            first_bytes = expected
    if head_reserve + first_bytes >= max_total_bytes:
        raise ValueError("archive total budget cannot preserve native output head and first measured batch")
    return head_reserve


def logical_tensor(value: torch.Tensor, *, expected_shape: tuple[int, ...], name: str) -> torch.Tensor:
    """Materialize a DTensor collectively; ordinary tensors must already be logical."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    # DTensor.full_tensor is a collective: every TP rank must invoke this helper.
    if hasattr(value, "full_tensor") and hasattr(value, "placements"):
        value = value.full_tensor()
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} has {tuple(value.shape)}; expected logical shape {expected_shape}")
    return value


class CaptureHooks:
    """Capture selected native tensors at semantic module boundaries on all ranks."""
    def __init__(self, model: Any, identity: dict, selected_layers: list[int], target_layer: int):
        self.model, self.identity = model, identity
        self.selected_layers, self.target_layer = tuple(selected_layers), target_layer
        if len(set(selected_layers)) != len(selected_layers) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < identity["num_layers"] for i in selected_layers):
            raise ValueError("selected layer indices must be unique native layer indices")
        if isinstance(target_layer, bool) or not isinstance(target_layer, int) or not 0 <= target_layer < identity["num_layers"]:
            raise ValueError("target layer is outside the original model")
        self.values, self.handles, self.batch = {}, [], None

    def _put(self, name: str, value: torch.Tensor, *, flattened: bool = False, last_dim: int | None = None):
        if self.batch is None or name in self.values:
            raise RuntimeError(f"capture hook {name} called outside its one-forward contract")
        batch_size, seq_len = self.batch["input_ids"].shape
        last_dim = self.identity["hidden_size"] if last_dim is None else last_dim
        shape = (batch_size * seq_len, last_dim) if flattened else (batch_size, seq_len, last_dim)
        value = logical_tensor(value, expected_shape=shape, name=name)
        if flattened:
            value = value.reshape(batch_size, seq_len, last_dim)
        # Copy only selected positions, preserving native activation precision.
        selected = value[self.batch["selected_batch"], self.batch["selected_position"]]
        self.values[name] = selected.detach().contiguous().cpu().clone()

    def install(self):
        base = self.model.model
        if len(base.layers) != self.identity["num_layers"]:
            raise ValueError("native decoder layer count differs from pinned identity")
        for layer in self.selected_layers:
            self.handles.append(base.layers[layer].register_forward_hook(
                lambda _module, _args, output, layer=layer: self._put(
                    f"layer_{layer}_hidden", output[0] if isinstance(output, tuple) else output)))
        target = base.layers[self.target_layer].mlp
        if not hasattr(target, "experts") or not hasattr(target, "gate"):
            raise ValueError("target must be the native MoE module, including experts and router")
        self.handles.append(target.experts.register_forward_pre_hook(
            lambda _module, args: self._put(f"layer_{self.target_layer}_expert_input", args[0], flattened=True)))
        def router_hook(_module, _args, output):
            if not isinstance(output, tuple) or len(output) != 3:
                raise ValueError("native router must expose logits, selected weights, selected IDs")
            _logits, weights, indices = output
            count = int(getattr(target.gate, "top_k", weights.shape[-1]))
            self._put(f"layer_{self.target_layer}_router_weights", weights, flattened=True, last_dim=count)
            self._put(f"layer_{self.target_layer}_router_ids", indices, flattened=True, last_dim=count)
        self.handles.append(target.gate.register_forward_hook(router_hook))
        return self

    def begin(self, batch: dict):
        self.batch, self.values = batch, {}

    def finish(self, final_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        self._put("final_hidden", final_hidden)
        expected = {"final_hidden", f"layer_{self.target_layer}_expert_input",
                    f"layer_{self.target_layer}_router_ids", f"layer_{self.target_layer}_router_weights"}
        expected.update(f"layer_{i}_hidden" for i in self.selected_layers)
        if set(self.values) != expected:
            raise RuntimeError(f"native hooks did not capture every required state: {expected - set(self.values)}")
        values = self.values
        values["positions"] = self.batch["selected_position"].detach().cpu().clone()
        values["token_ids"] = torch.tensor([row["token_id"] for row in self.batch["rows"]], dtype=torch.int64)
        values["next_token_ids"] = torch.tensor([row["next_token_id"] for row in self.batch["rows"]], dtype=torch.int64)
        for key, tensor in values.items():
            if (key == "final_hidden" or key.endswith("_hidden") or key.endswith("_expert_input")) and tensor.dtype != torch.bfloat16:
                raise ValueError(f"{key} changed the expected native BF16 activation precision")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"nonfinite measured state in {key}")
        self.batch, self.values = None, {}
        return values

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.batch, self.values = None, {}


class CaptureDeadline:
    def __init__(self, deadline_epoch: float, reserve_seconds: float = 45.0,
                 *, wall_clock: Callable = time.time, monotonic_clock: Callable = time.monotonic):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (deadline_epoch, reserve_seconds)) or reserve_seconds < 0:
            raise ValueError("deadline and save reserve must be finite; reserve cannot be negative")
        self.clock, self.reserve = monotonic_clock, float(reserve_seconds)
        self.deadline = monotonic_clock() + max(0.0, float(deadline_epoch) - wall_clock())

    def remaining(self) -> float:
        return max(0.0, self.deadline - self.clock())

    def may_start(self, expected_seconds: float = 0.0) -> bool:
        if not math.isfinite(expected_seconds) or expected_seconds < 0:
            raise ValueError("estimated forward seconds must be finite and nonnegative")
        return self.remaining() > self.reserve + expected_seconds


def validate_checkpoint_metadata(checkpoint_dir: Path, identity: dict) -> dict:
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    expected = {"model_type": "glm_moe_dsa", "hidden_size": identity["hidden_size"],
                "vocab_size": identity["vocab_size"], "num_hidden_layers": identity["num_layers"]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint architecture dimensions differ from capture scope")
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "fp8" or quant.get("fmt", "e4m3") != "e4m3" or quant.get("weight_block_size") != [128, 128] or quant.get("dequantize", False):
        raise ValueError("checkpoint must retain original dynamic e4m3 block-FP8 weights")
    if quant.get("activation_scheme") != "dynamic":
        raise ValueError("original GLM dynamic activation scheme is required")
    return config


def verify_fp8_kernel_cache(*, commit: str, cache_dir: Path, offline: bool, download_file=None) -> dict:
    """Retain the official version ref as well as the immutable kernel snapshot."""
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise RuntimeError("native FP8 kernel must resolve to an immutable commit")
    cache_dir = Path(cache_dir).resolve()
    repository = cache_dir / "kernels--kernels-community--finegrained-fp8"
    snapshot = repository / "snapshots" / commit
    metadata = sorted(snapshot.glob("build/*/metadata.json"))
    if not metadata:
        raise RuntimeError("native FP8 kernel snapshot has no cached build metadata")
    if not offline:
        if download_file is None:
            from huggingface_hub import hf_hub_download
            download_file = hf_hub_download
        # Loading version=4 downloads by commit and does not itself cache refs/v4.
        # Downloading an official metadata file by that version creates the ref.
        download_file(repo_id="kernels-community/finegrained-fp8", repo_type="kernel",
                      filename=metadata[0].relative_to(snapshot).as_posix(), revision="v4",
                      cache_dir=str(cache_dir))
    ref = repository / "refs" / "v4"
    if not ref.is_file() or ref.read_text().strip() != commit:
        raise RuntimeError("cached native FP8 version ref differs from the loaded commit")
    artifacts, total = [], 0
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if not path.resolve().is_relative_to(cache_dir):
            raise RuntimeError("native FP8 cache artifact escapes its cache")
        size = path.stat().st_size
        total += size
        if size > 256 * 1024**2 or total > 1024**3 or len(artifacts) >= 4096:
            raise RuntimeError("native FP8 cache exceeds its inspection budget")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        artifacts.append({"file": path.relative_to(snapshot).as_posix(), "bytes": size,
                          "sha256": digest.hexdigest()})
    return {"repo_id": "kernels-community/finegrained-fp8", "version": 4, "commit": commit,
            "cache_artifacts": artifacts, "cache_bytes": total}


def validate_fp8_kernel_bundle(bundle, loaded_kernels, *, cache_dir: Path):
    """Match Transformers' callable bundle to its actual registered native module."""
    symbols = (("matmul", "matmul_2d"), ("batched_matmul", "matmul_batched"),
               ("grouped_matmul", "matmul_grouped"))
    loaded = [entry for entry in loaded_kernels if all(
        callable(getattr(bundle, name, None)) and
        getattr(entry.module, symbol, None) is getattr(bundle, name)
        for name, symbol in symbols)]
    if len(loaded) != 1 or loaded[0].metadata.version != 4:
        raise RuntimeError("native FP8 kernel identity is ambiguous or incompatible")
    cache_dir = Path(cache_dir).resolve()
    snapshot = cache_dir / "kernels--kernels-community--finegrained-fp8" / "snapshots" / FP8_KERNEL_COMMIT
    module_path = Path(loaded[0].module.__file__).absolute()
    if not module_path.is_relative_to(snapshot / "build") or not module_path.resolve().is_relative_to(cache_dir):
        raise RuntimeError("native FP8 module came from outside the pinned cache build")
    if loaded[0].repo_info is not None and loaded[0].repo_info.revision != FP8_KERNEL_COMMIT:
        raise RuntimeError("native FP8 module revision differs from the pinned commit")


def runtime_capabilities() -> dict:
    """Source-level CPU preflight; intentionally does not assert GPU validation."""
    import transformers
    if transformers.__version__ != "5.15.0":
        raise RuntimeError("production adapter is pinned to transformers==5.15.0")
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.integrations.finegrained_fp8 import (
        FP8Experts, ALL_FP8_EXPERTS_FUNCTIONS, load_finegrained_fp8_kernel,
    )
    plan = GlmMoeDsaConfig.base_model_tp_plan
    required = {"layers.*.mlp.experts.gate_up_proj", "layers.*.mlp.experts.down_proj", "layers.*.mlp.experts"}
    if not required.issubset(plan) or "grouped_mm" not in ALL_FP8_EXPERTS_FUNCTIONS:
        raise RuntimeError("installed runtime lacks source-backed GLM TP / grouped FP8 experts")
    from kernels import get_kernel, get_loaded_kernels
    from huggingface_hub import constants as hub_constants
    cache_dir = Path(os.environ.get("KERNELS_CACHE") or hub_constants.HF_HUB_CACHE).resolve()
    snapshot = cache_dir / "kernels--kernels-community--finegrained-fp8" / "snapshots" / FP8_KERNEL_COMMIT
    # Warm the trusted official immutable build online. Hub 1.33's offline
    # snapshot lookup requires every build variant, whereas kernels downloads
    # only the selected variant. Its supported local override uses that same
    # native build without claiming the whole repository is cached.
    if not hub_constants.HF_HUB_OFFLINE:
        get_kernel("kernels-community/finegrained-fp8", revision=FP8_KERNEL_COMMIT)
    kernel_cache = verify_fp8_kernel_cache(commit=FP8_KERNEL_COMMIT, cache_dir=cache_dir,
                                         offline=hub_constants.HF_HUB_OFFLINE)
    override = "kernels-community/finegrained-fp8=" + str(snapshot)
    if os.environ.get("LOCAL_KERNELS") not in (None, override):
        raise RuntimeError("native FP8 local override differs from the pinned cache")
    os.environ["LOCAL_KERNELS"] = override
    kernel = load_finegrained_fp8_kernel()
    if not all(callable(getattr(kernel, name, None)) for name in ("matmul", "batched_matmul", "grouped_matmul")):
        raise RuntimeError("native finegrained-FP8 kernel lacks required entry points")
    validate_fp8_kernel_bundle(kernel, get_loaded_kernels(), cache_dir=cache_dir)
    return {"transformers": transformers.__version__, "torch": torch.__version__,
            "glm_native_tp": True, "native_fp8_experts": FP8Experts.__name__,
            "experts_implementation": "grouped_mm", "fp8_kernel_cache_warmed": True,
            "fp8_kernel": kernel_cache,
            "gpu_forward_validated": False}


def validate_native_tp_head(head: Any, identity: dict, *, world_size: int) -> dict:
    """Check local TP storage without confusing it with gathered vocabulary logits."""
    world_size = _positive_int(world_size, "tensor parallel world size")
    vocab, hidden = identity["vocab_size"], identity["hidden_size"]
    if vocab % world_size:
        raise ValueError("native gathered vocabulary must divide evenly across TP ranks")
    if getattr(head, "_hf_tp_plan", None) != "colwise_gather_output":
        raise ValueError("native output head must gather its colwise tensor-parallel output")
    weight = head.weight
    local_shape = (vocab // world_size, hidden)
    if hasattr(weight, "to_local") and hasattr(weight, "placements"):
        if tuple(weight.shape) != (vocab, hidden):
            raise ValueError("native output head has an invalid logical layout")
        weight = weight.to_local()
    if tuple(weight.shape) != local_shape or weight.dtype != torch.bfloat16:
        raise ValueError("native output head must retain its BF16 vocabulary-row shard")
    return {"logical_shape": [vocab, hidden], "local_weight_shape": list(local_shape),
            "tp_style": "colwise_gather_output", "dtype": "bfloat16"}


def _source_head_blocks(checkpoint_dir: Path, identity: dict, rows_per_block: int = 1024):
    from safetensors import safe_open
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard_name = index.get("weight_map", {}).get("lm_head.weight")
    if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
        raise ValueError("output head must map to a single contained safetensors shard")
    with safe_open(str(checkpoint_dir / shard_name), framework="pt", device="cpu") as source:
        head = source.get_slice("lm_head.weight")
        if tuple(head.get_shape()) != (identity["vocab_size"], identity["hidden_size"]) or head.get_dtype() != "BF16":
            raise ValueError("source output head must retain native BF16 [vocabulary, hidden] layout")
        for start in range(0, identity["vocab_size"], rows_per_block):
            yield start, head[start:min(start + rows_per_block, identity["vocab_size"]), :].clone().contiguous()


def _verify_replicated(values: dict[str, torch.Tensor], device: torch.device):
    """First-batch check that ordinary logical captures agree across TP ranks."""
    import torch.distributed as dist
    for name in sorted(values):
        value = values[name].to(device)
        reference = value.clone()
        dist.broadcast(reference, src=0)
        failed = torch.tensor(int(not torch.equal(value, reference)), device=device, dtype=torch.int64)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(f"TP ranks disagree on logical capture {name}; refusing rank-zero partial states")


def native_load_options(world_size: int = 8) -> dict:
    """The pinned Transformers 5.15 distributed loading API."""
    if world_size != 8:
        raise RuntimeError("native capture loading requires exactly eight ranks")
    from transformers.distributed import DistributedConfig
    return {"local_files_only": True, "trust_remote_code": False, "dtype": torch.bfloat16,
            "distributed_config": DistributedConfig(tp_size=world_size),
            "attn_implementation": "sdpa", "experts_implementation": "grouped_mm"}


def run_capture(config: dict) -> dict:
    """Production torchrun entry. Caller must enforce a hard process timeout."""
    identity = validate_identity(config["identity"], production=True)
    deadline = CaptureDeadline(config["deadline_epoch"], config.get("save_reserve_seconds", 45.0))
    checkpoint = Path(config["checkpoint_dir"])
    validate_checkpoint_metadata(checkpoint, identity)
    corpus_path = Path(config["corpus_path"])
    corpus_bytes = corpus_path.read_bytes()
    if len(corpus_bytes) > 128 * 1024 * 1024:
        raise ValueError("corpus manifest exceeds 128MiB")
    corpus = json.loads(corpus_bytes)
    examples = validate_examples(corpus, identity, max_seq_len=config.get("max_seq_len", 1024))
    batch_size = _positive_int(config.get("batch_size", 2), "batch_size", 8)
    selected_layers = config.get("selected_layers", [3, 12, 24, 36, 48, 60, 77])
    target_layer = config.get("target_layer", 3)
    target_ids = config.get("target_expert_ids", list(range(8)))
    if target_layer != 3 or target_ids != list(range(8)):
        raise ValueError("first native capture is scoped to layer 3 gate/up logical experts 0..7")
    spotcheck_rows = _positive_int(config.get("spotcheck_rows", 2), "spotcheck_rows", 16)
    max_chunk_bytes = config.get("max_chunk_bytes", 64 * 1024 ** 2)
    max_total_bytes = config.get("max_total_bytes", 8 * 1024 ** 3)
    head_reserve = validate_capture_budget(examples, identity, selected_layers, batch_size=batch_size,
        max_chunk_bytes=max_chunk_bytes, max_total_bytes=max_total_bytes, spotcheck_rows=spotcheck_rows)
    runtime_info = runtime_capabilities()
    if not torch.cuda.is_available():
        raise RuntimeError("native FP8 capture requires CUDA; CPU dequantization is forbidden")
    import torch.distributed as dist
    from datetime import timedelta
    rank, local_rank, world = int(os.environ.get("RANK", "-1")), int(os.environ.get("LOCAL_RANK", "-1")), int(os.environ.get("WORLD_SIZE", "0"))
    if world != 8 or not 0 <= local_rank < 8 or not 0 <= rank < 8:
        raise RuntimeError("run original GLM FP8 with torchrun eight ranks on one eight-GPU node")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("initial runtime is scoped to Hopper H200 compute capability 9.0")
    dist.init_process_group("nccl", timeout=timedelta(seconds=max(30, min(1200, int(deadline.remaining())))))
    writer, hooks, model = None, None, None
    chunks, rows_done, last_duration, head_rows, committed_bytes = 0, 0, 0.0, 0, 0
    status, failure = "partial", None
    try:
        if not deadline.may_start():
            raise TimeoutError("capture deadline expired before loading")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **native_load_options(world))
        model.eval()
        quantizer = getattr(model, "hf_quantizer", None)
        quant_config = getattr(quantizer, "quantization_config", None)
        experts = model.model.layers[target_layer].mlp.experts
        expert_weight = experts.gate_up_proj
        if hasattr(expert_weight, "to_local"):
            expert_weight = expert_weight.to_local()
        if quant_config is None or getattr(quant_config, "dequantize", False) or expert_weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError("loader did not retain native FP8 experts; refusing altered baseline")
        head_layout = validate_native_tp_head(model.lm_head, identity, world_size=world)
        if rank == 0:
            from tda.glm_capture_archive import CaptureArchiveWriter
            writer = CaptureArchiveWriter(config["output_dir"], identity,
                max_total_bytes=max_total_bytes, max_chunk_bytes=max_chunk_bytes,
                origin="measured_model_state", run_metadata={
                    "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
                    "runtime": runtime_info, "tensor_parallel_world_size": world,
                    "output_head_layout": head_layout,
                    "capture_adapter": "transformers_5_15_native_glm_tp",
                    "forward_mode": "teacher_forced_causal_no_generation",
                    "deadline_epoch": config["deadline_epoch"], "selected_layers": selected_layers,
                    "target_layer": target_layer, "target_expert_ids": target_ids,
                    "down_projection_inputs_captured": False})
        hooks = CaptureHooks(model, identity, selected_layers, target_layer).install()
        pad_id = corpus.get("pad_token_id", 0)
        for offset in range(0, len(examples), batch_size):
            next_bytes = estimate_chunk_bytes(examples[offset:offset + batch_size], identity, selected_layers,
                spotcheck_rows=spotcheck_rows if chunks == 0 else 0)
            may_start = torch.tensor(int(deadline.may_start(last_duration * 1.5) and
                committed_bytes + next_bytes + head_reserve < max_total_bytes), device=device, dtype=torch.int64)
            dist.all_reduce(may_start, op=dist.ReduceOp.MIN)
            if not may_start.item():
                break
            started = time.monotonic()
            batch = prepare_batch(examples[offset:offset + batch_size], pad_token_id=pad_id, device=device)
            hooks.begin(batch)
            with torch.inference_mode():
                output = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                tensors = hooks.finish(output.last_hidden_state)
                if chunks == 0:
                    _verify_replicated(tensors, device)
                    anchors = torch.arange(min(spotcheck_rows, len(batch["rows"])), device=device)
                    selected = output.last_hidden_state[batch["selected_batch"][anchors], batch["selected_position"][anchors]]
                    logits = model.lm_head(selected)
                    logits = logical_tensor(logits, expected_shape=(len(anchors), identity["vocab_size"]), name="full_logits_anchor")
                    tensors["spotcheck_logits"] = logits.detach().cpu().clone()
                    tensors["spotcheck_rows"] = anchors.detach().cpu().clone()
            metadata = {"rows": batch["rows"], "pairs": batch["pair_rows"],
                        "examples": [row.metadata for row in examples[offset:offset + batch_size]],
                        "state_sites": {"final_hidden": "post_final_norm_lm_head_input",
                            f"layer_{target_layer}_expert_input": "post_attention_norm_expert_gate_up_input",
                            **{f"layer_{i}_hidden": "decoder_block_output" for i in selected_layers}},
                        "router_layer": target_layer, "router_semantics": "native_selected_ids_and_scaled_normalized_weights",
                        "target_projection": "gate_up_shared_input", "target_expert_ids": target_ids,
                        "source_tensor_layout": "batch_sequence_hidden_then_selected_rows",
                        "rank": 0, "world_size": world, "logical_tensors_complete": True}
            write_failed = torch.zeros((), device=device, dtype=torch.int64)
            if rank == 0:
                try:
                    writer.write_chunk(f"chunk-{chunks:06d}", tensors, metadata)
                except Exception as error:
                    failure, write_failed = f"{type(error).__name__}: {error}", torch.ones((), device=device, dtype=torch.int64)
            dist.broadcast(write_failed, src=0)
            if write_failed.item():
                raise RuntimeError(failure or "rank-zero archive write failed")
            written = torch.tensor(writer.bytes_written if rank == 0 else 0, device=device, dtype=torch.int64)
            dist.broadcast(written, src=0)
            committed_bytes = int(written.item())
            rows_done += len(batch["rows"])
            chunks += 1
            del output, tensors, batch
            torch.cuda.synchronize(device)
            last_duration = time.monotonic() - started
        # Source head is copied in bounded CPU blocks rather than all-gathering it
        # onto every GPU. It is the same unchanged pinned BF16 output projection.
        head_failed = torch.zeros((), device=device, dtype=torch.int64)
        if rank == 0 and writer is not None:
            try:
                for start, block in _source_head_blocks(checkpoint, identity):
                    if deadline.remaining() <= 0:
                        break
                    writer.write_head_block(start, block)
                    head_rows += block.shape[0]
                status = "complete" if (chunks > 0 and head_rows == identity["vocab_size"] and
                    rows_done == sum(len(row.selected_positions) for row in examples)) else "partial"
                writer.finalize(status=status, details={"captured_rows": rows_done, "chunks": chunks,
                    "remaining_seconds": deadline.remaining(), "gpu_forward_validated": chunks > 0,
                    "output_head_rows": head_rows, "all_corpus_examples_captured":
                    rows_done == sum(len(row.selected_positions) for row in examples)})
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                head_failed.fill_(1)
                writer.finalize(status="failed", details={"error": failure, "captured_rows": rows_done, "chunks": chunks})
        dist.broadcast(head_failed, src=0)
        if head_failed.item():
            raise RuntimeError(failure or "rank-zero output-head archive write failed")
        return {"status": status, "rank": rank, "captured_rows": rows_done, "chunks": chunks,
                "remaining_seconds": deadline.remaining(), "gpu_forward_validated": chunks > 0}
    except Exception as error:
        if writer is not None and not writer.closed:
            writer.finalize(status="failed", details={"error": f"{type(error).__name__}: {error}",
                            "captured_rows": rows_done, "chunks": chunks})
        raise
    finally:
        if hooks is not None:
            hooks.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def check_rendezvous() -> dict:
    """Exercise the same eight local torchrun ranks without CUDA or model weights."""
    import torch.distributed as dist
    from datetime import timedelta
    rank, world = int(os.environ.get("RANK", "-1")), int(os.environ.get("WORLD_SIZE", "0"))
    if world != 8 or not 0 <= rank < 8 or os.environ.get("MASTER_ADDR") != "127.0.0.1":
        raise RuntimeError("CPU rendezvous requires eight ranks and literal loopback")
    dist.init_process_group("gloo", timeout=timedelta(seconds=45))
    try:
        value = torch.tensor(rank, dtype=torch.int64)
        dist.all_reduce(value)
        if value.item() != 28:
            raise RuntimeError("CPU rendezvous rank agreement failed")
        # Exercise a real save/load through the exact native constructor and
        # distributed API, without the production checkpoint or CUDA inference.
        import tempfile
        from transformers import AutoModelForCausalLM
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        temporary = tempfile.TemporaryDirectory(prefix="glm-tp-preflight-") if rank == 0 else None
        path = [temporary.name if temporary else None]
        if rank == 0:
            tiny_config = GlmMoeDsaConfig(vocab_size=32, hidden_size=128, intermediate_size=128,
                moe_intermediate_size=128, num_hidden_layers=2, num_attention_heads=8,
                num_key_value_heads=8, n_shared_experts=1, n_routed_experts=8,
                kv_lora_rank=32, q_lora_rank=32, qk_rope_head_dim=16, v_head_dim=16,
                qk_nope_head_dim=16, num_experts_per_tok=2, max_position_embeddings=32,
                index_topk=4, index_head_dim=16, index_n_heads=8, first_k_dense_replace=1)
            tiny = AutoModelForCausalLM.from_config(tiny_config)
            tiny.save_pretrained(temporary.name)
            del tiny
        dist.broadcast_object_list(path, src=0)
        try:
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (8,), mesh_dim_names=("tp",))
            loaded = AutoModelForCausalLM.from_pretrained(path[0], device_mesh=mesh, **native_load_options(world))
            validate_native_tp_head(loaded.lm_head, {"vocab_size": 32, "hidden_size": 128}, world_size=world)
            if loaded.config.distributed_config.tp_size != 8 or loaded.lm_head.weight.device.type != "cpu":
                raise RuntimeError("tiny GLM distributed CPU placement differs from its declared plan")
            with torch.inference_mode():
                logits = loaded.lm_head(torch.linspace(-1, 1, 128, dtype=torch.bfloat16).reshape(1, 128))
            if logits.shape != (1, 32) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("tiny GLM TP head did not gather finite full-vocabulary logits")
            reference = logits.clone()
            dist.broadcast(reference, src=0)
            if not torch.equal(reference, logits):
                raise RuntimeError("tiny GLM TP head outputs differ across ranks")
            del loaded
            dist.barrier()
        finally:
            if temporary is not None:
                temporary.cleanup()
        return {"loopback_rendezvous_passed": True, "tiny_glm_tp_load_passed": True,
                "tiny_glm_tp_head_forward_passed": True,
                "tiny_fixture_precision": "bfloat16", "gpu_forward_validated": False,
                "world_size": world, "rank": rank}
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--check-rendezvous", action="store_true")
    args = parser.parse_args(argv)
    if args.check_rendezvous:
        result = check_rendezvous()
        if result["rank"] == 0:
            print(json.dumps(result, sort_keys=True))
        return 0
    if args.check_runtime:
        print(json.dumps(runtime_capabilities(), sort_keys=True))
        return 0
    if args.config is None:
        parser.error("--config is required for capture")
    result = run_capture(json.loads(args.config.read_text(encoding="utf-8")))
    if result["rank"] == 0:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
