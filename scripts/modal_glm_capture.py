"""Plan, CPU-stage, and run one budget-bounded GLM-5.3 Modal capture.

The default mode is entirely offline. Cloud modes require fresh, explicit
operator verification of the current workspace's credits and hard billing caps.
No credentials, local checkpoint, repository directory, or chat archive is sent.
Only an explicitly selected pretokenized corpus and four allowlisted source
files are uploaded. The image pins below are provisional until the CPU runtime
check succeeds; no full-model GPU validation is claimed by this launcher.

Provider contracts checked 2026-10-02:
https://modal.com/docs/sdk/py/latest/App
https://modal.com/docs/sdk/py/latest/Image
https://modal.com/docs/sdk/py/latest/Volume
https://modal.com/docs/guide/budgets
https://modal.com/docs/guide/timeouts
https://modal.com/pricing
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tempfile
import time
import warnings


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ID = "zai-org/GLM-5.3"
REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
PROTOCOL = "glm-modal-capture-v1"
HARD_CAP_USD = Decimal("30")
MAX_PAID_BACKUP_USD = Decimal("5")
GPU_SECOND = Decimal("0.001261")
CPU_SECOND = Decimal("0.0000131")
GIB_SECOND = Decimal("0.00000222")
PRICE_DATE = "2026-10-02"
MAX_EVIDENCE_AGE_SECONDS = 900
MAX_CORPUS_BYTES = 32 * 1024 ** 2
MAX_METADATA_BYTES = 64 * 1024 ** 2
MAX_CAPTURE_BYTES = 8 * 1024 ** 3
MAX_CHUNK_BYTES = 64 * 1024 ** 2
VOLUME_FREE_BYTES = 1024 ** 4
MOUNT = "/capture/data"
SOURCE_FILES = (
    "scripts/modal_glm_capture.py", "tda/__init__.py",
    "tda/glm_capture_runtime.py", "tda/glm_capture_archive.py",
)
METADATA_NAMES = (
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
)
DEFAULT_IMAGE = {
    "base_image": "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    "python_version": "3.12",
    "torch_package": "torch==2.11.0+cu128",
    "torch_index_url": "https://download.pytorch.org/whl/cu128",
    "packages": ["transformers==5.15.0", "accelerate==1.15.0",
                 "kernels==0.16.0", "huggingface-hub==1.33.0",
                 "safetensors==0.8.0", "numpy==2.2.6"],
    "validated_on_glm_h200": False,
}


class CaptureLaunchError(ValueError):
    pass


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_bounded_json(path, limit):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise CaptureLaunchError(f"Expected a regular JSON file of at most {limit} bytes: {path}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise CaptureLaunchError("JSON file grew beyond its byte limit.")
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise CaptureLaunchError("Invalid JSON input.") from exc
    if not isinstance(value, dict):
        raise CaptureLaunchError("JSON inputs must contain an object.")
    return value, data


def _money(value, field):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CaptureLaunchError(f"{field} must be a finite nonnegative amount.") from exc
    if isinstance(value, bool) or not result.is_finite() or result < 0:
        raise CaptureLaunchError(f"{field} must be a finite nonnegative amount.")
    return result


def verify_budget_evidence(evidence, workspace, *, now=None, needs_gpu=False, required_volume_bytes=0):
    """Validate an operator receipt, never infer or change provider billing caps.

    The receipt is an attestation of the dashboard, not a programmatic billing
    audit. Its workspace must match the operator-verified active SDK profile.
    """
    now = now or datetime.now(timezone.utc)
    if evidence.get("protocol") != PROTOCOL or evidence.get("provider") != "modal":
        raise CaptureLaunchError("Budget evidence has an unsupported provider/protocol.")
    if not workspace or evidence.get("workspace") != workspace:
        raise CaptureLaunchError("Budget evidence must name the active Modal workspace.")
    for field in ("verified_by_operator", "active_profile_workspace_verified", "other_workloads_stopped"):
        if evidence.get(field) is not True:
            raise CaptureLaunchError(f"Fresh dashboard verification is required: {field}.")
    if evidence.get("billing_cycle") != now.strftime("%Y-%m"):
        raise CaptureLaunchError("Budget evidence belongs to a different billing cycle.")
    try:
        observed = datetime.fromisoformat(str(evidence["observed_at"]).replace("Z", "+00:00"))
        if observed.utcoffset() is None:
            raise ValueError("missing timezone")
        age = (now - observed.astimezone(timezone.utc)).total_seconds()
    except (KeyError, ValueError, TypeError) as exc:
        raise CaptureLaunchError("Budget evidence needs a timezone-aware observed_at.") from exc
    if not 0 <= age <= MAX_EVIDENCE_AGE_SECONDS:
        raise CaptureLaunchError("Budget evidence is stale or future-dated; verify the dashboard again.")
    used = _money(evidence.get("workspace_usage_usd"), "workspace_usage_usd")
    cap = _money(evidence.get("workspace_usage_limit_usd"), "workspace_usage_limit_usd")
    credit = _money(evidence.get("credits_remaining_usd"), "credits_remaining_usd")
    spend = _money(evidence.get("workspace_spend_limit_usd"), "workspace_spend_limit_usd")
    backup = _money(evidence.get("authorized_paid_backup_usd", 0), "authorized_paid_backup_usd")
    paid = _money(evidence.get("workspace_paid_spend_usd", 0), "workspace_paid_spend_usd")
    if backup > MAX_PAID_BACKUP_USD or spend > backup or cap <= 0:
        raise CaptureLaunchError("Cloud work requires the verified net spend limit to stay within the explicitly authorized backup (default $0; maximum $5).")
    if used >= cap or paid > backup or min(credit + backup - paid, HARD_CAP_USD + backup - used) <= 0:
        raise CaptureLaunchError("No verified trial compute remains.")
    free_volume = evidence.get("free_volume_bytes_remaining")
    if (isinstance(free_volume, bool) or not isinstance(free_volume, int) or
            free_volume < required_volume_bytes):
        raise CaptureLaunchError("Verified free Volume allowance cannot accommodate the requested phase.")
    if needs_gpu and (not isinstance(evidence.get("gpu_concurrency_limit"), int) or
                      isinstance(evidence.get("gpu_concurrency_limit"), bool) or
                      evidence["gpu_concurrency_limit"] < 8):
        raise CaptureLaunchError("Verified GPU concurrency must accommodate eight H200s.")
    # The provider's gross workspace cap can be higher. Its independent net
    # spend limit enforces the user's paid ceiling; this launcher additionally
    # refuses requests beyond the trial's $30 credit plus explicit backup.
    return min(credit + backup - paid, cap - used, HARD_CAP_USD + backup - used)


def budget_template(workspace=""):
    return {
        "protocol": PROTOCOL, "provider": "modal", "workspace": workspace,
        "billing_cycle": datetime.now(timezone.utc).strftime("%Y-%m"),
        "observed_at": None, "verified_by_operator": False,
        "active_profile_workspace_verified": False, "other_workloads_stopped": False,
        "workspace_usage_usd": None, "workspace_usage_limit_usd": None,
        "workspace_spend_limit_usd": None, "credits_remaining_usd": None,
        "authorized_paid_backup_usd": "0", "workspace_paid_spend_usd": None,
        "free_volume_bytes_remaining": None, "gpu_concurrency_limit": None,
        "note": "Fill only from the current Modal Usage & Billing/limits dashboard; this template is not evidence.",
    }


def resource_policy(mode, execution_seconds):
    if mode not in ("build", "stage", "capture") or not isinstance(execution_seconds, int) or execution_seconds < 1:
        raise CaptureLaunchError("Invalid cloud resource policy.")
    capture = mode == "capture"
    return {
        "gpu": "H200:8" if capture else None,
        "cpu": (16, 16) if capture else (2, 2),
        "memory": (131072, 131072) if capture else (8192, 8192),
        "min_containers": 0, "buffer_containers": 0, "max_containers": 1,
        "scaledown_window": 2, "single_use_containers": True, "retries": 0,
        "startup_timeout": 60, "timeout": execution_seconds,
        "nonpreemptible": False, "enable_memory_snapshot": False,
        "include_source": False, "serialized": True,
    }


def cost_ceiling(policy, execution_seconds):
    """Both startup and execution are billable; resource limits stop bursting."""
    if policy["cpu"][0] != policy["cpu"][1] or policy["memory"][0] != policy["memory"][1]:
        raise CaptureLaunchError("Cost bounds require hard CPU and RAM limits.")
    seconds = execution_seconds + policy["startup_timeout"] + policy["scaledown_window"]
    gpu_count = 8 if policy["gpu"] == "H200:8" else 0
    rate = GPU_SECOND * gpu_count + CPU_SECOND * policy["cpu"][1] + GIB_SECOND * Decimal(policy["memory"][1]) / 1024
    return rate * seconds


def capture_seconds(remaining_usd, *, reserve_usd=Decimal("2"), max_seconds=3000):
    remaining = _money(remaining_usd, "remaining_usd") - _money(reserve_usd, "reserve_usd")
    policy = resource_policy("capture", 1)
    rate = cost_ceiling(policy, 1) / Decimal(1 + policy["startup_timeout"] + policy["scaledown_window"])
    seconds = int((remaining / rate).to_integral_value(rounding=ROUND_FLOOR)) - 62
    seconds = min(seconds, int(max_seconds))
    if seconds < 120:
        raise CaptureLaunchError("Remaining verified credit cannot cover a bounded capture and save reserve.")
    return seconds


def _filename(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise CaptureLaunchError("Only root-level official checkpoint filenames are allowed.")
    if value in (".", "..") or PurePosixPath(value).is_absolute():
        raise CaptureLaunchError("Unsafe checkpoint filename.")
    return value


def load_stage_payload(metadata_dir):
    directory = Path(metadata_dir).resolve()
    manifest, _ = read_bounded_json(directory / "manifest.json", MAX_METADATA_BYTES)
    if manifest.get("id") != MODEL_ID or manifest.get("sha") != REVISION:
        raise CaptureLaunchError("Only the explicitly pinned official original FP8 checkpoint is supported.")
    config, _ = read_bounded_json(directory / "config.json", MAX_METADATA_BYTES)
    quant = config.get("quantization_config", {})
    if config.get("model_type") != "glm_moe_dsa" or quant.get("quant_method") != "fp8" or quant.get("fmt") != "e4m3":
        raise CaptureLaunchError("Metadata does not describe the native GLM FP8 source.")
    inventory = {}
    for row in manifest.get("siblings", []):
        name = row.get("rfilename")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", name) or
                PurePosixPath(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))):
            raise CaptureLaunchError("Unsafe manifest filename.")
        if name in inventory:
            raise CaptureLaunchError("Duplicate manifest filename.")
        inventory[name] = row
    index, _ = read_bounded_json(directory / "model.safetensors.index.json", MAX_METADATA_BYTES)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CaptureLaunchError("Missing checkpoint weight index.")
    shards = sorted(set(_filename(name) for name in weight_map.values()))
    if not all(name.endswith(".safetensors") for name in shards):
        raise CaptureLaunchError("Only safetensors source shards can be staged.")
    files = []
    for name in list(METADATA_NAMES) + shards:
        row = inventory.get(name)
        if row is None and name == "generation_config.json":
            continue
        if row is None:
            raise CaptureLaunchError(f"Pinned manifest is missing {name}.")
        size = row.get("size", row.get("lfs", {}).get("size"))
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CaptureLaunchError(f"Pinned byte size is missing for {name}.")
        digest = row.get("lfs", {}).get("sha256")
        if name in METADATA_NAMES:
            path = directory / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size != size or size > MAX_METADATA_BYTES:
                raise CaptureLaunchError(f"Pinned metadata must be present and byte-bounded: {name}.")
            data = path.read_bytes()
            local_digest = hashlib.sha256(data).hexdigest()
            if digest and digest != local_digest:
                raise CaptureLaunchError(f"Pinned metadata digest mismatch: {name}.")
            blob_id = row.get("blobId")
            if not digest and blob_id and hashlib.sha1(b"blob " + str(size).encode() + b"\0" + data).hexdigest() != blob_id:
                raise CaptureLaunchError(f"Pinned Git blob mismatch: {name}.")
            digest = local_digest
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CaptureLaunchError(f"Pinned SHA-256 is missing for {name}.")
        files.append({"name": name, "bytes": size, "sha256": digest})
    total = sum(row["bytes"] for row in files)
    if total + MAX_CAPTURE_BYTES + 1024 ** 3 >= VOLUME_FREE_BYTES:
        raise CaptureLaunchError("Checkpoint plus capture exceeds the free Volume envelope.")
    hashes = {row["name"]: row["sha256"] for row in files}
    identity = {
        "model_id": MODEL_ID, "revision": REVISION,
        "hidden_size": config["hidden_size"], "vocab_size": config["vocab_size"],
        "num_layers": config["num_hidden_layers"], "source_precision": "native_fp8_e4m3fn",
        "tokenizer_sha256": hashes["tokenizer.json"],
        "chat_template_sha256": hashes["chat_template.jinja"],
    }
    return {"protocol": PROTOCOL, "repository": MODEL_ID, "revision": REVISION,
            "files": files, "total_bytes": total, "manifest": manifest, "identity": identity}


def load_corpus(path, identity, *, workspace_root=ROOT):
    path = Path(path).resolve()
    allowed = (Path(workspace_root) / "invariants" / "out").resolve()
    if not path.is_relative_to(allowed) or path.suffix != ".json":
        raise CaptureLaunchError("Select one prepared .json corpus under this workspace's invariants/out.")
    corpus, data = read_bounded_json(path, MAX_CORPUS_BYTES)
    examples = corpus.get("examples")
    if not isinstance(examples, list) or not examples or len(examples) > 4096:
        raise CaptureLaunchError("Corpus must contain 1..4096 pretokenized examples.")
    from tda.glm_capture_corpus import validate_corpus
    try:
        validate_corpus(corpus, expected_revision=identity["revision"])
        if corpus["tokenizer_identity"]["vocab_size"] != identity["vocab_size"]:
            raise ValueError("Corpus vocabulary differs from the pinned output head")
        for key in ("tokenizer_sha256", "chat_template_sha256"):
            if corpus["tokenizer_identity"][key] != identity[key]:
                raise ValueError(f"Corpus {key} differs from the pinned teacher")
        if any(len(row["token_ids"]) > 1024 for row in examples):
            raise ValueError("First capture contexts must fit 1024 tokens without runtime truncation")
        # This launcher deliberately needs no local PyTorch installation. The
        # GPU adapter repeats this conservative native BF16 budget check.
        for offset in range(0, len(examples), 2):
            rows = sum(len(row["selected_positions"]) for row in examples[offset:offset + 2])
            payload_bytes = rows * (identity["hidden_size"] * 2 * 9 + 120) + 128 * 1024
            if offset == 0:
                payload_bytes += min(rows, 2) * (identity["vocab_size"] * 2 + 8)
            if payload_bytes > MAX_CHUNK_BYTES:
                raise ValueError("Selected-position batch exceeds the capture chunk budget")
    except (ValueError, KeyError) as exc:
        raise CaptureLaunchError(f"Invalid prepared corpus: {exc}") from exc
    return corpus, data, hashlib.sha256(data).hexdigest()


def validate_image_config(value):
    value = dict(value)
    if not isinstance(value.get("base_image"), str) or not re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value["base_image"]):
        raise CaptureLaunchError("Use a public explicit image tag or digest.")
    if value.get("python_version") != "3.12" or value.get("torch_index_url") != "https://download.pytorch.org/whl/cu128":
        raise CaptureLaunchError("The provisional image protocol requires Python3.12 and the official CUDA12.8 torch index.")
    packages = value.get("packages")
    all_packages = [value.get("torch_package")] + (packages if isinstance(packages, list) else [])
    if not isinstance(packages, list) or not packages or any(
        not isinstance(package, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", package) for package in all_packages
    ):
        raise CaptureLaunchError("All direct runtime dependencies need exact version pins.")
    if "transformers==5.15.0" not in packages or not value["torch_package"].startswith("torch==2.11.0+cu128"):
        raise CaptureLaunchError("Runtime image must preserve the checked GLM5.15/PyTorch2.11 CUDA12.8 source contract.")
    if "kernels==0.16.0" not in packages:
        raise CaptureLaunchError("Transformers5.15 finegrained FP8 requires compatible kernels0.16; kernels0.17 fails its runtime guard.")
    return value


def _sha256_file(path, deadline_epoch):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            if time.time() >= deadline_epoch:
                raise CaptureLaunchError("CPU staging deadline reached; partial files remain resumable.")
            block = handle.read(8 * 1024 ** 2)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stage_checkpoint(payload, destination, *, deadline_epoch, download_file, commit=lambda: None):
    """Cloud-only worker body; individual HF files use local_dir, without a second full cache."""
    if payload.get("protocol") != PROTOCOL or payload.get("revision") != REVISION or payload.get("repository") != MODEL_ID:
        raise CaptureLaunchError("Unsupported immutable staging request.")
    destination = Path(destination)
    final = destination / REVISION
    marker = final / "stage_receipt.json"
    if final.exists():
        receipt, _ = read_bounded_json(marker, 1024 ** 2)
        if receipt.get("files") != payload["files"] or receipt.get("revision") != REVISION:
            raise CaptureLaunchError("Existing staged checkpoint has a different receipt.")
        if any(not (final / row["name"]).is_file() or (final / row["name"]).stat().st_size != row["bytes"] for row in payload["files"]):
            raise CaptureLaunchError("Previously staged checkpoint is incomplete.")
        return receipt
    partial = destination / (REVISION + ".incomplete")
    partial.mkdir(parents=True, exist_ok=True)
    checked = []
    for row in payload["files"]:
        name = _filename(row["name"])
        if name not in METADATA_NAMES and not name.endswith(".safetensors"):
            raise CaptureLaunchError("Staging cannot download model source code or unrelated files.")
        if time.time() >= deadline_epoch:
            raise CaptureLaunchError("CPU staging deadline reached.")
        downloaded = Path(download_file(repo_id=MODEL_ID, revision=REVISION, filename=name,
                                        local_dir=str(partial), token=False))
        expected_path = partial / name
        if downloaded.resolve() != expected_path.resolve() or expected_path.is_symlink():
            raise CaptureLaunchError("Download escaped the single checkpoint directory.")
        if expected_path.stat().st_size != row["bytes"] or _sha256_file(expected_path, deadline_epoch) != row["sha256"]:
            raise CaptureLaunchError(f"Downloaded source failed its pinned size/digest check: {name}.")
        checked.append(row)
        atomic_write(partial / "stage_progress.json", _json({"revision": REVISION, "files": checked}))
        commit()
    receipt = {"protocol": PROTOCOL, "revision": REVISION, "repository": MODEL_ID,
               "files": checked, "total_bytes": sum(row["bytes"] for row in checked),
               "identity": payload["identity"], "all_sha256_verified": True,
               "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_write(partial / "manifest.json", _json(payload["manifest"]))
    atomic_write(partial / "stage_receipt.json", _json(receipt))
    os.replace(partial, final)
    commit()
    return receipt


def runtime_config(payload, *, corpus_path, output_dir, deadline_epoch):
    return {
        "checkpoint_dir": MOUNT + "/checkpoints/" + REVISION,
        "corpus_path": corpus_path, "output_dir": output_dir,
        "deadline_epoch": float(deadline_epoch), "save_reserve_seconds": 45,
        "selected_layers": [3, 12, 24, 36, 48, 60, 77], "target_layer": 3,
        "target_expert_ids": list(range(8)), "batch_size": 2, "max_seq_len": 1024,
        "spotcheck_rows": 2, "max_total_bytes": MAX_CAPTURE_BYTES,
        "max_chunk_bytes": MAX_CHUNK_BYTES, "identity": payload["identity"],
    }


def run_runtime_process(config_path, *, timeout_seconds, cwd="/opt/capture", check_rendezvous=False):
    """No shell interpolation and no torch elastic restart; terminate the whole worker group."""
    # --standalone's elastic rendezvous advertises the container hostname,
    # which need not resolve when the capture is offline. All ranks are local.
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--node-rank=0",
               "--rdzv-backend=static", "--rdzv-conf=timeout=45",
               "--master-addr=127.0.0.1", "--master-port=29500",
               "--nproc-per-node=8", "--max-restarts=0", "-m", "tda.glm_capture_runtime"]
    command += ["--check-rendezvous"] if check_rendezvous else ["--config", str(config_path)]
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        raise CaptureLaunchError("GPU worker reached its wall-clock limit; completed chunks remain saved.")


def upload_sources(image, *, root=ROOT):
    image = image.env({"PYTHONPATH": "/opt/capture:/opt/capture/scripts"})
    for relative in SOURCE_FILES:
        path = (Path(root) / relative).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 ** 2:
            raise CaptureLaunchError(f"Missing or unsafe explicit runtime source: {relative}.")
        image = image.add_local_file(str(path), "/opt/capture/" + relative, copy=False)
    return image


def source_fingerprint(root=ROOT):
    return hashlib.sha256(_json({relative: hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest()
                                 for relative in SOURCE_FILES})).hexdigest()


def make_plan(payload, *, corpus_digest=None, remaining_usd=Decimal("30")):
    seconds = capture_seconds(remaining_usd)
    policy = resource_policy("capture", seconds)
    return {
        "protocol": PROTOCOL, "mode": "plan", "cloud_started": False,
        "repository": MODEL_ID, "revision": REVISION,
        "checkpoint_bytes": payload["total_bytes"], "corpus_sha256": corpus_digest,
        "gross_monthly_cap_usd": str(HARD_CAP_USD), "assumed_remaining_usd": str(remaining_usd),
        "remaining_credit_verified": False, "prices_checked_on": PRICE_DATE,
        "capture_execution_seconds_ceiling": seconds,
        "capture_resource_cost_ceiling_usd": str(cost_ceiling(policy, seconds)),
        "resource_policy": policy, "max_capture_bytes": MAX_CAPTURE_BYTES,
        "upload_source_allowlist": list(SOURCE_FILES), "image": DEFAULT_IMAGE,
        "required_sequence": ["build (CPU capability check)", "stage (CPU checkpoint download)",
                              "refresh dashboard evidence", "capture (one eight-H200 input)"],
        "runtime_validation": "Provisional; native FP8 plus TP8 capture must pass cloud CPU checks and GPU load.",
        "budget_evidence_template": budget_template(),
    }


def launch_cloud(mode, payload, *, evidence, workspace, volume_name, execution_seconds,
                 corpus_bytes=None, image_config=None, root=ROOT):
    """The sole cloud boundary. Its checks precede importing/initializing Modal."""
    warnings.filterwarnings("ignore", message="Your Modal token .* expires in .*")
    required_volume = (payload["total_bytes"] if mode == "stage" else 0) + MAX_CAPTURE_BYTES + 1024 ** 3
    remaining = verify_budget_evidence(evidence, workspace, needs_gpu=mode == "capture",
                                       required_volume_bytes=required_volume)
    policy = resource_policy(mode, execution_seconds)
    if cost_ceiling(policy, execution_seconds) + Decimal("2") > remaining:
        raise CaptureLaunchError("Requested cloud duration exceeds verified remaining credit and reserve.")
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,63}", volume_name):
        raise CaptureLaunchError("Use a simple dedicated Modal Volume name.")
    config = validate_image_config(image_config or DEFAULT_IMAGE)
    if mode == "capture" and (not corpus_bytes or len(corpus_bytes) > MAX_CORPUS_BYTES):
        raise CaptureLaunchError("A bounded, verified pretokenized corpus is required before any GPU call.")
    try:
        modal = importlib.import_module("modal")
    except ImportError as exc:
        raise CaptureLaunchError("Use an isolated Modal client environment; the project environment is not modified.") from exc
    app = modal.App("glm-credit-capture", include_source=False)
    volume = modal.Volume.from_name(volume_name, create_if_missing=mode in ("stage", "build"))
    image_digest = hashlib.sha256(_json(config)).hexdigest()
    sources_digest = source_fingerprint(root)
    if mode == "capture":
        # Refuse before creating a GPU Function if the CPU preflight is absent.
        data = bytearray()
        for chunk in volume.read_file("/build_receipt.json"):
            data.extend(chunk)
            if len(data) > 1024 ** 2:
                raise CaptureLaunchError("CPU build receipt exceeded its byte limit.")
        built = json.loads(data)
        if (built.get("cpu_check_passed") is not True or built.get("image_config_sha256") != image_digest or
                built.get("source_sha256") != sources_digest):
            raise CaptureLaunchError("The current image and exact runtime sources require a successful CPU build check first.")
    if mode == "stage":
        image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface-hub==1.33.0")
    else:
        image = modal.Image.from_registry(config["base_image"], add_python=config["python_version"])
        image = image.pip_install(config["torch_package"], index_url=config["torch_index_url"])
        image = image.pip_install(*config["packages"])
    image = image.env({"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "45",
                       "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
                       "HF_HOME": MOUNT + "/kernel-cache", "HF_HUB_CACHE": MOUNT + "/kernel-cache/hub",
                       "KERNELS_CACHE": MOUNT + "/kernel-cache/hub", "GLOO_SOCKET_IFNAME": "lo",
                       "NCCL_SOCKET_IFNAME": "=lo", "NCCL_SOCKET_FAMILY": "AF_INET"})
    if mode == "capture":
        image = image.env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        policy["block_network"] = True
    # copy=False local files must be the final image layers in Modal 1.6.
    image = upload_sources(image, root=root)

    @app.function(image=image, volumes={MOUNT: volume}, **policy)
    def worker(request, input_bytes=None):
        start = time.time()
        if mode == "stage":
            from huggingface_hub import hf_hub_download
            return stage_checkpoint(request, MOUNT + "/checkpoints", deadline_epoch=start + execution_seconds - 15,
                                    download_file=hf_hub_download, commit=volume.commit)
        if mode == "build":
            import importlib.metadata
            command = [sys.executable, "-m", "tda.glm_capture_runtime", "--check-runtime"]
            checked = subprocess.run(command, cwd="/opt/capture", timeout=execution_seconds - 15,
                                     check=False, capture_output=True, text=True)
            if checked.returncode:
                raise CaptureLaunchError("Online CPU runtime check failed: " + checked.stderr[-6000:])
            online_info = json.loads(checked.stdout.strip().splitlines()[-1])
            offline_env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            offline = subprocess.run(command, cwd="/opt/capture", env=offline_env,
                timeout=min(120, max(1, execution_seconds - (time.time() - start) - 15)),
                check=False, capture_output=True, text=True)
            if offline.returncode:
                raise CaptureLaunchError("Offline CPU runtime check failed: " + offline.stderr[-6000:])
            offline_info = json.loads(offline.stdout.strip().splitlines()[-1])
            if online_info["fp8_kernel"] != offline_info["fp8_kernel"]:
                raise CaptureLaunchError("Fresh offline CPU process used different native FP8 kernel artifacts.")
            rendezvous_code = run_runtime_process(None, timeout_seconds=min(120, max(1, execution_seconds - (time.time() - start) - 15)),
                                                 check_rendezvous=True)
            if rendezvous_code != 0:
                raise CaptureLaunchError("Eight-rank loopback CPU rendezvous preflight failed.")
            result = {"protocol": PROTOCOL, "cpu_check_passed": checked.returncode == 0,
                      "loopback_rendezvous_world_size": 8, "loopback_rendezvous_passed": True,
                      "tiny_glm_tp_load_passed": True,
                      "tiny_glm_tp_head_forward_passed": True,
                      "offline_runtime_check_passed": True, "fp8_kernel": offline_info["fp8_kernel"],
                      "versions": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
                      "image_config_sha256": image_digest, "source_sha256": sources_digest,
                      "glm_gpu_capture_validated": False}
            atomic_write(Path(MOUNT) / "build_receipt.json", _json(result))
            volume.commit()
            return result
        revision_dir = Path(MOUNT) / "checkpoints" / REVISION
        receipt, _ = read_bounded_json(revision_dir / "stage_receipt.json", 1024 ** 2)
        if receipt.get("files") != request["files"] or receipt.get("identity") != request["identity"] or receipt.get("all_sha256_verified") is not True:
            raise CaptureLaunchError("GPU launch requires the complete verified native FP8 stage receipt.")
        digest = hashlib.sha256(input_bytes).hexdigest()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + digest[:12]
        run_dir = Path(MOUNT) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        corpus_path = run_dir / "corpus.json"
        atomic_write(corpus_path, input_bytes)
        runtime = runtime_config(request, corpus_path=str(corpus_path), output_dir=str(run_dir / "archive"),
                                 deadline_epoch=start + execution_seconds - 15)
        atomic_write(run_dir / "runtime.json", _json(runtime))
        volume.commit()
        try:
            code = run_runtime_process(run_dir / "runtime.json", timeout_seconds=max(1, execution_seconds - (time.time() - start) - 15))
            return {"protocol": PROTOCOL, "run_id": run_id, "return_code": code,
                    "output_volume_path": "/runs/" + run_id, "corpus_sha256": digest}
        finally:
            volume.commit()

    with modal.enable_output():
        with app.run():
            # Building can take long enough to make the operator receipt stale.
            verify_budget_evidence(evidence, workspace, needs_gpu=mode == "capture",
                                   required_volume_bytes=required_volume)
            return worker.remote(payload, corpus_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "build", "stage", "capture"), default="plan")
    parser.add_argument("--metadata", type=Path, default=ROOT / "invariants/out/glm_5_3_preflight")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--budget-evidence", type=Path)
    parser.add_argument("--workspace", default="")
    parser.add_argument("--volume-name", default="glm-5-3-credit-capture")
    parser.add_argument("--image-config", type=Path)
    parser.add_argument("--seconds", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-template", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.budget_template:
            atomic_write(args.budget_template, _json(budget_template(args.workspace)))
        payload = load_stage_payload(args.metadata)
        corpus_bytes = None
        digest = None
        if args.corpus:
            _, corpus_bytes, digest = load_corpus(args.corpus, payload["identity"])
        if args.mode == "plan":
            result = make_plan(payload, corpus_digest=digest)
        else:
            if not args.budget_evidence:
                raise CaptureLaunchError("Cloud modes need fresh dashboard evidence of the existing $30 free-credit constraint.")
            evidence, _ = read_bounded_json(args.budget_evidence, 64 * 1024)
            remaining = verify_budget_evidence(evidence, args.workspace, needs_gpu=args.mode == "capture")
            seconds = args.seconds if args.seconds is not None else (
                capture_seconds(remaining) if args.mode == "capture" else 3600 if args.mode == "stage" else 600)
            image_config = read_bounded_json(args.image_config, 64 * 1024)[0] if args.image_config else DEFAULT_IMAGE
            result = launch_cloud(args.mode, payload, evidence=evidence, workspace=args.workspace,
                                  volume_name=args.volume_name, execution_seconds=seconds,
                                  corpus_bytes=corpus_bytes, image_config=image_config)
        if args.output:
            atomic_write(args.output, _json(result))
        print(_json(result).decode("utf-8"), end="")
        return 2 if args.mode == "capture" and result.get("return_code") != 0 else 0
    except (CaptureLaunchError, OSError, KeyError) as exc:
        print(f"[GLM capture] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
