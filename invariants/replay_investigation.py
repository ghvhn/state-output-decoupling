"""Matched archived-context reruns for testing indirect probe interventions.

Reconstructed contexts are explicitly partial. Rerun effects describe measured
generation states and do not establish semantic truth, motive, or experience.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import torch

from invariants import engine
from invariants.agentic_engine import generate_agentic_text
from invariants.config import AgenticConfig
from invariants.engine import _steer_handles, steer_band_layers
from invariants.indirect_steering import (IndirectSteering, _controls, _is_inline_composition_token,
                                         _objectives, compose_direction, definition_fingerprints)
from invariants.generation_budget import check_context_budget

SOURCE = "controlled_replay_investigation"
FINAL_PREFIX = "__replay_final__"


def _get(record, key, default=None):
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, default=str, allow_nan=False).encode()).hexdigest()


def select_archive_cases(records, limit=4, context_turns=8, scope=None):
    """Select recent paired turns, preserving only genuine preceding context."""
    if int(limit) < 1 or int(context_turns) < 1:
        raise ValueError("limit and context_turns must be positive")
    histories, pending, cases = {}, {}, []
    for index, record in enumerate(records):
        record_scope = _get(record, "scope", "default")
        session = _get(record, "session_id", "")
        role, text = _get(record, "role"), _get(record, "text", "")
        if (not session or (scope is not None and record_scope != scope)
                or _get(record, "kind", "turn") != "turn"
                or role not in {"system", "user", "assistant"} or not isinstance(text, str) or not text.strip()):
            continue
        key = (record_scope, session)
        history = histories.setdefault(key, [])
        record_id = _get(record, "record_id") or f"archive-index-{index}"
        if role == "assistant" and key in pending:
            user = pending.pop(key)
            excerpt = history[-int(context_turns):]
            systems = [item for item in history if item["role"] == "system"]
            if systems and systems[-1] not in excerpt:
                excerpt = [systems[-1]] + excerpt
            # A clipped excerpt may begin with an orphan assistant response.
            first_dialogue = 1 if excerpt and excerpt[0]["role"] == "system" else 0
            while len(excerpt) > first_dialogue and excerpt[first_dialogue]["role"] == "assistant":
                excerpt = excerpt[:first_dialogue] + excerpt[first_dialogue + 1:]
            messages = [{"role": item["role"], "content": item["content"]} for item in excerpt]
            if messages and messages[-1]["role"] == "user":
                cases.append({"case_id": _hash([record_scope, session, record_id, user["id"]])[:24],
                    "source_record_id": record_id, "user_record_id": user["id"],
                    "session_id": session, "scope": record_scope,
                    "timestamp": _get(record, "timestamp"), "prompt_messages": messages,
                    "archived_reply": text, "context_quality": "partial_reconstruction",
                    "context_truncated": len(excerpt) < len(history),
                    "original_messages_omitted": len(history) - len(excerpt),
                    "context_record_ids": [item["id"] for item in excerpt],
                    "reconstruction_notes": "Only saved same-session/scope turns are available; "
                    "original system injections, tool state, prompt tokens, and runtime hooks are not reconstructed."})
        history.append({"role": role, "content": text, "id": record_id})
        if role == "user":
            pending[key] = history[-1]
    return cases[-int(limit):]


def model_identity(model):
    config = model.model.config
    try:
        dtype = str(next(model.model.parameters()).dtype)
    except StopIteration:
        dtype = str(getattr(config, "torch_dtype", None))
    return {"model": getattr(config, "_name_or_path", ""),
            "layers": int(model.n_layers), "width": int(model.d_model),
            "quantized_4bit": bool(getattr(model.model, "is_loaded_in_4bit", False)), "dtype": dtype}


def _model_hash(model):
    config = model.model.config
    definition = {"identity": model_identity(model),
                  "config": config.to_dict() if hasattr(config, "to_dict") else vars(config),
                  "checkpoint_commit": getattr(config, "_commit_hash", None)}
    parameters = list(model.model.named_parameters())
    definition["parameters"] = [(name, list(value.shape), str(value.dtype)) for name, value in parameters]
    # Tiny test/research models can be identified exactly without transferring
    # an entire production checkpoint from the GPU merely to compute a hash.
    if sum(value.numel() for _, value in parameters) <= 1_000_000:
        digest = hashlib.sha256()
        for name, value in parameters:
            digest.update(name.encode())
            digest.update(value.detach().cpu().float().contiguous().numpy().tobytes())
        definition["small_model_weights_sha256"] = digest.hexdigest()
    definition["weight_hash_scope"] = "full small-model parameters; otherwise checkpoint identity/config/parameter metadata"
    return _hash(definition), definition


def _write_json(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _hook_ids(model):
    return {(id(module), attribute, key) for module in model.model.modules()
            for attribute in ("_forward_hooks", "_forward_pre_hooks")
            for key in getattr(module, attribute, {})}


def _remove_new_hooks(model, original):
    for module in model.model.modules():
        for attribute in ("_forward_hooks", "_forward_pre_hooks"):
            hooks = getattr(module, attribute, {})
            for key in list(hooks):
                if (id(module), attribute, key) not in original:
                    del hooks[key]


@contextmanager
def _preserve_rng():
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


@contextmanager
def _isolated_steer_telemetry():
    previous = (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count,
                engine._steer_cap_fraction, engine._steer_band)
    engine._steer_attempts = deque(maxlen=previous[0].maxlen)
    engine._steer_apply_count = engine._steer_clip_count = 0
    try:
        yield
    finally:
        (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count,
         engine._steer_cap_fraction, engine._steer_band) = previous


def _seed(value):
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(value)


def _prepare_helpers(probes, objectives, controls, layers, width, max_helpers):
    from invariants.other_probes import is_other_probe
    owner = IndirectSteering()
    owner._refresh(probes)
    blocked = set().union(*(owner._blocked_target(target, probes) for target in objectives))
    explicit = controls is not None
    role_rank = {"thinking": 0, "execution": 0, "perception": 1, "unclassified": 2}
    names = sorted(probes, key=lambda name: (role_rank.get(probes[name].get("role", "unclassified"), 2), name))
    candidates = _controls(controls) if explicit else [{"name": name, "sign": None} for name in names]
    selected, rejected = [], {}
    for candidate in candidates:
        name = candidate["name"]
        if name not in probes and explicit and _is_inline_composition_token(name, probes):
            probes[name] = compose_direction(probes, name, blocked=blocked, layers=layers)
            owner._refresh(probes)
            blocked = set().union(*(owner._blocked_target(target, probes) for target in objectives))
        if name not in probes:
            raise ValueError(f"Unknown helper probe: {name}")
        if name in blocked:
            if explicit:
                raise ValueError(f"Helper {name} is an objective or its direct proxy")
            rejected[name] = "objective, alias/inverse, or target-derived composite"
            continue
        payload = probes[name]
        if payload.get("released") or is_other_probe(payload):
            rejected[name] = "released or observational-only helper"
            continue
        directions = {int(layer): torch.as_tensor(value).detach().cpu().float().reshape(-1)
                      for layer, value in payload.get("direction", {}).items()}
        if any(layer not in directions or directions[layer].numel() != width
               or not bool(torch.isfinite(directions[layer]).all()) or not bool(directions[layer].norm() > 0)
               for layer in layers):
            rejected[name] = "incompatible/missing intervention direction"
            continue
        if any(owner._is_alias(name, helper["name"], probes) for helper in selected):
            rejected[name] = "duplicate helper axis"
            continue
        if len(selected) < max_helpers:
            helper_entry = {**candidate, "vectors": {layer: directions[layer] / directions[layer].norm() for layer in layers}}
            if payload.get("prior_weight") is not None:
                helper_entry["prior_weight"] = float(payload["prior_weight"])
            if isinstance(payload.get("layer_weights"), dict):
                helper_entry["layer_weights"] = dict(payload["layer_weights"])
            selected.append(helper_entry)
    return selected, rejected


def _read_trials(path, manifest):
    rows, keys = [], set()
    planned = {(item["case_id"], item["arm"], item["repeat"])
               for item in manifest["trial_plan"]}
    if path.exists():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Incomplete trial JSONL at line {number}; preserve and repair before resuming") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Incompatible completed trial at line {number}: expected an object")
            case_id, arm, repeat = row.get("case_id"), row.get("arm"), row.get("repeat")
            if (not isinstance(case_id, str) or not isinstance(arm, str)
                    or isinstance(repeat, bool) or not isinstance(repeat, int)):
                raise ValueError(f"Incompatible completed trial identity at line {number}")
            key = (case_id, arm, repeat)
            if (row.get("fingerprint") != manifest["fingerprint"]
                    or row.get("source") != SOURCE or key not in planned or key in keys):
                raise ValueError(f"Incompatible or duplicate completed trial at line {number}")
            expected = manifest["arm_specs"][arm]
            if (row.get("helper") != expected["helper"]
                    or isinstance(row.get("sign"), bool) or not isinstance(row.get("sign"), int)
                    or row["sign"] != expected["sign"]
                    or not isinstance(row.get("is_control"), bool)
                    or row["is_control"] != expected["is_control"]):
                raise ValueError(f"Incompatible completed trial arm metadata at line {number}")
            prompt_hash = manifest["prepared_prompts"][case_id]["prompt_sha256"]
            telemetry = row.get("telemetry")
            if not isinstance(telemetry, dict):
                raise ValueError(f"Incompatible completed trial telemetry at line {number}")
            capture = telemetry.get("generation_probe_capture") or {}
            if (not isinstance(capture, dict) or row.get("prompt_sha256") != prompt_hash
                    or capture.get("prompt_sha256") not in (None, prompt_hash)
                    or capture.get("source", "live_assistant_prediction") != "live_assistant_prediction"):
                raise ValueError(f"Incompatible completed trial prompt or source at line {number}")
            rows.append(row)
            keys.add(key)
    return rows, keys


def _summary(manifest, trials):
    baselines = {(row["case_id"], row["repeat"]): row for row in trials if row["sign"] == 0}
    effects, negative_controls = [], []
    grouped, missing_readouts = {}, set()
    for row in trials:
        reference = baselines.get((row["case_id"], row["repeat"]))
        if not row["sign"] or reference is None or reference["prompt_sha256"] != row["prompt_sha256"]:
            continue
        for target, desired in manifest["settings"]["objectives"].items():
            metric_name = FINAL_PREFIX + target
            active = row["telemetry"].get("generation_probe_readings", {}).get(metric_name, {})
            baseline = reference["telemetry"].get("generation_probe_readings", {}).get(metric_name, {})
            if "mean_token_score" not in active or "mean_token_score" not in baseline:
                missing_readouts.add(target)
                continue
            delta = active["mean_token_score"] - baseline["mean_token_score"]
            if math.isfinite(delta):
                grouped.setdefault((target, row["helper"], row["sign"], row["is_control"]), {}).setdefault(row["case_id"], []).append(delta)
    for (target, helper, sign, is_control), values in grouped.items():
        case_deltas = [{"case_id": case_id, "mean_delta": statistics.mean(deltas), "n_repeats": len(deltas)}
                       for case_id, deltas in sorted(values.items())]
        deltas = [item["mean_delta"] for item in case_deltas]
        mean = statistics.mean(deltas)
        error = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else None
        effect = {"target": target, "helper": helper, "sign": sign, "n_cases": len(deltas),
                  "n": sum(len(items) for items in values.values()), "mean_delta": mean,
                  "metric": "final_layer_mean_token_score", "readout_layer": manifest["model_identity"]["layers"] - 1,
                  "standard_error": error, "stderr": error, "case_deltas": case_deltas,
                  "desired_mean_delta": manifest["settings"]["objectives"][target] * mean}
        (negative_controls if is_control else effects).append(effect)
    return {"source": SOURCE, "fingerprint": manifest["fingerprint"],
            "model_identity": manifest["model_identity"], "model_sha256": manifest["model_sha256"],
            "definition_fingerprints": manifest["definition_fingerprints"], "settings": manifest["settings"],
            "archive_provenance": manifest["cases"], "case_ids": [case["case_id"] for case in manifest["cases"]],
            "effects": effects, "negative_controls": negative_controls,
            "unavailable_final_readouts": sorted(missing_readouts), "completed_trials": len(trials),
            "expected_trials": len(manifest["trial_plan"]),
            "interpretation": "Matched controlled rerun effects on mean token cosine scores. "
            "Output tokens/lengths can differ; repeated greedy trials do not add independent cases. "
            "These results do not establish semantic truth, intent, empathy, coherence, or experience."}


def run_investigation(model, probes, objectives, controls=None, *, cases, output_dir,
                      alpha=.25, cap_fraction=.02, layers=None, seed=0,
                      max_new_tokens=64, max_helpers=3, repeats=1, progress=print,
                      resume=True, random_control=True):
    """Run isolated matched baseline and signed helper arms; never train live state."""
    if _hook_ids(model):
        raise ValueError("Replay investigation requires a fresh model without active PyTorch forward hooks")
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(cap_fraction) or cap_fraction <= 0:
        raise ValueError("alpha and cap_fraction must be positive finite numbers")
    if min(int(max_new_tokens), int(max_helpers), int(repeats)) < 1:
        raise ValueError("Token, helper, and repeat counts must be positive")
    probes = dict(probes)
    objectives = _objectives(objectives)
    if any(target not in probes for target in objectives):
        raise ValueError("Every objective must name an available probe")
    cases = list(cases)
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Supply nonempty cases with distinct case_id values")
    layers = list(layers) if layers is not None else steer_band_layers(model.n_layers, .4, .7)
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= layer < model.n_layers - 1 for layer in layers):
        raise ValueError("Use distinct intervention layers below the final normalized readout endpoint")
    helpers, rejected = _prepare_helpers(probes, objectives, controls, layers, model.d_model, int(max_helpers))
    if not helpers:
        raise ValueError("No eligible independent helper direction is available")
    measured = {name: probes[name] for name in dict.fromkeys([*objectives, *(helper["name"] for helper in helpers)])}
    for name, payload in measured.items():
        directions = payload.get("direction", {})
        if not directions or any(not 0 <= int(layer) < model.n_layers
                or torch.as_tensor(value).numel() != model.d_model
                or not bool(torch.isfinite(torch.as_tensor(value)).all())
                or not bool(torch.as_tensor(value).float().norm() > 0)
                for layer, value in directions.items()):
            raise ValueError(f"Incompatible readout probe: {name}")
    capture_probes = dict(measured)
    for target in objectives:
        if FINAL_PREFIX + target in probes:
            raise ValueError(f"Probe name collides with replay readout namespace: {FINAL_PREFIX + target}")
        directions = {int(layer): value for layer, value in probes[target]["direction"].items()}
        if model.n_layers - 1 in directions:
            capture_probes[FINAL_PREFIX + target] = {"direction": {model.n_layers - 1: directions[model.n_layers - 1]},
                                                     "token_mode": "last"}
    arms = [{"arm": "baseline", "helper": None, "sign": 0, "is_control": False, "vectors": {}}]
    for helper in helpers:
        for sign in ((helper["sign"],) if helper["sign"] is not None else (1, -1)):
            arms.append({"arm": f"{helper['name']}:{'+' if sign > 0 else '-'}", "helper": helper["name"],
                         "sign": sign, "is_control": False, "vectors": helper["vectors"]})
    if random_control:
        generator = torch.Generator().manual_seed(int(seed) + 104729)
        vectors = {}
        for layer in layers:
            vector = torch.randn(model.d_model, generator=generator)
            vectors[layer] = vector / vector.norm()
        arms += [{"arm": f"random_control:{label}", "helper": "__random_control__", "sign": sign,
                  "is_control": True, "vectors": vectors} for label, sign in (("+", 1), ("-", -1))]
    prepared = {}
    for case in cases:
        messages = case["prompt_messages"]
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("Each replay prompt must end with the actual preceding user turn")
        text = model.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = model.tok(text, return_tensors="pt", add_special_tokens=False)
        check_context_budget(model.model, encoded, reserve=int(max_new_tokens))
        token_ids = encoded["input_ids"].detach().cpu().tolist()
        prepared[case["case_id"]] = {"text": text, "token_ids": token_ids,
                                    "prompt_sha256": _hash(token_ids)}
    settings = {"objectives": objectives, "controls": [{"name": helper["name"], "sign": helper["sign"]} for helper in helpers],
                "alpha": alpha, "cap_fraction": cap_fraction, "layers": layers, "seed": int(seed),
                "max_new_tokens": int(max_new_tokens), "max_helpers": int(max_helpers), "repeats": int(repeats),
                "random_control": bool(random_control), "do_sample": False, "generation_channel": "reader",
                "max_tool_calls": 0, "source": SOURCE, "base_model_only": True,
                "activation_memory_enabled": False, "repetition_penalty": 1.,
                "effect_metric": "final_layer_mean_token_score", "readout_layer": model.n_layers - 1,
                "final_readout_available": {target: FINAL_PREFIX + target in capture_probes for target in objectives},
                "injection_scope": "all tokens at selected block outputs, including prompt prefill"}
    trial_plan = []
    for repeat in range(int(repeats)):
        for index, case in enumerate(cases):
            offset = (index + repeat) % len(arms)
            ordered = arms[offset:] + arms[:offset]
            if (index + repeat) % 2:
                ordered = list(reversed(ordered))
            trial_plan += [{"case_id": case["case_id"], "arm": arm["arm"], "repeat": repeat} for arm in ordered]
    model_sha, model_definition = _model_hash(model)
    scope = {"version": 2, "source": SOURCE, "settings": settings, "cases": cases,
             "model_identity": model_identity(model), "model_sha256": model_sha, "model_definition": model_definition,
             "definition_fingerprints": definition_fingerprints(measured),
             "arm_vector_sha256": {arm["arm"]: definition_fingerprints({"arm": {"direction": arm["vectors"]}})["arm"] for arm in arms},
             "prepared_prompts": prepared, "trial_plan": trial_plan,
             "arm_specs": {arm["arm"]: {key: arm[key] for key in ("helper", "sign", "is_control")} for arm in arms},
             "rejected_helpers": rejected,
             "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                               (Path(__file__), Path(__file__).with_name("engine.py"),
                                *(Path(__file__).with_name(name) for name in
                                  ("agentic_engine.py", "generation_probe_capture.py", "indirect_steering.py",
                                   "other_probes.py", "config.py", "generation_budget.py")))}}
    manifest = {**scope, "fingerprint": _hash(scope), "created_at": datetime.now(timezone.utc).isoformat()}
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, trials_path = output / "run_manifest.json", output / "trials.jsonl"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or old["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Existing replay manifest differs; choose a new output directory")
        manifest = old
    elif trials_path.exists():
        raise ValueError("Cannot resume trials without their original manifest")
    else:
        _write_json(manifest_path, manifest)
        torch.save({arm["arm"]: arm["vectors"] for arm in arms}, output / "intervention_vectors.pt")
    trials, completed = _read_trials(trials_path, manifest)
    arm_lookup = {arm["arm"]: arm for arm in arms}
    config = AgenticConfig(base_model_only=True, activation_memory_enabled=False,
        activation_memory_write_enabled=False, activation_memory_autosave=False,
        cache_enabled=False, cache_write_enabled=False, use_expert_vectors=False,
        synthesis_enabled=False, repetition_penalty=1.)
    def announce(message):
        if progress is not None:
            print(message, flush=True) if progress is print else progress(message)
    announce(f"Replay investigation: {len(completed)}/{len(trial_plan)} trials complete; output {output}")
    with _preserve_rng(), _isolated_steer_telemetry(), trials_path.open("a", encoding="utf-8", buffering=1) as stream:
        for item in trial_plan:
            key = (item["case_id"], item["arm"], item["repeat"])
            if key in completed:
                continue
            arm, prompt = arm_lookup[item["arm"]], prepared[item["case_id"]]
            _seed(int(seed) + item["repeat"])
            original = _hook_ids(model)
            if original:
                raise ValueError("Unexpected active hooks appeared between replay trials")
            started, handles = time.perf_counter(), []
            before_steering = engine.steer_telemetry_stats()
            announce(f"[{len(trials)+1}/{len(trial_plan)}] {item['case_id']} {item['arm']} repeat {item['repeat']} starting")
            try:
                if arm["sign"]:
                    handles = _steer_handles(model, arm["vectors"], layers, arm["sign"] * alpha, cap_fraction=cap_fraction)
                reply, telemetry = generate_agentic_text(model, instruction=prompt["text"], pre_formatted=True,
                    vecs={}, config=config, max_new_tokens=int(max_new_tokens), max_tool_calls=0,
                    generation_channel="reader", return_telemetry=True, generation_probes=capture_probes, chatty_log=False)
            finally:
                for handle in handles:
                    handle.remove()
                _remove_new_hooks(model, original)
            capture = telemetry.get("generation_probe_capture", {})
            if capture.get("prompt_sha256") not in (None, prompt["prompt_sha256"]):
                raise ValueError("Live capture prompt hash differs from the matched replay prompt")
            if telemetry.get("activation_memory"):
                raise ValueError("Unexpected activation-memory activity during isolated replay")
            for profile in telemetry.get("generation_probe_readings", {}).values():
                if profile.get("by_token"):
                    profile.setdefault("first_token_score", profile["by_token"][0])
            after_steering = engine.steer_telemetry_stats()
            row = {**item, "source": SOURCE, "fingerprint": manifest["fingerprint"],
                   "helper": arm["helper"], "sign": arm["sign"], "is_control": arm["is_control"],
                   "prompt_sha256": prompt["prompt_sha256"], "reply": reply, "telemetry": telemetry,
                   "elapsed_sec": time.perf_counter() - started,
                   "steering_applications": after_steering["applied"] - before_steering["applied"],
                   "steering_clips": after_steering["clipped"] - before_steering["clipped"],
                   "token_cap_reached": telemetry.get("tokens_generated", 0) >= int(max_new_tokens),
                   "timestamp": datetime.now(timezone.utc).isoformat()}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            trials.append(row)
            completed.add(key)
            _write_json(output / "summary.json", _summary(manifest, trials))
            announce(f"[{len(trials)}/{len(trial_plan)}] {item['arm']} {row['elapsed_sec']:.2f}s "
                     f"{telemetry.get('tokens_generated', '?')} tokens | {reply[:120].replace(chr(10), ' ')}")
    summary = _summary(manifest, trials)
    _write_json(output / "summary.json", summary)
    return summary
