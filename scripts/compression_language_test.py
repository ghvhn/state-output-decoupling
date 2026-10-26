"""Offline CPU language pilot for scoped point/trajectory compression.

GPT-2 attention query HEADS are the matrix slots in this pilot. They are not
GLM experts. A complete learned decoder is evaluated after replacing one query
projection in memory; key/value weights, head order, bias and disk weights stay
intact. Calibration and evaluation use separate, purpose-written trajectories.
The experiment tests transfer of an objective, not GLM quality or novelty.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# These are independently written calibration/evaluation inputs, not measured
# memories from another model. No probe names or meaning labels are injected.
CALIBRATION = (
    ("evidence", "The initial estimate was twelve. A new measurement gave forty-one. The researcher checked the instrument and revised the estimate using the measurement."),
    ("evidence", "The map suggested the door was on the left. Looking at the actual building showed the entrance on the right. The visitor corrected the route."),
    ("reference", "Alice handed the book to Ben. He opened it and read the first page. Alice asked him whether the text was clear."),
    ("reference", "The assistant described its answer and the reader described her question. They compared their accounts to identify which person each statement concerned."),
    ("procedure", "First compare the examples. Then remove details that change between them. Keep the recurring difference and express it in a brief sentence."),
    ("procedure", "To repair the device, inspect the cable, test the switch, and replace the broken connector. Finally repeat the original test to check the repair."),
    ("arithmetic", "There were seven cups on the table. Three were removed and two were added. Counting the remaining cups gives six."),
    ("arithmetic", "A train traveled ten miles in one hour and fifteen miles in the next hour. Its total distance was twenty-five miles."),
)
HELDOUT = (
    ("evidence", "A prediction gave twenty units. The observed result was thirty-six. After checking the observation, the analyst changed the estimate to thirty-six."),
    ("reference", "Maya asked Omar to explain his plan. He described the route, and she asked about the first stop. Their different accounts concerned the same journey."),
    ("procedure", "An example describes one case. A definition identifies a recurring feature across cases. Comparing several examples can reveal which details belong in the definition."),
    ("arithmetic", "A box contained nine pencils. Four were taken out and one was put back. Six pencils remained in the box."),
)


def write_json(path, value):
    path = Path(path)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def load_evaluation_fixture(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > 16384:
        raise ValueError("evaluation fixture exceeds its byte bound")
    fixture = json.loads(raw)
    if not isinstance(fixture, dict) or set(fixture) != {"provenance", "texts", "generation_prompts"}:
        raise ValueError("invalid sealed evaluation fixture")
    if fixture["provenance"] != "independently_authored_sealed_evaluation" or not isinstance(fixture["texts"], list) or not 4 <= len(fixture["texts"]) <= 16:
        raise ValueError("sealed evaluation requires independently authored bounded texts")
    texts = []
    for row in fixture["texts"]:
        if not isinstance(row, dict) or set(row) != {"environment_id", "text"}:
            raise ValueError("invalid evaluation trajectory")
        if row["environment_id"] not in {"evidence", "reference", "procedure", "arithmetic"} or not isinstance(row["text"], str) or not 80 <= len(row["text"]) <= 800:
            raise ValueError("evaluation text or environment outside fixture contract")
        texts.append((row["environment_id"], row["text"]))
    prompts = fixture["generation_prompts"]
    if not isinstance(prompts, list) or len(prompts) != 2 or any(not isinstance(prompt, str) or not 10 <= len(prompt) <= 400 for prompt in prompts):
        raise ValueError("two bounded free-generation prompts required")
    return tuple(texts), tuple(prompts), hashlib.sha256(raw).hexdigest()


def greedy_rollout(model, tokenizer, prompt, budget=16):
    import torch
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"]
    with torch.inference_mode():
        full = model.generate(input_ids=ids, attention_mask=encoded["attention_mask"], max_new_tokens=budget, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id, use_cache=True)
    generated = full[0, ids.shape[-1]:].cpu().tolist()
    return {"prompt": prompt, "generated_ids": generated,
            "continuation": tokenizer.decode(generated, skip_special_tokens=True),
            "budget": budget, "semantic_success_evaluated": False}


def rollout_comparison(teacher, student):
    original, candidate = teacher["generated_ids"], student["generated_ids"]
    common = 0
    for previous, next_ in zip(original, candidate):
        if previous != next_:
            break
        common += 1
    return {**student, "original_continuation": teacher["continuation"],
            "original_generated_ids": original, "common_prefix_tokens": common,
            "identical_token_sequence": original == candidate,
            "interpretation": "greedy self-generated rollout; different prefixes imply different later contexts"}


def query_heads(weight, heads):
    """GPT-2 Conv1D layout: [input, concatenated Q/K/V output]."""
    import torch
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] != 3*weight.shape[0]:
        raise ValueError("expected the GPT-2 fused query/key/value Conv1D layout")
    width = weight.shape[0]
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1 or width % heads:
        raise ValueError("head count must divide the query width")
    return weight[:, :width].T.reshape(heads, width//heads, width).detach().cpu().contiguous().clone()


def install_query_heads(module, weights):
    import torch
    if weights.ndim != 3 or weights.shape[0]*weights.shape[1] != weights.shape[2]:
        raise ValueError("query head slots do not match the fused query dimension")
    width = weights.shape[2]
    if tuple(module.weight.shape) != (width, 3*width):
        raise ValueError("target module layout mismatch")
    with torch.no_grad():
        module.weight[:, :width].copy_(weights.reshape(width, width).T)


def query_factor_output(inputs, native_output, bias, mean, basis, coefficients):
    """Differentiable query replacement; native key/value slots stay intact."""
    import torch
    width = inputs.shape[-1]
    heads = coefficients.shape[0]
    if width % heads or mean.numel() != (width//heads)*width:
        raise ValueError("factor query layout mismatch")
    weights = (mean[None, :]+coefficients@basis).reshape(width, width)
    query = inputs.float()@weights.T+bias[:width].float()
    return torch.cat((query, native_output[..., width:]), dim=-1)


def predictive_correction(model, module, reader, initial, scope, train, pairs,
                          teacher_rows, heldout, *, contrast_weight):
    """Train only against stored calibration logits, never held-out forwards."""
    import torch
    from tda.predictive_compress import PredictiveTrajectory, refine_predictive_factors
    lookup = {row["trajectory_id"]: row for row in teacher_rows}
    batches = []
    grouped = {}
    for record in train:
        grouped.setdefault(record.trajectory_id, []).append(record)
    for trajectory, records in grouped.items():
        ids = tuple(record.record_id for record in records)
        positions = tuple(int(record_id.rsplit(":", 1)[1]) for record_id in ids)
        row = lookup[trajectory]
        edges = tuple((ids.index(pair.previous_id), ids.index(pair.next_id)) for pair in pairs
                      if pair.previous_id in ids and pair.next_id in ids)
        batches.append(PredictiveTrajectory(trajectory_id=trajectory,
            environment_id=records[0].environment_id, scope=scope, record_ids=ids,
            token_positions=positions, teacher_logits=row["logits"][list(positions)].contiguous(),
            observed_next_token_ids=row["ids"][0, [position+1 for position in positions]].contiguous(),
            pairs=edges, provenance_weight=1., stimulation_weight=1., origin="measured_model_logits"))
    calls = [0]
    def forward(batch, mean, basis, coefficients):
        def override(_module, args, output):
            return query_factor_output(args[0], output, module.bias, mean, basis, coefficients)
        handle = module.register_forward_hook(override)
        try:
            logits = model(input_ids=lookup[batch.trajectory_id]["ids"], use_cache=False).logits[0]
            calls[0] += 1
            if calls[0] % 16 == 0:
                print(f"[Compression] Calibration-only correction forwards: {calls[0]}.", flush=True)
            return logits[list(batch.token_positions)]
        finally:
            handle.remove()
    requires = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        for parameter, _ in requires:
            parameter.requires_grad_(False)
        return refine_predictive_factors(reader, initial, batches, forward,
            expected_scope=scope, heldout_trajectory_ids=tuple(sorted({record.trajectory_id for record in heldout})),
            heldout_record_ids=tuple(record.record_id for record in heldout), epochs=4,
            learning_rate=.0005, point_weight=1., contrast_weight=contrast_weight, top_k=16,
            callback_working_bytes=sum(parameter.numel()*parameter.element_size() for parameter in model.parameters())+(128 << 20),
            max_working_bytes=1 << 30)
    finally:
        for parameter, required in requires:
            parameter.requires_grad_(required)


def token_metrics(teacher, student, token_ids):
    """Teacher-forced full-vocabulary distributions and observed next tokens."""
    import torch
    import torch.nn.functional as F
    if teacher.shape != student.shape or teacher.ndim != 2 or token_ids.ndim != 1 or len(token_ids) != len(teacher)+1:
        raise ValueError("logits and token timeline disagree")
    if len(teacher) < 1 or teacher.shape[-1] < 2 or not bool(torch.isfinite(teacher).all() & torch.isfinite(student).all()):
        raise ValueError("nonfinite or empty prediction evidence")
    teacher_log = F.log_softmax(teacher.float(), -1)
    student_log = F.log_softmax(student.float(), -1)
    kl = (teacher_log.exp()*(teacher_log-student_log)).sum(-1)
    targets = token_ids[1:]
    original_nll = F.nll_loss(teacher_log, targets, reduction="none")
    student_nll = F.nll_loss(student_log, targets, reduction="none")
    top = teacher.topk(2, dim=-1).indices
    teacher_margin = teacher.gather(-1, top).diff(dim=-1).neg().squeeze(-1)
    student_margin = student.gather(-1, top).diff(dim=-1).neg().squeeze(-1)
    return {"positions": len(teacher), "mean_kl_nats": float(kl.mean()),
            "mean_observed_token_nll_nats": float(student_nll.mean()),
            "mean_original_token_nll_nats": float(original_nll.mean()),
            "nll_change_nats": float((student_nll-original_nll).mean()),
            "argmax_agreement": float((teacher.argmax(-1) == student.argmax(-1)).float().mean()),
            "mean_top2_margin_absolute_error": float((teacher_margin-student_margin).abs().mean()),
            "minimum_kl_nats": float(kl.min()), "maximum_kl_nats": float(kl.max()),
            "evaluation": "teacher_forced_observed_next_tokens; no semantic correctness judgment"}


def collect(model, tokenizer, module, texts, scope, split, max_tokens):
    import torch
    from tda.trajectory_compress import CalibrationRecord, TrajectoryPair
    captured = []
    def capture(_module, args):
        captured.append(args[0].detach().float().cpu().clone())
    records, pairs, evaluated, index = [], [], [], []
    hook = module.register_forward_pre_hook(capture)
    try:
        for number, (environment, text) in enumerate(texts):
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)["input_ids"]
            captured.clear()
            with torch.inference_mode():
                logits = model(input_ids=ids, use_cache=False).logits[0].float().cpu()
            if len(captured) != 1:
                raise RuntimeError("expected one measured input capture per trajectory")
            states = captured[0][0]
            trajectory = f"{split}:{number}"
            positions = sorted(set(torch.linspace(1, len(ids[0])-2, min(12, len(ids[0])-2)).round().long().tolist()))
            previous = None
            for position in positions:
                record_id = f"{trajectory}:{position}"
                correct = bool(logits[position].argmax() == ids[0, position+1])
                record = CalibrationRecord(record_id=record_id, environment_id=environment,
                    trajectory_id=trajectory, scope=scope, state=states[position].clone(),
                    provenance_weight=1., stimulation_weight=1., prediction_correct=correct,
                    origin="measured_model_state")
                records.append(record)
                index.append({"record_id": record_id, "environment": environment,
                              "trajectory_id": trajectory, "token_position": position,
                              "prediction_correct": correct, "split": split})
                if previous is not None:
                    pairs.append(TrajectoryPair(f"{previous}->{record_id}", previous, record_id))
                previous = record_id
            evaluated.append({"ids": ids, "logits": logits[:-1], "text": text,
                              "environment": environment, "trajectory_id": trajectory})
    finally:
        hook.remove()
    return records, pairs, evaluated, index


def shuffled_pairs(records, seed=132):
    """Shuffle temporal successors within each trajectory; no cross-scope pairing."""
    from tda.trajectory_compress import TrajectoryPair
    by_trajectory = {}
    for record in records:
        by_trajectory.setdefault(record.trajectory_id, []).append(record.record_id)
    rng = random.Random(seed)
    result = []
    for ids in by_trajectory.values():
        order = list(ids)
        rng.shuffle(order)
        for first, second in zip(order, order[1:]):
            result.append(TrajectoryPair("shuffle:"+first+"->"+second, first, second))
    return result


def response_metrics(original, restored, records, pairs):
    import torch
    lookup = {record.record_id: record for record in records}
    states = torch.stack([record.state for record in records])
    before = torch.einsum("eoi,ni->eno", original, states)
    after = torch.einsum("eoi,ni->eno", restored, states)
    point_energy = float(before.square().sum())
    point_error = float((after-before).square().sum())
    deltas = torch.stack([lookup[p.next_id].state-lookup[p.previous_id].state for p in pairs])
    old_delta = torch.einsum("eoi,ni->eno", original, deltas)
    new_delta = torch.einsum("eoi,ni->eno", restored, deltas)
    delta_energy = float(old_delta.square().sum())
    delta_error = float((new_delta-old_delta).square().sum())
    return {"point_response_relative_error": math.sqrt(point_error/point_energy) if point_energy else 0.,
            "trajectory_response_relative_error": math.sqrt(delta_error/delta_energy) if delta_energy else 0.,
            "scope": "heldout_original_model_query_input_states_and_within_trajectory_differences"}


def run(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader, fit_expert_group, reconstruct_expert, save_expert_compression, load_expert_compression
    from tda.trajectory_compress import InputStateScope, fit_trajectory_group
    from tda.block_quant import quantize_reader, save_block_quantization, load_block_quantization

    torch.set_num_threads(args.threads)
    torch.manual_seed(132)
    if args.out.exists():
        raise FileExistsError("use a new output directory for each experiment")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    results = []
    write_json(args.out/"status.json", {"status": "running", "device": "cpu"})
    try:
        evaluation_texts, generation_prompts, fixture_sha = HELDOUT, (), None
        if args.evaluation_fixture:
            evaluation_texts, generation_prompts, fixture_sha = load_evaluation_fixture(args.evaluation_fixture)
        print("[Compression] Loading the cached GPT-2 decoder on CPU.", flush=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=True, dtype=torch.float32).eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        if not 0 <= args.layer < len(model.transformer.h):
            raise ValueError("layer outside the model")
        module = model.transformer.h[args.layer].attn.c_attn
        original = query_heads(module.weight, model.config.n_head)
        scope = ExpertGroupIdentity("gpt2", model.config._commit_hash, args.layer, "attention_query_heads",
            "gpt2_c_attn_query_rows_preserving_head_slots", tuple(f"head_{i}" for i in range(len(original))),
            tuple(original.shape[1:]), "float32")
        input_scope = InputStateScope.from_identity(scope, state_site="attention_query_input_after_ln1")
        reader = TensorExpertReader(scope, original)
        print("[Compression] Capturing separate calibration and held-out language trajectories.", flush=True)
        if args.reuse_artifacts:
            train, train_pairs, train_teacher, train_index = [], [], [], []
        else:
            train, train_pairs, train_teacher, train_index = collect(model, tokenizer, module, CALIBRATION, input_scope, "calibration", args.max_tokens)
        heldout, heldout_pairs, teacher, heldout_index = collect(model, tokenizer, module, evaluation_texts, input_scope,
            "sealed" if args.evaluation_fixture else "heldout", args.max_tokens)
        original_rollouts = [greedy_rollout(model, tokenizer, prompt) for prompt in generation_prompts]
        write_json(args.out/"evidence.json", {"scope": scope.to_dict(), "state_site": input_scope.state_site,
            "calibration_texts": () if args.reuse_artifacts else CALIBRATION, "heldout_texts": evaluation_texts, "records": train_index+heldout_index,
            "calibration_pairs": [asdict(p) for p in train_pairs], "heldout_pairs": [asdict(p) for p in heldout_pairs],
            "correct_and_incorrect_predictions_are_valid": True,
            "source_query_weight_sha256": hashlib.sha256(original.numpy().tobytes()).hexdigest(),
            "source_kind": "attention_heads; not MoE experts", "training_and_heldout_trajectories_disjoint": True})
        torch.save({"calibration": torch.stack([r.state for r in train]) if train else torch.empty((0, scope.matrix_shape[1])),
                    "heldout": torch.stack([r.state for r in heldout])}, args.out/"states.pt")
        settings = {"model": "gpt2", "revision": model.config._commit_hash, "layer": args.layer,
            "rank": args.rank, "factor_dtype": "float16", "max_tokens": args.max_tokens,
            "device": "cpu", "glm_behavior_evaluated": False, "semantic_truth_checked": False,
            "whole_model_compressed": False, "one_query_projection_replaced_in_memory": True,
            "predictive_correction": args.predictive_correction,
            "correction_epochs": 4 if args.predictive_correction else 0,
            "correction_learning_rate": .0005 if args.predictive_correction else None,
            "heldout_previously_seen_in_exploratory_iteration": bool(args.predictive_correction and not args.evaluation_fixture),
            "heldout_not_used_for_optimization_or_checkpoint_selection": True,
            "evaluation_fixture_sha256": fixture_sha,
            "reuse_artifacts": str(args.reuse_artifacts) if args.reuse_artifacts else None,
            "no_fit_or_refinement_in_this_run": bool(args.reuse_artifacts)}
        write_json(args.out/"settings.json", settings)
        controls = [
            ("weight_pca", None, 0., 0.),
            ("point_only", train_pairs, 1., 0.),
            ("trajectory", train_pairs, 1., 1.),
            ("shuffled_successors", shuffled_pairs(train), 1., 1.),
            ("symmetric_int4", None, 0., 0.),
        ]
        if args.predictive_correction:
            controls.extend((("language_point_corrected", None, 1., 0.),
                             ("language_trajectory_corrected", None, 1., .25)))
        trajectory_encoded = None
        for name, pairs, point_weight, contrast_weight in controls:
            print(f"[Compression] {'Loading fixed factors and evaluating' if args.reuse_artifacts else 'Fitting and evaluating'} {name}.", flush=True)
            if args.reuse_artifacts:
                artifact = args.reuse_artifacts/name
                if name == "symmetric_int4":
                    encoded = load_block_quantization(artifact, expected_identity=scope)
                    restored = encoded.read_columns(0, scope.column_count).view_as(original)
                else:
                    encoded = load_expert_compression(artifact, expected_identity=scope)
                    if encoded.rank != args.rank or encoded.metadata["storage_dtype"] != "float16":
                        raise ValueError("reused artifact rank or factor precision differs from declared settings")
                    observed_ids = {record.record_id for record in heldout}
                    observed_trajectories = {record.trajectory_id for record in heldout}
                    objective = encoded.metadata.get("objective", {})
                    if observed_ids & set(objective.get("training_record_ids", [])) or observed_trajectories & set(objective.get("training_trajectory_ids", [])):
                        raise ValueError("sealed evaluation overlaps artifact calibration")
                    restored = torch.stack([reconstruct_expert(encoded, slot) for slot in scope.expert_ids])
                serialized = encoded.metadata
            elif name == "weight_pca":
                encoded = fit_expert_group(reader, args.rank, storage_dtype="float16")
                serialized = save_expert_compression(encoded, args.out/name)
                restored = torch.stack([reconstruct_expert(encoded, slot) for slot in scope.expert_ids])
            elif name.startswith("language_"):
                if trajectory_encoded is None:
                    raise RuntimeError("trajectory initializer has not been fitted")
                encoded = predictive_correction(model, module, reader, trajectory_encoded, input_scope,
                    train, train_pairs, train_teacher, heldout, contrast_weight=contrast_weight)
                serialized = save_expert_compression(encoded, args.out/name)
                restored = torch.stack([reconstruct_expert(encoded, slot) for slot in scope.expert_ids])
            elif name == "symmetric_int4":
                encoded = quantize_reader(reader, bits=4, block_size=128)
                serialized = save_block_quantization(encoded, args.out/name)
                restored = encoded.read_columns(0, scope.column_count).view_as(original)
            else:
                encoded = fit_trajectory_group(reader, args.rank, train, pairs, expected_scope=input_scope,
                    heldout_records=heldout, heldout_pairs=heldout_pairs, point_weight=point_weight,
                    contrast_weight=contrast_weight, weight_floor=.05, storage_dtype="float16")
                serialized = save_expert_compression(encoded, args.out/name)
                restored = torch.stack([reconstruct_expert(encoded, slot) for slot in scope.expert_ids])
                if name == "trajectory":
                    trajectory_encoded = encoded
            install_query_heads(module, restored)
            metrics = []
            rollouts = []
            try:
                with torch.inference_mode():
                    for row in teacher:
                        logits = model(input_ids=row["ids"], use_cache=False).logits[0, :-1].float().cpu()
                        metrics.append({"environment": row["environment"], "trajectory": row["trajectory_id"],
                                        **token_metrics(row["logits"], logits, row["ids"][0])})
                    rollouts = [rollout_comparison(before, greedy_rollout(model, tokenizer, prompt))
                                for before, prompt in zip(original_rollouts, generation_prompts)]
            finally:
                install_query_heads(module, original)
            total = sum(row["positions"] for row in metrics)
            aggregate = {key: sum(row[key]*row["positions"] for row in metrics)/total
                for key in ("mean_kl_nats", "nll_change_nats", "argmax_agreement", "mean_top2_margin_absolute_error")}
            result = {"variant": name, "serialized_total_bytes": serialized["serialized_total_bytes"],
                "factor_tensor_bytes": serialized["factor_tensor_bytes"],
                "source_tensor_bytes": scope.actual_source_bytes,
                "serialized_storage_ratio": serialized["serialized_storage_ratio"],
                "full_decoder_teacher_forced_metrics": aggregate, "per_trajectory": metrics,
                "local_response_metrics": response_metrics(original, restored, heldout, heldout_pairs),
                "greedy_rollouts": rollouts,
                "byte_control": "PCA/point/trajectory/shuffled factor payloads identical; INT4 budget separately reported",
                "glm_quality_established": False, "semantic_quality_established": False}
            results.append(result)
            write_json(args.out/"results.json", results)
            print(f"[Compression] {name}: KL={aggregate['mean_kl_nats']:.6f}; argmax agreement={aggregate['argmax_agreement']:.1%}.", flush=True)
        # Verify restoration independently of the decoder comparisons.
        torch.testing.assert_close(query_heads(module.weight, model.config.n_head), original, rtol=0, atol=0)
        write_json(args.out/"status.json", {"status": "finished", "elapsed_seconds": time.monotonic()-started,
            "completed_variants": [r["variant"] for r in results], "source_weights_restored": True, "device": "cpu"})
    except BaseException as exc:
        write_json(args.out/"status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "completed_variants": [r["variant"] for r in results], "device": "cpu"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--predictive-correction", action="store_true")
    parser.add_argument("--evaluation-fixture", type=Path)
    parser.add_argument("--reuse-artifacts", type=Path,
                        help="Evaluate fixed existing factors on a sealed fixture, without fitting or refinement.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rank <= 11 or not 8 <= args.max_tokens <= 96 or not 1 <= args.threads <= 16:
        parser.error("rank must be1..11, max-tokens8..96, threads1..16")
    if args.reuse_artifacts and not args.evaluation_fixture:
        parser.error("reused factors require a sealed evaluation fixture")
    run(args)


if __name__ == "__main__":
    main()
