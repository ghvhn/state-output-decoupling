"""CPU contract tests. No claim about full GLM CUDA loading or throughput."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tda.glm_capture_runtime import (
    CaptureDeadline, CaptureHooks, logical_tensor, prepare_batch, validate_checkpoint_metadata,
    validate_examples, validate_identity, validate_capture_budget, _source_head_blocks,
    validate_native_tp_head,
)


def identity():
    return dict(model_id="fixture/native-moe", revision="0" * 40, hidden_size=4,
                vocab_size=16, num_layers=2, source_precision="synthetic_fixture",
                tokenizer_sha256="a" * 64, chat_template_sha256="b" * 64)


def corpus():
    scope = identity()
    return {"model_id": scope["model_id"], "revision": scope["revision"],
            "tokenizer_identity": {key: scope[key] for key in ("tokenizer_sha256", "chat_template_sha256")},
            "examples": [
                dict(trajectory_id="a", environment_id="reference", split="calibration",
                     token_ids=[1, 2, 3, 4], selected_positions=[0, 1, 2], pairs=[[0, 1], [1, 2]],
                     provenance_weight=1.0, stimulation_weight=0.4),
                dict(trajectory_id="b", environment_id="procedure", split="heldout",
                     token_ids=[5, 6, 7], selected_positions=[0, 1], pairs=[[0, 1]])]}


class FixtureRouter(nn.Module):
    top_k = 2
    def forward(self, hidden):
        hidden = hidden.reshape(-1, hidden.shape[-1])
        logits = hidden[:, :2].float()
        weights = logits.softmax(-1).to(hidden.dtype)
        indices = torch.tensor([0, 1], dtype=torch.int64).expand(len(hidden), 2)
        return logits, weights, indices


class FixtureExperts(nn.Module):
    def forward(self, hidden, indices, weights):
        return hidden * weights.sum(-1, keepdim=True)


class FixtureMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate, self.experts = FixtureRouter(), FixtureExperts()
    def forward(self, hidden):
        shape = hidden.shape
        _, weights, indices = self.gate(hidden)
        return self.experts(hidden.reshape(-1, shape[-1]), indices, weights).reshape(shape)


class FixtureLayer(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.mlp, self.number = FixtureMoE(), number
    def forward(self, hidden):
        # Distinct pre-MLP and post-block boundaries so hook tests detect a shift.
        normalized = hidden * 0.5 + self.number
        return hidden + self.mlp(normalized), None


class FixtureBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 4, dtype=torch.bfloat16)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(64).reshape(16, 4) / 32)
        self.layers = nn.ModuleList([FixtureLayer(0), FixtureLayer(1)])
    def forward(self, input_ids, attention_mask, use_cache=False):
        # Causal prefix summary independently in every batch row; padding masked.
        hidden = (self.embed(input_ids) * attention_mask[..., None]).cumsum(1)
        for layer in self.layers:
            hidden, _ = layer(hidden)
        final = hidden / (hidden.square().mean(-1, keepdim=True) + 0.1).sqrt()
        return SimpleNamespace(last_hidden_state=final)


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FixtureBase()
        self.lm_head = nn.Linear(4, 16, bias=False, dtype=torch.bfloat16)


def capture(model, examples):
    batch = prepare_batch(examples, pad_token_id=0)
    hooks = CaptureHooks(model, identity(), [0, 1], 1).install()
    try:
        hooks.begin(batch)
        with torch.inference_mode():
            output = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            result = hooks.finish(output.last_hidden_state)
        return batch, result
    finally:
        hooks.close()


def test_causal_batch_isolation_and_native_selected_boundaries():
    examples = validate_examples(corpus(), identity())
    model = FixtureModel()
    batch, result = capture(model, examples)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4], [5, 6, 7, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert batch["pair_rows"] == [[0, 1], [1, 2], [3, 4]]
    assert result["final_hidden"].dtype == torch.bfloat16
    assert result["layer_1_expert_input"].dtype == torch.bfloat16
    assert not torch.equal(result["final_hidden"], result["layer_1_hidden"])
    assert torch.equal(result["layer_1_expert_input"], result["layer_0_hidden"] * 0.5 + 1)
    singles = [capture(model, [row])[1] for row in examples]
    for name in result:
        assert torch.equal(result[name], torch.cat([row[name] for row in singles])), name


def test_final_hidden_reconstructs_full_logits_without_forward_or_generation():
    model = FixtureModel()
    _, result = capture(model, validate_examples(corpus(), identity()))
    with torch.inference_mode():
        online = model.lm_head(result["final_hidden"])
        offline = torch.nn.functional.linear(result["final_hidden"], model.lm_head.weight.detach().clone())
    assert torch.equal(online, offline)
    assert online.shape == (5, 16)


def test_native_tp_head_accepts_local_storage_and_rejects_unsharded_or_wrong_axis():
    # Transformers 5.15 stores ordinary local Parameters; only forward logits
    # are gathered. Requiring a full vocabulary weight aborts a valid TP8 load.
    head = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    head._hf_tp_plan = "colwise_gather_output"
    result = validate_native_tp_head(head, identity(), world_size=8)
    assert result["logical_shape"] == [16, 4]
    assert result["local_weight_shape"] == [2, 4]
    for bad in (torch.zeros(16, 4, dtype=torch.bfloat16),
                torch.zeros(16, 1, dtype=torch.bfloat16), torch.zeros(2, 4)):
        head.weight = nn.Parameter(bad)
        with pytest.raises(ValueError, match="vocabulary-row shard"):
            validate_native_tp_head(head, identity(), world_size=8)


def test_native_tp_head_requires_verified_gather_plan_and_even_vocabulary():
    head = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="gather"):
        validate_native_tp_head(head, identity(), world_size=8)
    head._hf_tp_plan = "colwise_gather_output"
    uneven = identity() | {"vocab_size": 17}
    with pytest.raises(ValueError, match="divide evenly"):
        validate_native_tp_head(head, uneven, world_size=8)


def test_hooks_remove_and_retained_data_do_not_alias_forward_states():
    model = FixtureModel()
    _, result = capture(model, validate_examples(corpus(), identity()))
    retained = result["layer_1_expert_input"].clone()
    with torch.no_grad():
        model.model.embed.weight.add_(100)
    assert torch.equal(result["layer_1_expert_input"], retained)
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert not model.model.layers[1].mlp.experts._forward_pre_hooks


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c.update(revision="other"), "revision"),
    (lambda c: c["tokenizer_identity"].update(tokenizer_sha256="c" * 64), "tokenizer"),
    (lambda c: c["examples"][0].update(selected_positions=[2, 1]), "sorted"),
    (lambda c: c["examples"][0].update(selected_positions=[0, 0]), "unique"),
    (lambda c: c["examples"][0].update(selected_positions=[3]), "actual next"),
    (lambda c: c["examples"][0].update(token_ids=[1, 16]), "vocabulary"),
    (lambda c: c["examples"][0].update(token_ids=[True, 1]), "vocabulary"),
    (lambda c: c["examples"][0].update(pairs=[[0, 3]]), "endpoints"),
    (lambda c: c["examples"][0].update(pairs=[[1, 0]]), "increasing"),
    (lambda c: c["examples"][1].update(trajectory_id="a"), "exactly one"),
    (lambda c: c["examples"][0].update(provenance_weight=float("nan")), "finite"),
    (lambda c: c["examples"][0].update(stimulation_weight=2), "between"),
    (lambda c: c["examples"][0].update(split="unknown"), "split"),
])
def test_corpus_rejects_scope_leakage_invalid_positions_and_weights(mutate, match):
    value = corpus()
    mutate(value)
    with pytest.raises(ValueError, match=match):
        validate_examples(value, identity())


def test_probe_framing_last_position_retained_without_invented_next_token():
    value = corpus()
    value["examples"][0].update(kind="probe_framing", probe_id="p_opaque", polarity=1,
                                selected_positions=[3], pairs=[])
    batch = prepare_batch(validate_examples(value, identity()), pad_token_id=0)
    assert batch["rows"][0]["next_token_id"] == -1
    assert batch["rows"][0]["next_token_observed"] is False


def test_context_never_silently_truncated():
    with pytest.raises(ValueError, match="never truncates"):
        validate_examples(corpus(), identity(), max_seq_len=3)


def test_budget_rejects_unserializable_batches_before_loading():
    examples = validate_examples(corpus(), identity())
    with pytest.raises(ValueError, match="chunk budget"):
        validate_capture_budget(examples, identity(), [0, 1], batch_size=2,
                                max_chunk_bytes=1000, max_total_bytes=1024 ** 3)
    with pytest.raises(ValueError, match="native output head"):
        validate_capture_budget(examples, identity(), [0, 1], batch_size=2,
                                max_chunk_bytes=1024 ** 2, max_total_bytes=1024)


def test_native_capture_contract_is_accepted_by_archive_writer(tmp_path):
    from tda.glm_capture_archive import CaptureArchiveWriter, load_archive_manifest
    model = FixtureModel()
    batch, tensors = capture(model, validate_examples(corpus(), identity()))
    with torch.inference_mode():
        tensors["spotcheck_logits"] = model.lm_head(tensors["final_hidden"][:2]).clone()
        tensors["spotcheck_rows"] = torch.arange(2, dtype=torch.int64)
    sites = {"final_hidden": "post_final_norm_lm_head_input",
             "layer_1_expert_input": "post_attention_norm_expert_gate_up_input",
             "layer_0_hidden": "decoder_block_output", "layer_1_hidden": "decoder_block_output"}
    archive = tmp_path / "capture"
    writer = CaptureArchiveWriter(archive, identity(), origin="synthetic_fixture")
    writer.write_chunk("first", tensors, {"rows": batch["rows"], "pairs": batch["pair_rows"], "state_sites": sites})
    writer.write_head_block(0, model.lm_head.weight.detach().clone())
    writer.finalize()
    manifest = load_archive_manifest(archive)
    assert manifest["status"] == "complete"
    assert manifest["origin"] == "synthetic_fixture"


def test_identity_rejects_altered_teacher_and_extra_scope_fields():
    with pytest.raises(ValueError, match="pinned original"):
        validate_identity(identity(), production=True)
    with pytest.raises(ValueError, match="exactly"):
        validate_identity({**identity(), "unknown": "x"})


def test_logical_tensor_refuses_rank_local_feature_slices():
    with pytest.raises(ValueError, match="logical shape"):
        logical_tensor(torch.zeros(2, 2), expected_shape=(2, 4), name="hidden")


class FakeCollectiveTensor(torch.Tensor):
    calls = 0
    @property
    def placements(self):
        return ("Shard(feature)",)
    def full_tensor(self):
        type(self).calls += 1
        return torch.ones(2, 4, dtype=self.dtype)


def test_dtensor_path_materializes_one_logical_tensor_collectively():
    FakeCollectiveTensor.calls = 0
    local = torch.zeros(2, 2).as_subclass(FakeCollectiveTensor)
    result = logical_tensor(local, expected_shape=(2, 4), name="hidden")
    assert FakeCollectiveTensor.calls == 1
    assert result.shape == (2, 4)


def test_deadline_includes_loading_and_survives_wall_clock_changes():
    now = [100.0]
    clock = CaptureDeadline(160, 10, wall_clock=lambda: 100, monotonic_clock=lambda: now[0])
    now[0] = 140  # Loading consumes forty seconds of the already-started budget.
    assert clock.remaining() == 20
    assert clock.may_start(5)
    assert not clock.may_start(15)
    now[0] = 165
    assert clock.remaining() == 0
    assert not clock.may_start()


@pytest.mark.parametrize("deadline,reserve", [(float("nan"), 1), (10, -1), (True, 1)])
def test_deadline_rejects_nonfinite_or_invalid_limits(deadline, reserve):
    with pytest.raises(ValueError):
        CaptureDeadline(deadline, reserve)


def test_metadata_refuses_dequantized_or_other_architecture(tmp_path):
    value = dict(model_type="glm_moe_dsa", hidden_size=4, vocab_size=16, num_hidden_layers=2,
                 quantization_config=dict(quant_method="fp8", fmt="e4m3", weight_block_size=[128, 128], activation_scheme="dynamic"))
    (tmp_path / "config.json").write_text(json.dumps(value))
    assert validate_checkpoint_metadata(tmp_path, identity())["hidden_size"] == 4
    value["quantization_config"]["dequantize"] = True
    (tmp_path / "config.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="original"):
        validate_checkpoint_metadata(tmp_path, identity())


def test_source_head_export_preserves_native_precision_and_row_layout(tmp_path):
    from safetensors.torch import save_file
    weight = torch.arange(64).reshape(16, 4).to(torch.bfloat16)
    save_file({"lm_head.weight": weight}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "model.safetensors"}}))
    blocks = list(_source_head_blocks(tmp_path, identity(), rows_per_block=5))
    assert [start for start, _ in blocks] == [0, 5, 10, 15]
    assert torch.equal(torch.cat([block for _, block in blocks]), weight)


def test_source_head_export_rejects_uncontained_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="contained"):
        list(_source_head_blocks(tmp_path, identity()))


def test_cpu_rendezvous_refuses_hostname_or_missing_ranks_before_initialization(monkeypatch):
    import torch.distributed as dist
    from tda.glm_capture_runtime import check_rendezvous
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "modal")
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: pytest.fail("hostname accepted"))
    with pytest.raises(RuntimeError, match="literal loopback"):
        check_rendezvous()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("WORLD_SIZE", "7")
    with pytest.raises(RuntimeError, match="eight ranks"):
        check_rendezvous()


def test_online_kernel_warmup_retains_version_ref_for_fresh_offline_process(tmp_path):
    from tda.glm_capture_runtime import verify_fp8_kernel_cache
    commit = "a" * 40
    repository = tmp_path / "kernels--kernels-community--finegrained-fp8"
    metadata = repository / "snapshots" / commit / "build" / "native-variant" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"version":4}')
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        ref = repository / "refs" / "v4"
        ref.parent.mkdir()
        ref.write_text(commit)
        return str(metadata)
    online = verify_fp8_kernel_cache(commit=commit, cache_dir=tmp_path, offline=False, download_file=download)
    offline = verify_fp8_kernel_cache(commit=commit, cache_dir=tmp_path, offline=True,
        download_file=lambda **_: pytest.fail("offline process used network"))
    assert online == offline
    assert calls[0]["revision"] == "v4"
    assert calls[0]["repo_type"] == "kernel"
    assert online["cache_artifacts"][0]["file"] == "build/native-variant/metadata.json"
    metadata.write_text('{"version":4,"changed":true}')
    assert verify_fp8_kernel_cache(commit=commit, cache_dir=tmp_path, offline=True) != online


@pytest.mark.parametrize("ref", [None, "b" * 40])
def test_kernel_snapshot_alone_or_different_version_cannot_pass_offline_check(tmp_path, ref):
    from tda.glm_capture_runtime import verify_fp8_kernel_cache
    commit = "a" * 40
    repository = tmp_path / "kernels--kernels-community--finegrained-fp8"
    metadata = repository / "snapshots" / commit / "build" / "variant" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}")
    if ref is not None:
        (repository / "refs").mkdir()
        (repository / "refs" / "v4").write_text(ref)
    with pytest.raises(RuntimeError, match="version ref"):
        verify_fp8_kernel_cache(commit=commit, cache_dir=tmp_path, offline=True)


def test_fp8_callable_bundle_is_attested_to_pinned_module_for_local_loading(tmp_path):
    from tda.glm_capture_runtime import validate_fp8_kernel_bundle, FP8_KERNEL_COMMIT
    module_file = tmp_path / "kernels--kernels-community--finegrained-fp8" / "snapshots" / FP8_KERNEL_COMMIT / "build" / "variant" / "__init__.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("# fixture")
    a, b, c = (lambda: None), (lambda: None), (lambda: None)
    module = SimpleNamespace(__file__=str(module_file), matmul_2d=a, matmul_batched=b, matmul_grouped=c)
    entry = SimpleNamespace(module=module, metadata=SimpleNamespace(version=4), repo_info=None)
    bundle = SimpleNamespace(matmul=a, batched_matmul=b, grouped_matmul=c)
    validate_fp8_kernel_bundle(bundle, [entry], cache_dir=tmp_path)
    bundle.grouped_matmul = lambda: None
    with pytest.raises(RuntimeError, match="identity"):
        validate_fp8_kernel_bundle(bundle, [entry], cache_dir=tmp_path)
    bundle.grouped_matmul = c
    module.__file__ = str(tmp_path / "other.py")
    with pytest.raises(RuntimeError, match="outside"):
        validate_fp8_kernel_bundle(bundle, [entry], cache_dir=tmp_path)
