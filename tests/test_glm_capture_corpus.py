from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from tda.glm_capture_corpus import (CorpusError, FORMAT, MODEL_ID, PINNED_REVISION,
    NativeTokenizer, SourceArchive, add_probe_framings, build_corpus,
    conversation_splits, export_corpus, read_memory_sources, reference_conversations,
    selected_timeline, validate_corpus)


class FakeTokenizer:
    model_id = MODEL_ID
    revision = PINNED_REVISION
    vocab_size = 512
    identity = {"tokenizer_sha256": "a" * 64, "tokenizer_config_sha256": "b" * 64,
                "chat_template_sha256": "c" * 64, "config_sha256": "d" * 64,
                "vocab_size": 512, "actual_tokenizer_vocabulary": 512}

    def __init__(self):
        self.contexts = []

    def encode_context(self, messages, *, add_generation_prompt=False):
        self.contexts.append(deepcopy(messages))
        text = "\n".join(row["role"] + ":" + row["content"] for row in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return [ord(char) % 512 for char in text], hashlib.sha256(text.encode()).hexdigest()


def row(session, number, role, text, *, scope="interactive_phenomenality", **fields):
    timestamp = datetime(2026, 10, 2, tzinfo=timezone.utc) + timedelta(seconds=number)
    return {"kind": "turn", "role": role, "text": text, "record_id": f"{session}:{number}",
            "session_id": session, "scope": scope, "timestamp": timestamp.isoformat(),
            "tags": ["conversation_trace"], "turn_id": f"{session}:{number}", **fields}


def write_memory(path, records):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    return path


def archive(tmp_path):
    records = []
    for session in ("alpha", "beta", "gamma", "delta"):
        records.extend([row(session, 0, "user", f"A complete question from {session}."),
                        row(session, 1, "assistant", "A complete answer that preserves its source."),
                        row(session, 2, "user", "Now check the earlier result."),
                        row(session, 3, "assistant", "Checking can revise the conclusion.")])
    return read_memory_sources([write_memory(tmp_path / "memory.jsonl", records)])


def corpus(tmp_path, **kwargs):
    return build_corpus(archive(tmp_path), FakeTokenizer(), max_tokens=256, **kwargs)[0]


def test_full_source_text_is_preserved_and_never_replaced_by_behavior_prefixes(tmp_path):
    question, answer = "Q" * 400, "R" * 900
    path = write_memory(tmp_path / "full.jsonl", [row("one", 0, "user", question),
        row("one", 1, "assistant", answer), row("two", 0, "user", "Other question."),
        row("two", 1, "assistant", "Other complete answer.")])
    source = read_memory_sources([path])
    tokenizer = FakeTokenizer()
    result, _ = build_corpus(source, tokenizer, include_reference=False, max_tokens=128)
    assert source.source_records[0]["text"] == question
    assert source.source_records[1]["text"] == answer
    assert any(context[-1]["content"] == answer for context in tokenizer.contexts)
    truncated = next(example for example in result["examples"] if len(example["token_ids"]) == 128)
    assert truncated["context"]["rendered_token_count"] > 128
    assert truncated["context"]["retained_token_range"] == [0, 128]
    assert truncated["context"]["token_prefix_truncated"] is True


def test_operator_records_and_shell_commands_do_not_enter_context(tmp_path):
    records = [row("one", 0, "user", "A real question."),
        {"kind": "internal_trace", "text": "Do not add this measurement as a reply.", "tags": []},
        row("one", 1, "user", ":probe backfill all"),
        row("one", 2, "assistant", "Measurement noise", tags=["operator_observation"]),
        row("one", 3, "assistant", "A real answer."),
        row("two", 0, "user", "Another real question."), row("two", 1, "assistant", "Another answer.")]
    source = read_memory_sources([write_memory(tmp_path / "memory.jsonl", records)])
    tokenizer = FakeTokenizer()
    result, _ = build_corpus(source, tokenizer, include_reference=False)
    text = json.dumps(tokenizer.contexts)
    assert "measurement as a reply" not in text
    assert ":probe" not in text
    assert "Measurement noise" not in text
    assert len(source.observations) == 3
    assert all(item["prompt_eligible"] is False for item in source.observations)
    assert result["privacy"]["operator_records_in_prompt"] is False


def test_context_never_crosses_sessions_and_uses_original_records(tmp_path):
    source = archive(tmp_path)
    tokenizer = FakeTokenizer()
    result, _ = build_corpus(source, tokenizer, include_reference=False)
    by_id = {record["record_id"]: record for record in source.source_records}
    for example in result["examples"]:
        sessions = {by_id[record_id]["session_id"] for record_id in example["source_record_ids"]}
        assert len(sessions) == 1
    for context in tokenizer.contexts:
        assert sum(session in json.dumps(context) for session in ("alpha", "beta", "gamma", "delta")) == 1


def test_split_is_deterministic_order_independent_and_conversation_scoped(tmp_path):
    source = archive(tmp_path)
    first = conversation_splits(source.conversations, seed="fixed")
    assert first == conversation_splits(list(reversed(source.conversations)), seed="fixed")
    result, _ = build_corpus(source, FakeTokenizer(), include_reference=False, seed="fixed")
    split_by_group = {}
    for example in result["examples"]:
        group = example["conversation_id"]
        assert split_by_group.setdefault(group, example["split"]) == example["split"]
    assert set(split_by_group.values()) == {"calibration", "heldout"}


def test_minimal_budget_preserves_both_splits_and_balances_reference_buckets(tmp_path):
    result = corpus(tmp_path, max_examples=2)
    assert {example["split"] for example in result["examples"]} == {"calibration", "heldout"}
    broad = corpus(tmp_path, max_examples=12)
    assert {"archive", "definitions", "corrections", "arithmetic", "evidence"} <= {
        example["selection_bucket"] for example in broad["examples"]}


def test_measured_pairs_are_adjacent_on_a_valid_next_token_timeline():
    for count in (3, 4, 32, 1024):
        positions, pairs = selected_timeline(count, max_positions=32)
        assert positions == sorted(set(positions))
        assert max(positions) < count - 1
        assert len(positions) <= 32
        assert all(b == a + 1 and a in positions and b in positions for a, b in pairs)


def test_whole_turn_window_is_explicit_and_does_not_relabel_token_offsets(tmp_path):
    source = archive(tmp_path)
    result, _ = build_corpus(source, FakeTokenizer(), include_reference=False,
                             max_context_turns=2, max_tokens=256)
    limited = [example for example in result["examples"] if example["context"]["dropped_prefix_record_ids"]]
    assert limited
    assert all(len(example["source_record_ids"]) == 2 for example in limited)
    assert all(example["context"]["retained_token_range"][0] == 0 for example in limited)


@pytest.mark.parametrize("change, match", [
    (lambda c: c.update(revision="main"), "revision"),
    (lambda c: c["tokenizer_identity"].update(vocab_size=1), "output-head"),
    (lambda c: c["examples"][0].update(selected_positions=[len(c["examples"][0]["token_ids"])-1]), "timeline"),
    (lambda c: c["examples"][0].update(selected_positions=[True]), "timeline"),
    (lambda c: c["examples"][0].update(selected_positions=[2, 1]), "sorted"),
    (lambda c: c["examples"][0].update(pairs=[[0, 9]]), "adjacent"),
    (lambda c: c["examples"][0]["context"].update(retained_token_range=[1, 2]), "timeline"),
    (lambda c: c["examples"][0]["context"].update(rendered_token_count=1), "shorter"),
])
def test_stale_head_and_timeline_contract_failures(tmp_path, change, match):
    value = corpus(tmp_path)
    change(value)
    with pytest.raises(CorpusError, match=match):
        validate_corpus(value)


def test_expected_tokenizer_and_revision_identity_are_enforced(tmp_path):
    value = corpus(tmp_path)
    with pytest.raises(CorpusError, match="stale"):
        validate_corpus(value, expected_revision="b" * 40)
    changed = dict(FakeTokenizer.identity, chat_template_sha256="e" * 64)
    with pytest.raises(CorpusError, match="identity"):
        validate_corpus(value, expected_tokenizer_identity=changed)


def test_source_record_context_cannot_leak_to_a_different_split(tmp_path):
    value = corpus(tmp_path)
    first = value["examples"][0]
    copied = deepcopy(first)
    copied.update(trajectory_id="different", conversation_id="different",
                  split="heldout" if first["split"] == "calibration" else "calibration")
    value["examples"].append(copied)
    with pytest.raises(CorpusError, match="leaks"):
        validate_corpus(value)


def test_conversation_cannot_cross_splits_even_when_source_ids_are_distinct(tmp_path):
    value = corpus(tmp_path)
    first = value["examples"][0]
    copied = deepcopy(first)
    copied.update(trajectory_id="different", source_record_ids=["unrelated"],
                  split="heldout" if first["split"] == "calibration" else "calibration")
    value["examples"].append(copied)
    with pytest.raises(CorpusError, match="conversation crosses"):
        validate_corpus(value)


def test_out_of_order_source_timestamps_are_rejected(tmp_path):
    path = write_memory(tmp_path / "memory.jsonl", [row("one", 2, "user", "Later."),
                                                         row("one", 1, "assistant", "Earlier.")])
    with pytest.raises(CorpusError, match="chronological"):
        read_memory_sources([path])


def test_conflicting_source_ids_are_rejected_and_identical_duplicates_deduplicate(tmp_path):
    first = row("one", 1, "user", "Question.")
    path = write_memory(tmp_path / "memory.jsonl", [first, first])
    assert read_memory_sources([path]).counts["duplicate_source_records"] == 1
    write_memory(path, [first, dict(first, text="A changed question.")])
    with pytest.raises(CorpusError, match="reused"):
        read_memory_sources([path])


def test_source_budgets_reject_before_import(tmp_path):
    path = write_memory(tmp_path / "memory.jsonl", [row("one", 0, "user", "Question.")])
    with pytest.raises(CorpusError, match="byte budget"):
        read_memory_sources([path], max_source_bytes=1)
    write_memory(path, [row("one", 0, "user", "Question."), row("one", 1, "assistant", "Answer.")])
    with pytest.raises(CorpusError, match="record count"):
        read_memory_sources([path], max_records=1)


def test_missing_source_ids_and_naive_timestamps_fail_closed(tmp_path):
    first = row("one", 0, "user", "Question.")
    first.pop("session_id")
    path = write_memory(tmp_path / "memory.jsonl", [first])
    with pytest.raises(CorpusError, match="session_id"):
        read_memory_sources([path])
    first = row("one", 0, "user", "Question.", timestamp="2026-10-02T01:00:00")
    write_memory(path, [first])
    with pytest.raises(CorpusError, match="UTC offset"):
        read_memory_sources([path])


def test_export_accounts_exact_bytes_keeps_operator_files_local_and_never_overwrites(tmp_path):
    source = archive(tmp_path)
    value, refs = build_corpus(source, FakeTokenizer())
    output = tmp_path / "export"
    summary = export_corpus(value, source.source_records + refs,
                           [{"record": {"text": "operator-only"}, "prompt_eligible": False}], output)
    assert summary["total_export_bytes"] == sum(path.stat().st_size for path in output.iterdir())
    assert summary["corpus_sha256"] == hashlib.sha256((output / "corpus.json").read_bytes()).hexdigest()
    assert summary["paid_compute_used"] is False
    assert "operator-only" not in (output / "corpus.json").read_text(encoding="utf-8")
    assert "operator-only" in (output / "operator_observations.local.jsonl").read_text(encoding="utf-8")
    assert [row["file"] for row in summary["files"] if row["cloud_capture_input"]] == ["corpus.json"]
    with pytest.raises(CorpusError, match="new output"):
        export_corpus(value, [], [], output)


def test_export_budget_failure_leaves_no_output(tmp_path):
    value = corpus(tmp_path)
    with pytest.raises(CorpusError, match="byte budget"):
        export_corpus(value, [], [], tmp_path / "too-large", max_output_bytes=1)
    assert not (tmp_path / "too-large").exists()


def test_explicit_probe_framings_use_native_last_token_without_name_prompting(tmp_path):
    value = corpus(tmp_path)
    path = tmp_path / "framings.json"
    path.write_text(json.dumps({"secret_operator_name": {"framings": ["The evidence agrees.", "The evidence conflicts."]}}), encoding="utf-8")
    tokenizer = FakeTokenizer()
    result, rows = add_probe_framings(value, tokenizer, path)
    frames = [example for example in result["examples"] if example["kind"] == "probe_framing"]
    assert {example["polarity"] for example in frames} == {-1, 1}
    assert len({example["probe_id"] for example in frames}) == 1
    assert all(example["selected_positions"] == [len(example["token_ids"])-1] for example in frames)
    assert all(not example["pairs"] for example in frames)
    assert "secret_operator_name" not in json.dumps(tokenizer.contexts)
    assert [row["text"] for row in rows] == ["The evidence agrees.", "The evidence conflicts."]
    assert result["examples"][:2] == frames
    assert result["examples"][2:] == value["examples"]


def test_missing_probe_pole_and_truncated_probe_framings_fail(tmp_path):
    value = corpus(tmp_path)
    path = tmp_path / "framings.json"
    path.write_text(json.dumps({"name": {"framings": ["Positive example.", "Negative example."]}}), encoding="utf-8")
    result, _ = add_probe_framings(value, FakeTokenizer(), path)
    result["examples"].pop(1)
    with pytest.raises(CorpusError, match="both explicitly paired"):
        validate_corpus(result)
    with pytest.raises(CorpusError, match="without changing"):
        add_probe_framings(value, FakeTokenizer(), path, max_tokens=2)


def native_metadata(tmp_path, *, vocab_size=5):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "Question": 1, "Answer": 2, "user": 3, "assistant": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    files = {"tokenizer.json": tokenizer.to_str().encode(),
             "tokenizer_config.json": json.dumps({"eos_token": "[UNK]"}).encode(),
             "config.json": json.dumps({"model_type": "glm_moe_dsa", "vocab_size": vocab_size}).encode(),
             "chat_template.jinja": b"{% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}"}
    manifest = {"id": MODEL_ID, "sha": PINNED_REVISION, "siblings": []}
    receipt = {"repository": MODEL_ID, "revision": PINNED_REVISION, "files": []}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
        git_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        manifest["siblings"].append({"rfilename": name, "size": len(data), "blobId": git_sha})
        receipt["files"].append({"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "metadata_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return tmp_path


def test_native_tokenizer_uses_pinned_template_and_actual_backend_without_model(tmp_path):
    native = NativeTokenizer(native_metadata(tmp_path))
    ids, digest = native.encode_context([{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}])
    assert ids == [3, 1, 4, 2]
    assert len(digest) == 64
    assert native.identity["vocab_size"] == 5
    assert native.revision == PINNED_REVISION


@pytest.mark.parametrize("file", ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"])
def test_tampered_native_metadata_is_rejected_by_pinned_content(file, tmp_path):
    directory = native_metadata(tmp_path)
    (directory / file).write_bytes((directory / file).read_bytes() + b" ")
    with pytest.raises(CorpusError, match="length|checksum|blob"):
        NativeTokenizer(directory)


def test_native_tokenizer_rejects_output_head_mismatch_and_missing_template(tmp_path):
    directory = native_metadata(tmp_path, vocab_size=4)
    with pytest.raises(CorpusError, match="output head"):
        NativeTokenizer(directory)
    (directory / "chat_template.jinja").unlink()
    with pytest.raises(CorpusError, match="missing chat_template"):
        NativeTokenizer(directory)


def test_native_tokenizer_rejects_stale_model_revision(tmp_path):
    directory = native_metadata(tmp_path)
    with pytest.raises(CorpusError, match="model/revision"):
        NativeTokenizer(directory, expected_revision="b" * 40)
    with pytest.raises(CorpusError, match="immutable"):
        NativeTokenizer(directory, expected_revision="main")
