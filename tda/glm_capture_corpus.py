"""Bounded, offline preparation of native GLM teacher-forced capture inputs.

The archive's complete user/assistant records are the source. ProbeBehavior's
short examples, operator measurements and cached activations are never used as
conversation context. Tokens retain the actual private text; hashing metadata
does not anonymize a corpus. No model is loaded and no network request is made.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import uuid

MODEL_ID = "zai-org/GLM-5.3"
PINNED_REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
FORMAT = "glm_native_capture_corpus_v1"
MAX_SOURCE_BYTES = 128 << 20
MAX_RECORD_BYTES = 4 << 20
MAX_RECORDS = 50000
MAX_OUTPUT_BYTES = 128 << 20
_OPERATOR_TAGS = frozenset({"operator_observation", "operator_log", "tool_result",
                          "measurement", "probe_reading"})
_COMMAND = re.compile(r"^:[a-zA-Z_][a-zA-Z_0-9]*(?:\s|$)")


class CorpusError(ValueError):
    """Invalid, stale, ambiguous or over-budget corpus evidence."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive(value: int, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise CorpusError(f"{name} must be an integer in [1, {maximum}].")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"{name} must be a nonempty string.")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CorpusError(f"{name} must be a lowercase SHA256.")
    return value


def _bounded_json(path: Path, limit: int) -> tuple[Any, bytes]:
    if path.stat().st_size > limit:
        raise CorpusError(f"{path.name} exceeds its byte budget.")
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"{path.name} is not valid UTF-8 JSON.") from exc
    return value, data


def _verified_file(directory: Path, name: str, manifest: dict, receipt: dict | None,
                   *, limit: int) -> bytes:
    path = directory / name
    if not path.is_file():
        raise CorpusError(f"Pinned native metadata is missing {name}.")
    if path.stat().st_size > limit:
        raise CorpusError(f"{name} exceeds its metadata byte budget.")
    data = path.read_bytes()
    entry = next((row for row in manifest.get("siblings", [])
                  if isinstance(row, dict) and row.get("rfilename") == name), None)
    if not entry:
        raise CorpusError(f"{name} is not present in the pinned manifest.")
    size = entry.get("size", entry.get("lfs", {}).get("size"))
    if isinstance(size, bool) or not isinstance(size, int) or size != len(data):
        raise CorpusError(f"{name} length does not match the pinned manifest.")
    lfs_sha = entry.get("lfs", {}).get("sha256")
    blob = entry.get("blobId")
    sha = hashlib.sha256(data).hexdigest()
    if isinstance(lfs_sha, str) and re.fullmatch(r"[0-9a-f]{64}", lfs_sha):
        if sha != lfs_sha:
            raise CorpusError(f"{name} checksum does not match the pinned manifest.")
    elif isinstance(blob, str) and re.fullmatch(r"[0-9a-f]{40}", blob):
        git_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if git_sha != blob:
            raise CorpusError(f"{name} Git blob does not match the pinned manifest.")
    else:
        raise CorpusError(f"{name} lacks a pinned content checksum.")
    if receipt:
        receipts = receipt.get("files", [])
        receipt_entry = next((row for row in receipts if row.get("file") == name), None)
        if receipt_entry and (receipt_entry.get("sha256") != sha or receipt_entry.get("bytes") != len(data)):
            raise CorpusError(f"{name} differs from its metadata receipt.")
    return data


class NativeTokenizer:
    """Pinned local tokenizers backend plus the official sandboxed chat template."""

    def __init__(self, metadata_dir: str | Path, *, expected_revision: str = PINNED_REVISION):
        if not isinstance(expected_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
            raise CorpusError("Native capture requires an immutable full commit revision.")
        directory = Path(metadata_dir)
        manifest, _ = _bounded_json(directory / "manifest.json", 2 << 20)
        if not isinstance(manifest, dict) or manifest.get("id") != MODEL_ID or manifest.get("sha") != expected_revision:
            raise CorpusError("Native metadata model/revision differs from the requested source.")
        receipt = None
        if (directory / "metadata_receipt.json").is_file():
            receipt, _ = _bounded_json(directory / "metadata_receipt.json", 1 << 20)
            if receipt.get("repository") != MODEL_ID or receipt.get("revision") != expected_revision:
                raise CorpusError("Metadata receipt has a stale model/revision.")
        config_data = _verified_file(directory, "config.json", manifest, receipt, limit=1 << 20)
        tokenizer_data = _verified_file(directory, "tokenizer.json", manifest, receipt, limit=64 << 20)
        tokenizer_config_data = _verified_file(directory, "tokenizer_config.json", manifest, receipt, limit=1 << 20)
        template_data = _verified_file(directory, "chat_template.jinja", manifest, receipt, limit=1 << 20)
        config, tokenizer_config = json.loads(config_data), json.loads(tokenizer_config_data)
        if config.get("model_type") != "glm_moe_dsa":
            raise CorpusError("Native model configuration has an incompatible architecture.")
        self.vocab_size = _positive(config.get("vocab_size"), "native vocabulary size", maximum=2_000_000)
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_str(tokenizer_data.decode("utf-8"))
        vocabulary = self.backend.get_vocab()
        if not vocabulary or max(vocabulary.values()) >= self.vocab_size:
            raise CorpusError("Native tokenizer IDs exceed the pinned output head vocabulary.")
        self.backend.no_truncation()
        self.backend.no_padding()
        self.template = template_data.decode("utf-8")
        self.special_tokens = {key: value for key, value in tokenizer_config.items()
                               if key in {"bos_token", "eos_token", "pad_token", "unk_token"} and isinstance(value, str)}
        self.model_id, self.revision = MODEL_ID, expected_revision
        self.identity = {
            "tokenizer_sha256": hashlib.sha256(tokenizer_data).hexdigest(),
            "tokenizer_config_sha256": hashlib.sha256(tokenizer_config_data).hexdigest(),
            "chat_template_sha256": hashlib.sha256(template_data).hexdigest(),
            "config_sha256": hashlib.sha256(config_data).hexdigest(),
            "vocab_size": self.vocab_size,
            "actual_tokenizer_vocabulary": len(vocabulary),
        }

    def encode_context(self, messages: list[dict[str, str]], *, add_generation_prompt: bool = False) -> tuple[list[int], str]:
        from transformers.utils.chat_template_utils import render_jinja_template
        try:
            rendered, _ = render_jinja_template([messages], chat_template=self.template,
                add_generation_prompt=add_generation_prompt, clear_thinking=False,
                reasoning_effort="max", **self.special_tokens)
            text = rendered[0]
            ids = self.backend.encode(text, add_special_tokens=False).ids
        except Exception as exc:
            raise CorpusError("The pinned native chat template could not be rendered/tokenized offline.") from exc
        if not ids or any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < self.vocab_size for t in ids):
            raise CorpusError("Native tokenization returned invalid output-head token IDs.")
        return ids, hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Conversation:
    conversation_id: str
    environment_id: str
    selection_bucket: str
    records: list[dict]


@dataclass
class SourceArchive:
    conversations: list[Conversation]
    source_records: list[dict]
    observations: list[dict]
    sources: list[dict]
    counts: dict


def read_memory_sources(paths: list[str | Path], *, max_source_bytes: int = MAX_SOURCE_BYTES,
                        max_records: int = MAX_RECORDS) -> SourceArchive:
    """Read full memory JSONL; maintain conversation chronology and context boundaries."""
    _positive(max_source_bytes, "source byte budget", maximum=1 << 30)
    _positive(max_records, "source record budget", maximum=1_000_000)
    if not isinstance(paths, (list, tuple)) or not paths:
        raise CorpusError("Supply at least one full source memory JSONL.")
    grouped, records, observations, sources, seen = defaultdict(list), [], [], [], {}
    count = Counter()
    total_bytes = 0
    last_times = {}
    for item in paths:
        path = Path(item)
        total_bytes += path.stat().st_size
        if total_bytes > max_source_bytes:
            raise CorpusError("Combined memory sources exceed the byte budget.")
        source_hash = hashlib.sha256()
        staged = []
        with path.open("rb") as handle:
            line_number = 0
            while True:
                raw = handle.readline(MAX_RECORD_BYTES + 1)
                if not raw:
                    break
                line_number += 1
                count["read_records"] += 1
                if count["read_records"] > max_records or len(raw) > MAX_RECORD_BYTES:
                    raise CorpusError("Memory record count or line bytes exceed the budget.")
                source_hash.update(raw)
                try:
                    row = json.loads(raw)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise CorpusError(f"Memory source line {line_number} is not valid JSON.") from exc
                if not isinstance(row, dict):
                    raise CorpusError("Memory source records must be JSON objects.")
                provenance = {"source_file": path.name, "source_line": line_number}
                eligible = row.get("kind") == "turn" and row.get("role") in {"user", "assistant"}
                tags = row.get("tags", [])
                text = row.get("text")
                if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
                    raise CorpusError("Memory record tags must be a list of strings.")
                operator = bool(_OPERATOR_TAGS.intersection(tags)) or (
                    row.get("role") == "user" and isinstance(text, str) and bool(_COMMAND.match(text.lstrip())))
                if not eligible or operator or not isinstance(text, str) or not text.strip():
                    observations.append({"record": row, "source": provenance, "prompt_eligible": False})
                    count["operator_or_nonconversation_records"] += 1
                    continue
                for key in ("record_id", "scope", "session_id", "timestamp"):
                    _nonempty(row.get(key), f"conversation {key}")
                try:
                    timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                except ValueError as exc:
                    raise CorpusError("Conversation record has an invalid timestamp.") from exc
                if timestamp.tzinfo is None:
                    raise CorpusError("Conversation timestamps must include their UTC offset.")
                group = (row["scope"], row["session_id"])
                stable = {key: row.get(key) for key in ("record_id", "scope", "session_id", "timestamp", "role", "text", "turn_id", "parent_turn_id")}
                fingerprint = _digest(stable)
                if row["record_id"] in seen:
                    if seen[row["record_id"]] != fingerprint:
                        raise CorpusError("A source record ID was reused with different conversation content.")
                    count["duplicate_source_records"] += 1
                    continue
                seen[row["record_id"]] = fingerprint
                if group in last_times and timestamp < last_times[group]:
                    raise CorpusError("Conversation records are not in chronological source order.")
                last_times[group] = timestamp
                clean = {**stable, "source": provenance, "source_record_sha256": fingerprint}
                records.append(clean)
                staged.append(clean)
                grouped[group].append(clean)
                count["full_conversation_records"] += 1
        digest = source_hash.hexdigest()
        for clean in staged:
            clean["source"]["source_sha256"] = digest
        sources.append({"file": path.name, "sha256": digest, "bytes": path.stat().st_size})
    conversations = []
    for group, rows in sorted(grouped.items()):
        group_id = "memory:" + _digest(list(group))[:24]
        conversations.append(Conversation(group_id, group_id, "archive", rows))
    return SourceArchive(conversations, records, observations, sources, dict(count))


def reference_conversations() -> list[Conversation]:
    """Independent, explicit neutral reference conversations; labels are metadata only."""
    examples = {
        "definitions": [
            ("What distinguishes a rectangle from a general quadrilateral?", "A rectangle is a quadrilateral whose four interior angles are right angles. Four sides alone do not distinguish it."),
            ("Define a hypothesis without treating it as an established fact.", "A hypothesis is a proposed explanation that can be examined against evidence. Its proposal does not establish its truth."),
            ("What is a checksum, and what does it fail to establish?", "A checksum is a value computed from data to help detect changes. Matching an untrusted checksum does not establish who supplied the data."),
            ("Define a queue by the ordering rule, rather than by a particular implementation.", "A queue is a collection whose removal rule normally selects the item that entered earliest. Arrays and linked lists can both implement that rule."),
        ],
        "corrections": [
            ("I said 12 multiplied by 8 equals 86. Please check that claim.", "The product is 96. Multiplying 12 by 4 gives 48, and doubling 48 gives 96; the earlier value was an arithmetic error."),
            ("I called every four-sided shape a square. How should that statement change?", "Every square has four sides, but the converse fails. A square also requires equal side lengths and four right angles."),
            ("I assumed that correlation proves a causal mechanism. Is that inference sound?", "Correlation alone does not establish a causal mechanism. A common cause or selection effect could generate the same observed association."),
            ("An earlier report says the test passed, but the saved exit status is one. What should the corrected report say?", "The recorded run failed. The report should describe the failure and reserve a passing claim for a subsequent verified run."),
        ],
        "arithmetic": [
            ("A box contains seven packs of six pencils. Nine pencils are removed. How many remain?", "There were 7 times 6, or 42 pencils. Removing 9 leaves 33 pencils."),
            ("A trip takes 90 minutes. A break adds 15 minutes, and a shortcut saves 20. What is the total?", "The total is 90 plus 15 minus 20, which is 85 minutes."),
            ("Five items cost eight units each, and the fifth item receives a fifty percent discount. Find the total.", "Four items cost 32 units. The discounted item costs 4 units, so the total is 36 units."),
            ("Three quarters of a quantity is 24. What is the original quantity?", "One quarter is 8, so four quarters give an original quantity of 32."),
        ],
        "evidence": [
            ("Two runs chose the same next word, but their probability distributions changed. What follows?", "The chosen word was preserved in those runs. That observation does not establish that the underlying distributions or later continuations were preserved."),
            ("A measurement was recorded correctly, but the prediction it tested was wrong. Should we discard the measurement?", "The measurement remains evidence about that prediction in its recorded setting. Its validity and the prediction's accuracy are separate questions."),
            ("A result was measured in one conversation. What is needed before assuming it holds in another?", "The new setting needs its own supporting measurements or a validated transfer argument. A similarity in wording alone is not enough."),
            ("Only the first of three planned experiments was completed. How should the outcome be described?", "Report the first experiment's measured outcome and mark the other two as unfinished. Planned checks do not count as observed results."),
        ],
    }
    result = []
    for bucket, pairs in examples.items():
        for number, (question, answer) in enumerate(pairs):
            group_id = f"reference:{bucket}:{number}"
            rows = []
            for index, (role, text) in enumerate((("user", question), ("assistant", answer))):
                row = {"record_id": f"{group_id}:{index}", "scope": "reference_fixture",
                       "session_id": group_id, "timestamp": f"2000-01-01T00:00:0{index}+00:00",
                       "role": role, "text": text, "turn_id": str(index), "parent_turn_id": None,
                       "source": {"source_file": "builtin_reference_v1", "source_line": index + 1}}
                row["source_record_sha256"] = _digest(row)
                rows.append(row)
            result.append(Conversation(group_id, group_id, bucket, rows))
    return result


def conversation_splits(conversations: list[Conversation], *, seed: str = "glm-capture-v1",
                        heldout_fraction: float = 0.2) -> dict[str, str]:
    if isinstance(heldout_fraction, bool) or not isinstance(heldout_fraction, (int, float)) or not math.isfinite(heldout_fraction) or not 0 < heldout_fraction < 1:
        raise CorpusError("Held-out fraction must be finite and strictly between zero and one.")
    _nonempty(seed, "split seed")
    grouped, seen = defaultdict(list), set()
    for conversation in conversations:
        if conversation.conversation_id in seen:
            raise CorpusError("Conversation IDs must be unique before splitting.")
        seen.add(conversation.conversation_id)
        grouped[conversation.selection_bucket].append(conversation.conversation_id)
    if len(seen) < 2:
        raise CorpusError("A disjoint calibration/held-out corpus requires at least two conversations.")
    splits = {}
    for ids in grouped.values():
        ordered = sorted(ids, key=lambda group: (_digest([seed, group]), group))
        count = min(len(ordered) - 1, max(1, round(len(ordered) * heldout_fraction))) if len(ordered) > 1 else 0
        for index, group in enumerate(ordered):
            splits[group] = "heldout" if index < count else "calibration"
    if "heldout" not in splits.values():
        splits[min(splits, key=lambda group: _digest([seed, group]))] = "heldout"
    return splits


def _spaced_indices(count: int, keep: int) -> list[int]:
    if count <= keep:
        return list(range(count))
    if keep == 1:
        return [count - 1]
    return sorted({round(index * (count - 1) / (keep - 1)) for index in range(keep)})


def selected_timeline(token_count: int, *, max_positions: int = 32) -> tuple[list[int], list[list[int]]]:
    _positive(max_positions, "selected position budget", maximum=4096)
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 3:
        raise CorpusError("A measured trajectory requires at least three native tokens.")
    if max_positions < 2:
        return [token_count - 2], []
    starts = _spaced_indices(token_count - 2, max(1, max_positions // 2))
    pairs, used = [], set()
    for start in starts:
        if start + 1 >= token_count - 1:
            continue
        pair = [start, start + 1]
        pairs.append(pair)
        used.update(pair)
    return sorted(used), pairs


def _context_example(conversation: Conversation, endpoint: int, tokenizer: NativeTokenizer, split: str,
                     *, max_tokens: int, max_positions: int, max_context_turns: int) -> dict | None:
    rows = conversation.records[:endpoint + 1]
    user_indices = [index for index, row in enumerate(rows) if row["role"] == "user"]
    if not user_indices or rows[-1]["role"] != "assistant":
        return None
    start = user_indices[0]
    while len(rows) - start > max_context_turns and any(index > start for index in user_indices):
        start = next(index for index in user_indices if index > start)
    while True:
        chosen = rows[start:]
        tokens, rendered_hash = tokenizer.encode_context([
            {"role": row["role"], "content": row["text"]} for row in chosen])
        later_users = [index for index in user_indices if index > start]
        if len(tokens) <= max_tokens or not later_users:
            break
        start = later_users[0]
    rendered_count = len(tokens)
    tokens = tokens[:max_tokens]
    if len(tokens) < 3:
        return None
    positions, pairs = selected_timeline(len(tokens), max_positions=max_positions)
    trajectory = conversation.conversation_id + ":" + rows[-1]["record_id"]
    return {"trajectory_id": trajectory, "conversation_id": conversation.conversation_id,
            "environment_id": conversation.environment_id, "split": split, "kind": "conversation",
            "token_ids": tokens, "selected_positions": positions, "pairs": pairs,
            "source_record_ids": [row["record_id"] for row in chosen],
            "source_sha256": _digest([row["source_record_sha256"] for row in chosen]),
            "selection_bucket": conversation.selection_bucket,
            "provenance_weight": 1.0, "stimulation_weight": 1.0,
            "context": {"dropped_prefix_record_ids": [row["record_id"] for row in rows[:start]],
                        "rendered_sha256": rendered_hash, "rendered_token_count": rendered_count,
                        "retained_token_range": [0, len(tokens)],
                        "token_prefix_truncated": rendered_count > len(tokens),
                        "template_options": {"add_generation_prompt": False, "clear_thinking": False, "reasoning_effort": "max"}},
            "observation_contract": "native teacher-forced replay of archived text; original model hidden/tool/system context is not reconstructed"}


def _balanced_take(examples: list[dict], count: int) -> list[dict]:
    groups = defaultdict(list)
    for example in examples:
        groups[(example["split"], example["selection_bucket"])].append(example)
    for values in groups.values():
        values.sort(key=lambda row: row["trajectory_id"])
    by_split = {split: [groups[key] for key in sorted(groups) if key[0] == split]
                for split in ("calibration", "heldout")}
    # Even a two-example budget gets one context from each split. Continue
    # alternating split/bucket queues rather than exhausting training buckets.
    queues = []
    for index in range(max(map(len, by_split.values()), default=0)):
        for split in ("calibration", "heldout"):
            if index < len(by_split[split]):
                queues.append(by_split[split][index])
    selected = []
    depth = 0
    while len(selected) < count:
        progress = False
        for values in queues:
            if depth < len(values) and len(selected) < count:
                selected.append(values[depth])
                progress = True
        if not progress:
            break
        depth += 1
    return selected


def build_corpus(archive: SourceArchive, tokenizer: NativeTokenizer, *, include_reference: bool = True,
                 max_examples: int = 256, max_examples_per_conversation: int = 4,
                 max_tokens: int = 1024, max_positions: int = 32, max_context_turns: int = 12,
                 seed: str = "glm-capture-v1", heldout_fraction: float = 0.2) -> tuple[dict, list[dict]]:
    _positive(max_examples, "example budget", maximum=10000)
    if max_examples < 2:
        raise CorpusError("At least two examples are required for disjoint splits.")
    _positive(max_examples_per_conversation, "per-conversation example budget", maximum=256)
    _positive(max_tokens, "sequence token budget", maximum=131072)
    _positive(max_context_turns, "context turn budget", maximum=256)
    _positive(max_positions, "selected position budget", maximum=4096)
    conversations = list(archive.conversations)
    references = reference_conversations() if include_reference else []
    conversations.extend(references)
    # Only complete user/assistant trajectories are eligible for allocation.
    conversations = [conversation for conversation in conversations if any(
        row["role"] == "assistant" and any(previous["role"] == "user" for previous in conversation.records[:index])
        for index, row in enumerate(conversation.records))]
    splits = conversation_splits(conversations, seed=seed, heldout_fraction=heldout_fraction)
    examples = []
    for conversation in conversations:
        endpoints = [index for index, row in enumerate(conversation.records) if row["role"] == "assistant"]
        for index in _spaced_indices(len(endpoints), max_examples_per_conversation):
            example = _context_example(conversation, endpoints[index], tokenizer, splits[conversation.conversation_id],
                       max_tokens=max_tokens, max_positions=max_positions, max_context_turns=max_context_turns)
            if example:
                examples.append(example)
    examples = _balanced_take(examples, max_examples)
    if {row["split"] for row in examples} != {"calibration", "heldout"}:
        raise CorpusError("The selected budget did not retain both disjoint splits.")
    corpus = {"format": FORMAT, "model_id": tokenizer.model_id, "revision": tokenizer.revision,
              "tokenizer_identity": dict(tokenizer.identity),
              "template_options": {"add_generation_prompt": False, "clear_thinking": False, "reasoning_effort": "max"},
              "split_policy": {"method": "seeded whole-conversation hash order within selection buckets",
                               "seed": seed, "heldout_fraction": heldout_fraction},
              "context_policy": "whole-turn context window followed by an explicitly recorded native token prefix cap",
              "sources": archive.sources, "examples": examples,
              "privacy": {"contains_actual_conversation_tokens": True, "hashes_anonymize_content": False,
                          "operator_records_in_prompt": False, "source_text_jsonl_for_local_review_only": True},
              "capture_contract": {"batch_examples_independent": True, "right_padding_requires_attention_mask": True,
                  "selected_positions_are_zero_based": True, "positions_cover_all_native_roles": True,
                  "next_token_labels_are_observations_not_correctness_judgments": True}}
    validate_corpus(corpus)
    return corpus, [row for conversation in references for row in conversation.records]


def add_probe_framings(corpus: dict, tokenizer: NativeTokenizer, path: str | Path, *, max_probes: int = 128,
                       max_tokens: int = 1024) -> tuple[dict, list[dict]]:
    """Explicit JSON framings only; no old model vectors or probe names enter text."""
    _positive(max_probes, "probe framing budget", maximum=2048)
    data, raw = _bounded_json(Path(path), 4 << 20)
    if not isinstance(data, dict) or len(data) > max_probes:
        raise CorpusError("Framings require a bounded JSON object keyed by operator probe name.")
    result = dict(corpus, examples=[])
    source_rows = []
    for name, payload in sorted(data.items()):
        _nonempty(name, "operator probe name")
        frames = payload.get("framings") if isinstance(payload, dict) else None
        if isinstance(frames, dict):
            pair = [frames.get("a"), frames.get("b")]
        elif isinstance(frames, (list, tuple)) and len(frames) == 2:
            pair = list(frames)
        else:
            raise CorpusError("Each probe must supply exactly two stored text framings.")
        for frame in pair:
            _nonempty(frame, "probe framing")
        if pair[0] == pair[1]:
            raise CorpusError("Probe framings must have a nonzero textual contrast.")
        probe_id = "p_" + _digest([tokenizer.model_id, tokenizer.revision, name, pair])[:24]
        group_id = "framing:" + probe_id
        for index, frame in enumerate(pair):
            tokens, rendered_hash = tokenizer.encode_context([{"role": "user", "content": frame}], add_generation_prompt=True)
            if len(tokens) > max_tokens:
                raise CorpusError("Probe framings must fit without changing their last-token measurement site.")
            record_id = group_id + ":" + str(index)
            result["examples"].append({"trajectory_id": record_id, "conversation_id": group_id,
                "environment_id": group_id, "split": "calibration", "kind": "probe_framing",
                "probe_id": probe_id, "polarity": 1 if index == 0 else -1,
                "token_ids": tokens, "selected_positions": [len(tokens) - 1], "pairs": [],
                "source_record_ids": [record_id], "source_sha256": hashlib.sha256(raw).hexdigest(),
                "selection_bucket": "probe_framings", "provenance_weight": 1.0, "stimulation_weight": 1.0,
                "context": {"rendered_sha256": rendered_hash, "rendered_token_count": len(tokens),
                            "retained_token_range": [0, len(tokens)], "token_prefix_truncated": False,
                            "dropped_prefix_record_ids": [],
                            "template_options": {"add_generation_prompt": True, "clear_thinking": False, "reasoning_effort": "max"}},
                "observation_contract": "last-token native contrast framing; no observed next-token label at this position"})
            source_rows.append({"record_id": record_id, "role": "user", "text": frame,
                                "source": {"source_file": Path(path).name, "source_sha256": hashlib.sha256(raw).hexdigest()}})
    # Cheap paired seed states come first, so a partial GPU capture can still
    # support native probe calibration. Conversation split/order is untouched.
    result["examples"].extend(corpus["examples"])
    result["capture_priority"] = "paired probe seed states, then split-balanced conversation contexts"
    validate_corpus(result)
    return result, source_rows


def validate_corpus(corpus: dict, *, expected_revision: str | None = None,
                    expected_tokenizer_identity: dict | None = None) -> None:
    if not isinstance(corpus, dict) or corpus.get("format") != FORMAT or corpus.get("model_id") != MODEL_ID:
        raise CorpusError("Unsupported native capture corpus model/format.")
    revision = corpus.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision) or (expected_revision is not None and revision != expected_revision):
        raise CorpusError("Capture corpus has a stale or mutable source revision.")
    identity = corpus.get("tokenizer_identity")
    if not isinstance(identity, dict):
        raise CorpusError("Capture corpus lacks native tokenizer/head identity.")
    for key in ("tokenizer_sha256", "tokenizer_config_sha256", "chat_template_sha256", "config_sha256"):
        _sha256(identity.get(key), key)
    vocab = _positive(identity.get("vocab_size"), "output-head vocabulary size", maximum=2_000_000)
    if expected_tokenizer_identity is not None and identity != expected_tokenizer_identity:
        raise CorpusError("Corpus tokenizer/template/output-head identity is stale.")
    examples = corpus.get("examples")
    if not isinstance(examples, list) or not examples or len(examples) > 20000:
        raise CorpusError("Corpus examples must be a nonempty bounded list.")
    seen, splits, source_splits = set(), {}, {}
    probe_poles = defaultdict(set)
    for example in examples:
        if not isinstance(example, dict):
            raise CorpusError("Corpus examples must be objects.")
        for key in ("trajectory_id", "conversation_id", "environment_id", "source_sha256"):
            _nonempty(example.get(key), key)
        _sha256(example["source_sha256"], "example source fingerprint")
        if example["trajectory_id"] in seen:
            raise CorpusError("Capture trajectory IDs must be unique.")
        seen.add(example["trajectory_id"])
        split = example.get("split")
        if split not in {"calibration", "heldout"}:
            raise CorpusError("Capture example has an unsupported split.")
        group = example["conversation_id"]
        if group in splits and splits[group] != split:
            raise CorpusError("A conversation crosses calibration/held-out splits.")
        splits[group] = split
        source_ids = example.get("source_record_ids")
        if not isinstance(source_ids, list) or not source_ids or any(not isinstance(item, str) or not item for item in source_ids):
            raise CorpusError("Examples require their complete source record IDs.")
        for source_id in source_ids:
            if source_id in source_splits and source_splits[source_id] != split:
                raise CorpusError("A source context record leaks across splits.")
            source_splits[source_id] = split
        tokens = example.get("token_ids")
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= 131072 or any(
            isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab for token in tokens):
            raise CorpusError("Native token IDs are invalid for the pinned output-head vocabulary.")
        positions = example.get("selected_positions")
        kind = example.get("kind")
        if kind not in {"conversation", "probe_framing"}:
            raise CorpusError("Unsupported capture example kind.")
        last_allowed = len(tokens) - 1 if kind == "probe_framing" else len(tokens) - 2
        if not isinstance(positions, list) or not positions or len(positions) > 4096 or any(
            isinstance(position, bool) or not isinstance(position, int) or not 0 <= position <= last_allowed for position in positions):
            raise CorpusError("Selected positions exceed the measured token/next-token timeline.")
        if positions != sorted(set(positions)):
            raise CorpusError("Selected token positions must be unique and sorted.")
        pairs = example.get("pairs")
        if not isinstance(pairs, list) or len(pairs) > 4096:
            raise CorpusError("Measured trajectory pairs must be a bounded list.")
        pair_seen = set()
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or any(isinstance(p, bool) or not isinstance(p, int) for p in pair) or pair[1] != pair[0] + 1 or not set(pair) <= set(positions):
                raise CorpusError("Trajectory pairs require two selected adjacent positions in the same example.")
            if tuple(pair) in pair_seen:
                raise CorpusError("Trajectory pairs may not be duplicated.")
            pair_seen.add(tuple(pair))
        if kind == "probe_framing":
            probe = _nonempty(example.get("probe_id"), "native probe namespace")
            polarity = example.get("polarity")
            if isinstance(polarity, bool) or polarity not in {-1, 1} or pairs or positions != [len(tokens) - 1]:
                raise CorpusError("Native probe framing must measure one final-token pole without next-token pairs.")
            probe_poles[probe].add(polarity)
        context = example.get("context")
        if not isinstance(context, dict) or context.get("retained_token_range") != [0, len(tokens)]:
            raise CorpusError("Capture context token range differs from its stored token timeline.")
        _sha256(context.get("rendered_sha256"), "rendered context fingerprint")
        rendered = context.get("rendered_token_count")
        if isinstance(rendered, bool) or not isinstance(rendered, int) or rendered < len(tokens):
            raise CorpusError("Rendered context count is shorter than its retained native token prefix.")
    if any(poles != {-1, 1} for poles in probe_poles.values()):
        raise CorpusError("Every native probe requires both explicitly paired framings.")


def export_corpus(corpus: dict, source_records: list[dict], observations: list[dict], output: str | Path,
                  *, max_output_bytes: int = MAX_OUTPUT_BYTES) -> dict:
    """Atomic new-directory export; source texts/observations remain local review files."""
    validate_corpus(corpus)
    _positive(max_output_bytes, "output byte budget", maximum=1 << 30)
    target = Path(output)
    if target.exists():
        raise CorpusError("Corpus export requires a new output directory; existing evidence is retained.")
    corpus_bytes = _canonical(corpus)
    records_bytes = b"".join(_canonical(row) + b"\n" for row in source_records)
    observations_bytes = b"".join(_canonical(row) + b"\n" for row in observations)
    files = {"corpus.json": corpus_bytes, "source_records.local.jsonl": records_bytes,
             "operator_observations.local.jsonl": observations_bytes}
    counts = Counter(row["split"] for row in corpus["examples"])
    bucket_counts = Counter(row["selection_bucket"] for row in corpus["examples"])
    summary = {"format": FORMAT, "model_id": corpus["model_id"], "revision": corpus["revision"],
               "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
               "examples": len(corpus["examples"]), "examples_by_split": dict(counts),
               "examples_by_selection_bucket": dict(bucket_counts),
               "input_tokens": sum(len(row["token_ids"]) for row in corpus["examples"]),
               "selected_positions": sum(len(row["selected_positions"]) for row in corpus["examples"]),
               "within_trajectory_pairs": sum(len(row["pairs"]) for row in corpus["examples"]),
               "source_record_count": len(source_records), "operator_observation_count": len(observations),
               "files": [{"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                          "cloud_capture_input": name == "corpus.json"} for name, data in files.items()],
               "privacy": corpus["privacy"], "model_loaded": False, "network_requests": 0,
               "native_capture_executed": False, "paid_compute_used": False}
    # Account for the size of the byte-count field itself. Its decimal width
    # reaches a fixed point after at most a few iterations.
    summary["total_export_bytes"] = 0
    for _ in range(8):
        files["summary.json"] = _canonical(summary)
        total = sum(map(len, files.values()))
        if total == summary["total_export_bytes"]:
            break
        summary["total_export_bytes"] = total
    files["summary.json"] = _canonical(summary)
    if sum(map(len, files.values())) > max_output_bytes:
        raise CorpusError("Serialized corpus/source/observation export exceeds its byte budget.")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / (target.name + ".preparing-" + uuid.uuid4().hex)
    stage.mkdir()
    for name, data in files.items():
        with (stage / name).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    if target.exists():
        raise CorpusError("Output appeared during corpus preparation; staged files were retained.")
    os.rename(stage, target)
    return summary
