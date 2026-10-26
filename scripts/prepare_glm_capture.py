"""Prepare reviewable, pinned native GLM capture inputs without models or network."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tda.glm_capture_corpus import (CorpusError, NativeTokenizer, PINNED_REVISION,
    add_probe_framings, build_corpus, export_corpus, read_memory_sources)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "invariants/out/glm_5_3_preflight")
    parser.add_argument("--memory", type=Path, action="append", help="Full memory JSONL; repeat for independent sources.")
    parser.add_argument("--out", type=Path, required=True, help="New workspace output directory for local review.")
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--max-per-conversation", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-positions", type=int, default=32)
    parser.add_argument("--max-context-turns", type=int, default=12)
    parser.add_argument("--max-source-bytes", type=int, default=128 << 20)
    parser.add_argument("--max-output-bytes", type=int, default=128 << 20)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="glm-capture-v1")
    parser.add_argument("--no-reference-fixtures", action="store_true")
    parser.add_argument("--framings-json", type=Path, help="Optional explicit JSON text framings; no saved model vectors are read.")
    args = parser.parse_args(argv)
    try:
        tokenizer = NativeTokenizer(args.metadata, expected_revision=args.revision)
        archive = read_memory_sources(args.memory or [ROOT / "invariants/out/interactive_memory.jsonl"],
                                      max_source_bytes=args.max_source_bytes)
        corpus, references = build_corpus(archive, tokenizer,
            include_reference=not args.no_reference_fixtures, max_examples=args.max_examples,
            max_examples_per_conversation=args.max_per_conversation, max_tokens=args.max_tokens,
            max_positions=args.max_positions, max_context_turns=args.max_context_turns,
            seed=args.seed, heldout_fraction=args.heldout_fraction)
        framing_sources = []
        if args.framings_json:
            corpus, framing_sources = add_probe_framings(corpus, tokenizer, args.framings_json, max_tokens=args.max_tokens)
        summary = export_corpus(corpus, archive.source_records + references + framing_sources,
                                archive.observations, args.out, max_output_bytes=args.max_output_bytes)
    except (CorpusError, OSError, ValueError) as exc:
        parser.exit(2, f"Native corpus preparation stopped: {exc}\n")
    print(json.dumps({key: summary[key] for key in ("examples", "examples_by_split", "input_tokens",
        "selected_positions", "within_trajectory_pairs", "total_export_bytes", "corpus_sha256")}, indent=2))
    print(f"Reviewable corpus: {args.out / 'corpus.json'}")
    print("Tokens contain actual conversation content. Source text and operator observations are separate local review files.")
    print("No model, network, GPU or paid compute was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
