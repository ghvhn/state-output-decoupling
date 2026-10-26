"""Download one pinned eight-expert GLM-5.3 gate, up, or down projection sample."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tda.glm_weight_sample import PINNED_REVISION, PROJECTION, PROJECTIONS, SampleError, download_sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--byte-cap', type=int, required=True,
                        help='Total permitted network bytes, including bounded shard headers; at most 128 MiB.')
    parser.add_argument('--revision', default=PINNED_REVISION,
                        help='Full pinned commit SHA, which must match the saved official metadata.')
    parser.add_argument('--projection', choices=PROJECTIONS, default=PROJECTION,
                        help='Projection to acquire for layer 3 experts 0..7; down inputs are the gated intermediate activations.')
    args = parser.parse_args(argv)
    workspace = Path(__file__).resolve().parents[1]
    metadata, output = args.metadata.resolve(), args.output.resolve()
    for path in (metadata, output):
        if not path.is_relative_to(workspace) or path == workspace or any(
                part in ('.git', '.venv', '.codex', '.agents') for part in path.relative_to(workspace).parts):
            parser.error('Metadata and output must be ordinary subdirectories of this repository workspace.')
    try:
        report = download_sample(metadata, output, byte_cap=args.byte_cap, expected_revision=args.revision,
                                 projection=args.projection)
    except (SampleError, OSError) as exc:
        print(f'Sample stopped: {type(exc).__name__}. Whole-shard downloads remain disabled.')
        if isinstance(exc, SampleError):
            print(str(exc))
        return 1
    print(f"Saved eight layer 3 {args.projection} projections at revision {report['identity']['revision']}.")
    print(f"Source payload bytes including scales: {report['source_tensor_bytes_including_scales']}.")
    print(f"Network bytes including headers: {report['network_bytes']}; cap: {report['byte_cap']}.")
    print(f"Sample manifest: {output / 'sample_manifest.json'}")
    print('This CPU sample is not a runnable model or a validated compressed checkpoint.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
