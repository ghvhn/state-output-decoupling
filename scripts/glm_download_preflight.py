"""Fetch pinned public GLM metadata and budget weights without downloading them.

This cannot fetch checkpoint shards, execute model code, load a model, upgrade
dependencies, or compress/export a checkpoint. Offline mode uses saved metadata.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


OFFICIAL_REPO = 'zai-org/GLM-5.3'
MAX_FILE_BYTES = 64 * 1024 ** 2
MAX_MANIFEST_BYTES = 16 * 1024 ** 2
MAX_TOTAL_BYTES = 99 * 1024 ** 2
METADATA_FILES = frozenset({
    'config.json', 'generation_config.json', 'model.safetensors.index.json',
    'pytorch_model.bin.index.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model',
    'vocab.json', 'vocab.txt', 'merges.txt', 'README.md', 'LICENSE',
    'LICENSE.txt', 'LICENSE.md',
})


class PreflightError(ValueError):
    pass


def validate_revision(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{40}', value):
        raise PreflightError('A pinned revision must be a full lowercase 40-character commit SHA.')
    return value


def safe_filename(value):
    if not isinstance(value, str) or not value or not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise PreflightError('Repository filename contains unsupported characters.')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise PreflightError('Repository filename must be a safe relative path.')
    return value


def metadata_allowed(value):
    return safe_filename(value) in METADATA_FILES


def _public_https(url):
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    if (parsed.scheme != 'https' or parsed.username or parsed.password or
            parsed.port not in (None, 443) or not (
                host == 'huggingface.co' or host.endswith('.huggingface.co') or
                host.endswith('.hf.co'))):
        raise PreflightError('Metadata requests must stay on public Hugging Face HTTPS hosts.')
    return url


class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class MetadataClient:
    """Bound every response and the whole operation; never sends credentials."""
    def __init__(self, *, opener=None, max_total_bytes=MAX_TOTAL_BYTES):
        self.opener = opener or build_opener(PublicRedirects())
        self.max_total_bytes = min(int(max_total_bytes), MAX_TOTAL_BYTES)
        if self.max_total_bytes <= 0:
            raise PreflightError('The metadata byte budget must be positive.')
        self.received_bytes = 0

    def get(self, url, *, limit=MAX_FILE_BYTES):
        _public_https(url)
        remaining = self.max_total_bytes - self.received_bytes
        limit = min(int(limit), MAX_FILE_BYTES, remaining)
        if limit <= 0:
            raise PreflightError('The total metadata byte budget is exhausted.')
        request = Request(url, headers={'User-Agent': 'tda-domain-mapper-metadata-preflight/1'})
        with self.opener.open(request, timeout=45) as response:
            _public_https(response.geturl())
            declared = response.headers.get('Content-Length')
            if declared is not None:
                try:
                    size = int(declared)
                except (TypeError, ValueError) as exc:
                    raise PreflightError('Invalid metadata Content-Length.') from exc
                if size < 0 or size > limit:
                    raise PreflightError('A metadata response exceeds its byte budget.')
            parts = []
            length = 0
            while True:
                chunk = response.read(min(1024 ** 2, limit + 1 - length))
                if not chunk:
                    break
                length += len(chunk)
                if length > limit:
                    raise PreflightError('A metadata response exceeds its byte budget.')
                parts.append(chunk)
            self.received_bytes += length
            return b''.join(parts)


def read_json_bytes(data, description):
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise PreflightError(f'{description} is not valid JSON.') from exc
    if not isinstance(value, dict):
        raise PreflightError(f'{description} must contain a JSON object.')
    return value


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest_files(manifest):
    siblings = manifest.get('siblings')
    if not isinstance(siblings, list):
        raise PreflightError('The manifest is missing its file inventory.')
    result = {}
    for row in siblings:
        if not isinstance(row, dict):
            raise PreflightError('Manifest file entries must be objects.')
        name = safe_filename(row.get('rfilename'))
        if name in result:
            raise PreflightError('The manifest contains duplicate filenames.')
        result[name] = row
    return result


def verify_manifest(manifest, expected_revision=None):
    if manifest.get('id') != OFFICIAL_REPO:
        raise PreflightError('The manifest does not describe the official GLM-5.3 repository.')
    revision = validate_revision(manifest.get('sha'))
    if expected_revision is not None and revision != validate_revision(expected_revision):
        raise PreflightError('The returned manifest does not match the pinned revision.')
    _manifest_files(manifest)
    return revision


def fetch_metadata(output, *, revision=None, client=None):
    output = Path(output)
    if revision is not None:
        validate_revision(revision)
    client = client or MetadataClient()
    api = 'https://huggingface.co/api/models/' + OFFICIAL_REPO
    if revision is not None:
        api += '/revision/' + revision
    first = read_json_bytes(client.get(api + '?blobs=true', limit=MAX_MANIFEST_BYTES), 'Manifest')
    pinned = verify_manifest(first, revision)
    if revision is None:
        raw = client.get(api + '/revision/' + pinned + '?blobs=true', limit=MAX_MANIFEST_BYTES)
        manifest = read_json_bytes(raw, 'Pinned manifest')
        verify_manifest(manifest, pinned)
    else:
        manifest = first
        raw = json.dumps(manifest, indent=2).encode('utf-8')
    files = _manifest_files(manifest)
    if 'config.json' not in files:
        raise PreflightError('The manifest does not list config.json.')
    if output.exists() and any(output.iterdir()):
        existing = output / 'manifest.json'
        if not existing.is_file():
            raise PreflightError('Refusing to overwrite an unrelated nonempty output directory.')
        saved = read_json_bytes(_local_metadata(existing, limit=MAX_MANIFEST_BYTES), 'Saved manifest')
        verify_manifest(saved, pinned)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / 'manifest.json', raw)
    saved_files = []
    for name in sorted(files):
        if not metadata_allowed(name):
            continue
        declared = _file_size(files[name])
        if declared is not None and declared > MAX_FILE_BYTES:
            raise PreflightError('A listed metadata file exceeds the per-file byte budget.')
        url = 'https://huggingface.co/' + OFFICIAL_REPO + '/resolve/' + pinned + '/' + quote(name)
        data = client.get(url)
        if name.endswith('.json'):
            read_json_bytes(data, name)
        atomic_write(output / name, data)
        saved_files.append({'file': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    receipt = {
        'repository': OFFICIAL_REPO, 'revision': pinned,
        'downloaded_weight_shards': 0, 'downloaded_model_code_files': 0,
        'network_metadata_bytes': client.received_bytes, 'files': saved_files,
    }
    atomic_write(output / 'metadata_receipt.json', _json(receipt))
    return manifest


def _file_size(row):
    size = row.get('size')
    if size is None and isinstance(row.get('lfs'), dict):
        size = row['lfs'].get('size')
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return size
    return None


def installed_architecture(model_type):
    """Inspect installed dispatch declarations without importing model code."""
    try:
        distribution = importlib.metadata.distribution('transformers')
        path = distribution.locate_file('transformers/models/auto/configuration_auto.py')
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                    target.id == 'CONFIG_MAPPING_NAMES' for target in node.targets):
                value = node.value
                if isinstance(value, ast.Call) and value.args:
                    value = value.args[0]
                entries = dict(ast.literal_eval(value))
                return {'transformers_version': distribution.version,
                        'config_registry_support': model_type in entries,
                        'scope': 'Configuration dispatch only; runtime, hooks, and compression support are untested.'}
    except (importlib.metadata.PackageNotFoundError, OSError, SyntaxError, ValueError, TypeError):
        pass
    return {'transformers_version': None, 'config_registry_support': None,
            'scope': 'Installed configuration registry could not be inspected.'}


def _capacity_bytes(gib):
    if gib is None:
        return None
    if not isinstance(gib, (int, float)) or isinstance(gib, bool) or not math.isfinite(gib) or gib < 0:
        raise PreflightError('Hardware capacities must be finite nonnegative GiB values.')
    return int(gib * 1024 ** 3)


def make_plan(manifest, config, *, index=None, ram_gib=None, vram_gib=None, free_disk_bytes=None,
              architecture=None):
    revision = verify_manifest(manifest)
    files = _manifest_files(manifest)
    warnings = []
    shard_names = sorted(name for name in files if name.endswith('.safetensors'))
    inventory_source = 'all manifest safetensors files'
    if index is not None:
        mapping = index.get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise PreflightError('The checkpoint index is missing its weight map.')
        shard_names = sorted({safe_filename(name) for name in mapping.values()})
        if any(not name.endswith('.safetensors') for name in shard_names):
            raise PreflightError('The safetensors index names an unexpected weight format.')
        if any(name not in files for name in shard_names):
            raise PreflightError('The index names a shard absent from the manifest.')
        inventory_source = 'pinned safetensors index weight map'
    if not shard_names:
        warnings.append('No safetensors weight shards were identified; checkpoint bytes remain unknown.')
    missing_sizes = [name for name in shard_names if _file_size(files[name]) is None]
    checkpoint_bytes = (sum(_file_size(files[name]) for name in shard_names)
                        if shard_names and not missing_sizes else None)
    if missing_sizes:
        warnings.append('At least one shard size is missing; refusing to present a complete download budget.')
    if index is None and len(shard_names) > 1:
        warnings.append('No index was supplied; the manifest inventory may include alternative checkpoints.')
    safetensors = manifest.get('safetensors') or {}
    counts = safetensors.get('parameters') if isinstance(safetensors, dict) else None
    parameter_count = None
    if (isinstance(counts, dict) and counts and all(isinstance(count, int) and
            not isinstance(count, bool) and count >= 0 for count in counts.values())):
        total = sum(counts.values())
        parameter_count = total if total > 0 else None
    if parameter_count is None:
        warnings.append('The manifest parameter-count histogram is unavailable; quantization floors remain unknown.')
    ram = _capacity_bytes(ram_gib)
    vram = _capacity_bytes(vram_gib)
    if free_disk_bytes is not None and (not isinstance(free_disk_bytes, int) or
            isinstance(free_disk_bytes, bool) or free_disk_bytes < 0):
        raise PreflightError('Free disk bytes must be a nonnegative integer.')
    architecture = architecture if architecture is not None else installed_architecture(config.get('model_type'))
    if architecture.get('config_registry_support') is False:
        warnings.append('The installed Transformers configuration registry does not support this model_type.')
    floors = {}
    for bits in (4, 2):
        floor = (parameter_count * bits + 7) // 8 if parameter_count is not None else None
        disk_floor = checkpoint_bytes + floor if checkpoint_bytes is not None and floor is not None else None
        combined = ram + vram if ram is not None and vram is not None else None
        fits = lambda amount, capacity: amount <= capacity if amount is not None and capacity is not None else None
        floors[str(bits) + 'bit'] = {
            'weight_only_floor_bytes': floor,
            'source_plus_output_floor_bytes': disk_floor,
            'source_plus_output_floor_fits_free_disk': fits(disk_floor, free_disk_bytes),
            'weight_only_floor_fits_ram': fits(floor, ram),
            'weight_only_floor_fits_vram': fits(floor, vram),
            'weight_only_floor_fits_combined_ram_vram': fits(floor, combined),
        }
    return {
        'repository': OFFICIAL_REPO, 'revision': revision,
        'model_type': config.get('model_type'), 'architecture': architecture,
        'metadata_ready_for_budgeting': (checkpoint_bytes is not None and parameter_count is not None
                                        and (index is not None or len(shard_names) == 1)),
        'checkpoint': {'inventory_source': inventory_source, 'shard_count': len(shard_names),
                       'total_bytes': checkpoint_bytes, 'missing_size_files': missing_sizes,
                       'manifest_parameter_count': parameter_count, 'dtype_parameter_counts': counts},
        'hardware': {'ram_bytes': ram, 'vram_bytes': vram, 'free_disk_bytes': free_disk_bytes},
        'ideal_quantization_floors': floors,
        'temporary_working_space_bytes': None,
        'downloaded_weight_shards': 0,
        'compression_export_pipeline_implemented': False,
        'compression_pipeline_scope': 'full_checkpoint_export; separate matrix-group prototypes are not a runnable export',
        'compression_or_inference_feasibility_established': False,
        'limitations': [
            'Floors assume every declared parameter occupies exactly 4 or 2 bits; they exclude scales, metadata, mixed precision, buffers, and runtime overhead.',
            'Source plus output is only a disk floor; actual export requires additional unknown working space.',
            'MoE active parameter counts do not reduce complete checkpoint storage.',
            'Configuration dispatch does not prove compatible attention kernels, steering hooks, quantization, or inference.',
            'This metadata tool does not compress model weights; separate matrix-group prototypes do not establish full checkpoint export or inference.',
        ],
        'warnings': warnings,
    }


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def _local_metadata(path, *, limit=MAX_FILE_BYTES):
    path = Path(path)
    if path.stat().st_size > limit:
        raise PreflightError('Saved metadata exceeds its byte budget.')
    with path.open('rb') as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise PreflightError('Saved metadata exceeds its byte budget.')
    return data


def offline_plan(output, *, revision=None, **hardware):
    output = Path(output)
    raw_manifest = _local_metadata(output / 'manifest.json', limit=MAX_MANIFEST_BYTES)
    manifest = read_json_bytes(raw_manifest, 'Saved manifest')
    verify_manifest(manifest, revision)
    raw_config = _local_metadata(output / 'config.json')
    config = read_json_bytes(raw_config, 'Saved configuration')
    path = output / 'model.safetensors.index.json'
    raw_index = _local_metadata(path) if path.is_file() else b''
    if len(raw_manifest) + len(raw_config) + len(raw_index) > MAX_TOTAL_BYTES:
        raise PreflightError('Saved metadata exceeds the total byte budget.')
    index = read_json_bytes(raw_index, 'Saved index') if raw_index else None
    return make_plan(manifest, config, index=index, **hardware)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', help='Optional full immutable commit SHA; branches/tags are rejected.')
    parser.add_argument('--offline', action='store_true', help='Use already saved metadata; perform no network requests.')
    parser.add_argument('--ram-gib', type=float)
    parser.add_argument('--vram-gib', type=float)
    parser.add_argument('--free-disk-bytes', type=int)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    workspace = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(workspace) or output == workspace or any(
            part in ('.git', '.venv', '.codex', '.agents') for part in output.relative_to(workspace).parts):
        parser.error('Output must be an ordinary subdirectory of this repository workspace.')
    try:
        if not args.offline:
            fetch_metadata(output, revision=args.revision)
        plan = offline_plan(output, revision=args.revision, ram_gib=args.ram_gib,
                            vram_gib=args.vram_gib, free_disk_bytes=args.free_disk_bytes)
        atomic_write(output / 'plan.json', _json(plan))
    except (OSError, PreflightError) as exc:
        # Avoid printing exception URLs: redirect query strings can contain signed credentials.
        print(f'Preflight stopped: {type(exc).__name__}. No model weights were requested.')
        if isinstance(exc, PreflightError):
            print(str(exc))
        return 1
    print(f"Pinned {OFFICIAL_REPO} at {plan['revision']}.")
    print(f"Weight shards requested: 0. Checkpoint shards listed: {plan['checkpoint']['shard_count']}.")
    print(f"Exact checkpoint bytes: {plan['checkpoint']['total_bytes']}; parameter count: {plan['checkpoint']['manifest_parameter_count']}.")
    print(f"Resource plan: {output / 'plan.json'}")
    print('Compression/export and local inference feasibility remain unestablished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
