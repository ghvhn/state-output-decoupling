"""Pinned, bounded GLM-5.3 FP8 expert sampling and CPU chunk decoding.

Only an explicitly selected layer 3 gate, up, or down projection for routed
experts 0..7 is sampled. Samples contain original tensor bytes and their
128x128 FP32 scales, not a runnable checkpoint. Range requests must return
exact HTTP 206 responses. Down projections consume the intermediate product
SiLU(gate(hidden)) * up(hidden), rather than the original hidden states.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import torch

from tda.expert_compress import ExpertGroupIdentity


MODEL_ID = 'zai-org/GLM-5.3'
PINNED_REVISION = 'aca966e4e02791568aa6a4ced368624b3d897f42'
LAYER = 3
PROJECTION = 'gate_proj'
PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')
EXPERT_IDS = tuple(str(value) for value in range(8))
BLOCK_SIZE = (128, 128)
MAX_HEADER_BYTES = 4 * 1024 ** 2
MAX_SAMPLE_BYTES = 128 * 1024 ** 2
FORMAT = 'glm-fp8-eight-expert-range-sample-v1'
LAYOUT = 'separate gate projection; row-major; FP8 times 128x128 FP32 block scales'
PROJECTION_LAYOUTS = {
    value: f'separate {value.removesuffix("_proj")} projection; row-major; FP8 times 128x128 FP32 block scales'
    for value in PROJECTIONS
}
HOSTS = frozenset({'huggingface.co', 'cdn-lfs.huggingface.co',
                  'cdn-lfs.hf.co', 'cdn-lfs-us-1.hf.co', 'cdn-lfs-eu-1.hf.co',
                  'cas-bridge.xethub.hf.co', 'us.aws.cdn.hf.co'})
DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
               'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
               'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}


class SampleError(ValueError):
    pass


def projection_layout(value):
    if not isinstance(value, str) or value not in PROJECTIONS:
        raise SampleError('Projection must be gate_proj, up_proj, or down_proj.')
    return PROJECTION_LAYOUTS[value]


def integer(value, description, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SampleError(f'{description} must be an integer >= {minimum}.')
    return value


def revision(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value):
        raise SampleError('An immutable full commit SHA is required.')
    return value


def shard_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'model-[0-9]+-of-[0-9]+\.safetensors', value):
        raise SampleError('An indexed shard must have a plain model safetensors filename.')
    return value


def public_https(url):
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise SampleError('Invalid storage URL port.') from exc
    if (parsed.scheme != 'https' or parsed.hostname not in HOSTS or
            parsed.username or parsed.password or port not in (None, 443)):
        raise SampleError('Weight ranges may only use known official HTTPS storage hosts.')
    return url


class StorageRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_https(newurl)
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and result.get_header('Range') != req.get_header('Range'):
            raise SampleError('A redirect discarded the bounded Range request.')
        return result


class RangeClient:
    """No credentials; no retries or fallback from partial to whole-shard reads."""
    def __init__(self, byte_cap, *, opener=None):
        self.byte_cap = integer(byte_cap, 'Byte cap', 1)
        if self.byte_cap > MAX_SAMPLE_BYTES:
            raise SampleError('The sample byte cap exceeds the fixed 128 MiB safety limit.')
        self.received_bytes = 0
        self.opener = opener or build_opener(StorageRedirects())
        self.requests = 0

    def get(self, url, start, stop, total_bytes):
        public_https(url)
        integer(start, 'Range start')
        integer(stop, 'Range stop', 1)
        integer(total_bytes, 'Shard length', 1)
        if not start < stop <= total_bytes:
            raise SampleError('The requested range is outside the indexed shard.')
        length = stop - start
        if self.received_bytes + length > self.byte_cap:
            raise SampleError('The requested range exceeds the total sample byte cap.')
        request = Request(url, headers={'Range': f'bytes={start}-{stop - 1}',
                                      'Accept-Encoding': 'identity',
                                      'User-Agent': 'tda-domain-mapper-bounded-weight-sample/1'})
        self.requests += 1
        with self.opener.open(request, timeout=60) as response:
            public_https(response.geturl())
            if response.getcode() != 206:
                raise SampleError('The server did not return HTTP 206; whole-shard reads are refused.')
            expected = f'bytes {start}-{stop - 1}/{total_bytes}'
            if response.headers.get('Content-Range') != expected:
                raise SampleError('Content-Range does not exactly match the requested bytes and shard length.')
            try:
                declared = int(response.headers.get('Content-Length', ''))
            except (TypeError, ValueError) as exc:
                raise SampleError('The partial response must declare its exact Content-Length.') from exc
            if declared != length or response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise SampleError('The partial response length or encoding is invalid.')
            chunks = []
            received = 0
            while received < length:
                data = response.read(min(1024 ** 2, length - received))
                if not data:
                    raise SampleError('The partial response ended before the requested range was complete.')
                chunks.append(data)
                received += len(data)
                self.received_bytes += len(data)
            if response.read(1):
                self.received_bytes += 1
                raise SampleError('The partial response exceeded its declared range.')
            return b''.join(chunks)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SampleError('Duplicate JSON keys are not permitted.')
        result[key] = value
    return result


def json_object(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise SampleError('Invalid JSON metadata.') from exc
    if not isinstance(value, dict):
        raise SampleError('Metadata must contain a JSON object.')
    return value


def read_bounded(path, limit):
    path = Path(path)
    if path.stat().st_size > limit:
        raise SampleError('A saved metadata file exceeds its byte bound.')
    with path.open('rb') as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise SampleError('A saved metadata file exceeds its byte bound.')
    return data


def parse_header(raw, shard_bytes):
    if not 0 < len(raw) <= MAX_HEADER_BYTES:
        raise SampleError('The safetensors header exceeds its permitted bound.')
    integer(shard_bytes, 'Shard bytes', 9)
    data_bytes = shard_bytes - 8 - len(raw)
    if data_bytes < 0:
        raise SampleError('The header is larger than the indexed shard.')
    header = json_object(raw)
    intervals = []
    for name, entry in header.items():
        if name == '__metadata__':
            if not isinstance(entry, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                  for k, v in entry.items()):
                raise SampleError('Invalid safetensors metadata.')
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise SampleError('Invalid safetensors tensor descriptor.')
        dtype = entry.get('dtype')
        shape = entry.get('shape')
        offsets = entry.get('data_offsets')
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or len(shape) > 8:
            raise SampleError('Unsupported safetensors dtype or shape.')
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise SampleError('Each tensor requires a start and end offset.')
        dimensions = [integer(dim, 'Tensor dimension') for dim in shape]
        start, stop = [integer(offset, 'Tensor offset') for offset in offsets]
        if stop < start or stop > data_bytes or stop - start != math.prod(dimensions) * DTYPE_BYTES[dtype]:
            raise SampleError('Tensor offsets do not match its shape, dtype, and indexed shard span.')
        intervals.append((start, stop))
    cursor = 0
    for start, stop in sorted(intervals):
        if start != cursor:
            raise SampleError('Safetensors payload ranges overlap or contain unindexed gaps.')
        cursor = stop
    if cursor != data_bytes:
        raise SampleError('The header does not index the complete safetensors payload.')
    return header


def _atomic(path, data):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while data := handle.read(1024 ** 2):
            digest.update(data)
    return digest.hexdigest()


def _serialize(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode('utf-8')


def _metadata(metadata_dir, expected_revision, projection=PROJECTION):
    projection_layout(projection)
    folder = Path(metadata_dir)
    raw = {name: read_bounded(folder / name, 64 * 1024 ** 2) for name in
           ('manifest.json', 'config.json', 'model.safetensors.index.json')}
    manifest, config, index = [json_object(raw[name]) for name in raw]
    if manifest.get('id') != MODEL_ID or revision(manifest.get('sha')) != revision(expected_revision):
        raise SampleError('Saved metadata must match the official repository and requested immutable revision.')
    quantization = config.get('quantization_config') or {}
    if (quantization.get('quant_method') != 'fp8' or quantization.get('fmt') != 'e4m3' or
            quantization.get('weight_block_size') != list(BLOCK_SIZE)):
        raise SampleError('The sample requires the declared E4M3 FP8 128x128 weight format.')
    shape = (integer(config.get('moe_intermediate_size'), 'Expert intermediate width', 1),
             integer(config.get('hidden_size'), 'Hidden width', 1))
    if projection == 'down_proj':
        shape = shape[::-1]
    files = {}
    for row in manifest.get('siblings', []):
        if not isinstance(row, dict) or not isinstance(row.get('rfilename'), str):
            raise SampleError('Invalid manifest file entry.')
        if row['rfilename'] in files:
            raise SampleError('Duplicate manifest filenames.')
        files[row['rfilename']] = row
    mapping = index.get('weight_map')
    if not isinstance(mapping, dict):
        raise SampleError('The checkpoint index is missing its weight map.')
    selected = []
    for expert in EXPERT_IDS:
        base = f'model.layers.{LAYER}.mlp.experts.{expert}.{projection}.'
        for kind, key in (('weight', base + 'weight'), ('scale', base + 'weight_scale_inv')):
            shard = shard_name(mapping.get(key))
            row = files.get(shard)
            if row is None:
                raise SampleError('An indexed sample shard is absent from the pinned manifest.')
            size = row.get('size')
            if size is None and isinstance(row.get('lfs'), dict):
                size = row['lfs'].get('size')
            selected.append({'expert_id': expert, 'kind': kind, 'key': key, 'shard': shard,
                             'shard_bytes': integer(size, 'Indexed shard bytes', 9)})
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    return selected, shape, hashes


def download_sample(metadata_dir, output, *, byte_cap, expected_revision=PINNED_REVISION,
                    projection=PROJECTION, client=None):
    layout = projection_layout(projection)
    selected, shape, metadata_hashes = _metadata(metadata_dir, expected_revision, projection)
    expected_revision = revision(expected_revision)
    byte_cap = integer(byte_cap, 'Byte cap', 1)
    if byte_cap > MAX_SAMPLE_BYTES:
        raise SampleError('The sample byte cap exceeds the fixed 128 MiB limit.')
    scale_shape = tuple((dim + block - 1) // block for dim, block in zip(shape, BLOCK_SIZE))
    payload_bytes = len(EXPERT_IDS) * (math.prod(shape) + 4 * math.prod(scale_shape))
    if payload_bytes + 9 > byte_cap:
        raise SampleError('The exact tensor payloads alone exceed the explicit byte cap.')
    output = Path(output)
    if output.exists():
        raise SampleError('The sample output already exists; existing files will not be replaced.')
    client = client or RangeClient(byte_cap)
    if getattr(client, 'byte_cap', byte_cap) > byte_cap:
        raise SampleError('The range client budget exceeds the explicit operation cap.')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
    headers = {}
    tensors = []
    try:
        # Verify all selected source headers before spending the payload budget.
        for entry in selected:
            shard = entry['shard']
            if shard not in headers:
                url = f'https://huggingface.co/{MODEL_ID}/resolve/{expected_revision}/{shard}'
                prefix = client.get(url, 0, 8, entry['shard_bytes'])
                if len(prefix) != 8:
                    raise SampleError('Invalid safetensors length prefix.')
                length = struct.unpack('<Q', prefix)[0]
                if not 0 < length <= MAX_HEADER_BYTES or length + 8 >= entry['shard_bytes']:
                    raise SampleError('The declared safetensors header length is invalid or exceeds its cap.')
                raw = client.get(url, 8, 8 + length, entry['shard_bytes'])
                if len(raw) != length:
                    raise SampleError('Invalid safetensors header response length.')
                parsed = parse_header(raw, entry['shard_bytes'])
                header_file = shard + '.header.json'
                _atomic(staging / header_file, raw)
                headers[shard] = {'descriptor': parsed, 'data_start': 8 + length,
                                  'shard_bytes': entry['shard_bytes'], 'header_file': header_file,
                                  'header_sha256': hashlib.sha256(raw).hexdigest()}
        pending = []
        for entry in selected:
            shard = entry['shard']
            header = headers[shard]
            tensor = header['descriptor'].get(entry['key'])
            wanted_shape = shape if entry['kind'] == 'weight' else scale_shape
            wanted_dtype = 'F8_E4M3' if entry['kind'] == 'weight' else 'F32'
            if not isinstance(tensor, dict) or tensor.get('shape') != list(wanted_shape) or tensor.get('dtype') != wanted_dtype:
                raise SampleError('The selected expert weight or paired scale has an unexpected shape or dtype.')
            relative_start, relative_stop = tensor['data_offsets']
            start, stop = relative_start + header['data_start'], relative_stop + header['data_start']
            pending.append((entry, wanted_shape, wanted_dtype, start, stop))
        if client.received_bytes + payload_bytes > byte_cap:
            raise SampleError('The verified headers plus all paired tensor payloads exceed the explicit byte cap.')
        for entry, wanted_shape, wanted_dtype, start, stop in pending:
            shard = entry['shard']
            url = f'https://huggingface.co/{MODEL_ID}/resolve/{expected_revision}/{shard}'
            raw = client.get(url, start, stop, entry['shard_bytes'])
            if len(raw) != stop - start:
                raise SampleError('The selected tensor response length is incorrect.')
            filename = f"expert_{entry['expert_id']}.{entry['kind']}.bin"
            _atomic(staging / filename, raw)
            tensors.append({**entry, 'shape': list(wanted_shape), 'dtype': wanted_dtype,
                            'source_range': [start, stop], 'filename': filename,
                            'payload_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        identity = ExpertGroupIdentity(
            model_id=MODEL_ID, revision=expected_revision, layer=LAYER, projection=projection,
            layout=layout,
            expert_ids=EXPERT_IDS, matrix_shape=shape, source_dtype='float8_e4m3fn',
            source_storage_bytes=payload_bytes)
        receipt = {'format': FORMAT, 'identity': identity.to_dict(), 'block_size': list(BLOCK_SIZE),
                   'byte_cap': byte_cap, 'network_bytes': client.received_bytes,
                   'source_tensor_bytes_including_scales': payload_bytes,
                   'metadata_sha256': metadata_hashes, 'tensors': tensors,
                   'headers': {name: {k: v for k, v in header.items() if k != 'descriptor'}
                               for name, header in headers.items()},
                   'limitations': ['Original selected tensor bytes only; not a runnable checkpoint.',
                                   'Partial payload hashes detect local corruption; they are not whole-shard LFS hash verification.',
                                   'No model forward passes, GPU operations, or runtime quality validation.']}
        if projection == 'down_proj':
            receipt['limitations'].append(
                'Down projection inputs are SiLU(gate(hidden)) * up(hidden), not pre-expert hidden states.')
        raw_manifest = _serialize(receipt)
        _atomic(staging / 'sample_manifest.json', raw_manifest)
        _atomic(staging / 'sample_manifest.sha256', hashlib.sha256(raw_manifest).hexdigest().encode('ascii'))
        # Read back every stored byte and contract before publishing the complete directory.
        with GLMWeightSampleReader(staging):
            pass
        os.replace(staging, output)
        return receipt
    finally:
        if staging.exists():
            target = staging.resolve()
            if target.parent != output.parent.resolve() or not target.name.startswith('.' + output.name + '.'):
                raise SampleError('Refusing to remove a staging path outside the intended output parent.')
            shutil.rmtree(target)


def _e4m3_values():
    values = np.arange(256, dtype=np.uint16)
    exponents = ((values >> 3) & 15).astype(np.int16)
    mantissas = (values & 7).astype(np.float32)
    decoded = np.where(exponents == 0, mantissas * (2.0 ** -9),
                       (1 + mantissas / 8) * np.exp2(exponents - 7)).astype(np.float32)
    decoded[(exponents == 15) & (mantissas == 7)] = np.nan
    return np.copysign(decoded, np.where(values & 128, -1.0, 1.0)).astype(np.float32)


FP8_VALUES = _e4m3_values()


class GLMWeightSampleReader:
    """CPU memmaps and bounded FP32 slices matching ExpertMatrixReader."""
    def __init__(self, folder, *, max_chunk_columns=262144):
        self.folder = Path(folder)
        self.max_chunk_columns = integer(max_chunk_columns, 'Maximum chunk columns', 1)
        if self.max_chunk_columns > 1048576:
            raise SampleError('Reader chunks may not exceed 1048576 columns.')
        raw = read_bounded(self.folder / 'sample_manifest.json', 4 * 1024 ** 2)
        expected_hash = read_bounded(self.folder / 'sample_manifest.sha256', 64).decode('ascii')
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise SampleError('The sample manifest hash does not match.')
        self.manifest = json_object(raw)
        if self.manifest.get('format') != FORMAT or self.manifest.get('block_size') != list(BLOCK_SIZE):
            raise SampleError('Unsupported sample format or FP8 block layout.')
        self.identity = ExpertGroupIdentity.from_dict(self.manifest.get('identity'))
        layout = projection_layout(self.identity.projection)
        if (self.identity.model_id != MODEL_ID or self.identity.layer != LAYER or
                self.identity.expert_ids != EXPERT_IDS or
                self.identity.source_dtype != 'float8_e4m3fn' or self.identity.layout != layout):
            raise SampleError('The sample does not describe the selected official expert group.')
        revision(self.identity.revision)
        self.expert_ids = self.identity.expert_ids
        self.shape = self.identity.matrix_shape
        self.n_columns = self.identity.column_count
        self.source_bytes = self.identity.actual_source_bytes
        if self.source_bytes > MAX_SAMPLE_BYTES:
            raise SampleError('Saved source tensors exceed the bounded sample size.')
        scale_shape = tuple((dim + block - 1) // block for dim, block in zip(self.shape, BLOCK_SIZE))
        header_descriptors = {}
        for name, record in self.manifest.get('headers', {}).items():
            shard_name(name)
            if record.get('header_file') != name + '.header.json':
                raise SampleError('Unsafe or mismatched header filename.')
            raw_header = read_bounded(self.folder / record['header_file'], MAX_HEADER_BYTES)
            if hashlib.sha256(raw_header).hexdigest() != record.get('header_sha256'):
                raise SampleError('A saved source header hash does not match.')
            if record.get('data_start') != 8 + len(raw_header):
                raise SampleError('The saved source header length does not match.')
            header_descriptors[name] = parse_header(raw_header, record.get('shard_bytes'))
        records = {}
        total = 0
        for tensor in self.manifest.get('tensors', []):
            expert, kind = tensor.get('expert_id'), tensor.get('kind')
            if expert not in EXPERT_IDS or kind not in ('weight', 'scale') or (expert, kind) in records:
                raise SampleError('Unexpected or duplicated sample tensor.')
            filename = f'expert_{expert}.{kind}.bin'
            shape = self.shape if kind == 'weight' else scale_shape
            dtype = 'F8_E4M3' if kind == 'weight' else 'F32'
            byte_count = math.prod(shape) * (1 if kind == 'weight' else 4)
            base = f'model.layers.{LAYER}.mlp.experts.{expert}.{self.identity.projection}.'
            key = base + ('weight' if kind == 'weight' else 'weight_scale_inv')
            shard = tensor.get('shard')
            header = self.manifest.get('headers', {}).get(shard)
            descriptor = header_descriptors.get(shard, {}).get(key)
            if (tensor.get('filename') != filename or tensor.get('key') != key or
                    tensor.get('shape') != list(shape) or tensor.get('dtype') != dtype or
                    tensor.get('payload_bytes') != byte_count or not descriptor or
                    descriptor.get('shape') != list(shape) or descriptor.get('dtype') != dtype):
                raise SampleError('Stored tensor identity, shape, dtype, or paired source header differs.')
            source_range = [offset + header['data_start'] for offset in descriptor['data_offsets']]
            if tensor.get('source_range') != source_range or tensor.get('shard_bytes') != header['shard_bytes']:
                raise SampleError('Stored tensor source range differs from its pinned source header.')
            path = self.folder / filename
            if path.stat().st_size != byte_count or _sha_file(path) != tensor.get('sha256'):
                raise SampleError('Stored tensor length or hash does not match.')
            records[(expert, kind)] = path
            total += byte_count
        if len(records) != 2 * len(EXPERT_IDS) or total != self.source_bytes or total != self.manifest.get('source_tensor_bytes_including_scales'):
            raise SampleError('Every selected weight requires its paired scale and exact original-byte baseline.')
        self._scales = {expert: np.fromfile(records[(expert, 'scale')], dtype='<f4').reshape(scale_shape)
                        for expert in EXPERT_IDS}
        if any(not np.isfinite(scale).all() or not (scale > 0).all() for scale in self._scales.values()):
            raise SampleError('FP8 block scales must be finite and positive.')
        self._weights = {expert: np.memmap(records[(expert, 'weight')], dtype=np.uint8, mode='r',
                                         shape=(self.n_columns,)) for expert in EXPERT_IDS}

    def close(self):
        for weight in self._weights.values():
            weight._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_columns(self, start, stop, expert_ids=None):
        integer(start, 'Column start')
        integer(stop, 'Column stop', 1)
        if not start < stop <= self.n_columns or stop - start > self.max_chunk_columns:
            raise SampleError('The requested columns are outside the matrix or exceed the chunk bound.')
        ids = self.expert_ids if expert_ids is None else tuple(expert_ids)
        if not ids or len(set(ids)) != len(ids) or any(expert not in self.expert_ids for expert in ids):
            raise SampleError('Requested expert IDs must be a unique subset of the stored group.')
        positions = np.arange(start, stop, dtype=np.int64)
        row_blocks = (positions // self.shape[1]) // BLOCK_SIZE[0]
        col_blocks = (positions % self.shape[1]) // BLOCK_SIZE[1]
        result = np.empty((len(ids), stop - start), dtype=np.float32)
        for index, expert in enumerate(ids):
            result[index] = FP8_VALUES[self._weights[expert][start:stop]] * self._scales[expert][row_blocks, col_blocks]
        if not np.isfinite(result).all():
            raise SampleError('The decoded sample chunk contains nonfinite values.')
        return torch.from_numpy(result)


GLMExpertSample = GLMWeightSampleReader
