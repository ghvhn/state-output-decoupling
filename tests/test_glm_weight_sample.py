from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
from urllib.request import Request

import numpy as np
import pytest
import torch

from tda import glm_weight_sample as sample


class Response(io.BytesIO):
    def __init__(self, data, *, status=206, content_range='bytes 2-5/10',
                 content_length='4', url='https://cas-bridge.xethub.hf.co/storage', encoding=None):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': content_range, 'Content-Length': content_length}
        if encoding is not None:
            self.headers['Content-Encoding'] = encoding
        self.url = url
        self.bytes_read = 0

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url

    def read(self, *args):
        data = super().read(*args)
        self.bytes_read += len(data)
        return data


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return next(self.responses)


def url():
    return f'https://huggingface.co/{sample.MODEL_ID}/resolve/{sample.PINNED_REVISION}/model-00001-of-00003.safetensors'


def fixture_metadata(tmp_path, shape=(130, 260), *, bad_scale=False, projection=sample.PROJECTION):
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    scale_shape = tuple((dim + 127) // 128 for dim in shape)
    shards = {}
    mapping = {}
    raw_weights = {}
    scales = {}
    for shard_number, ids in enumerate((('0', '1'), ('2',), ('3', '4', '5', '6', '7')), 1):
        shard = f'model-{shard_number:05d}-of-00003.safetensors'
        header = {}
        payloads = []
        cursor = 0
        for expert in ids:
            values = (np.arange(np.prod(shape), dtype=np.uint64) % 120).astype(np.uint8)
            if int(expert) % 2:
                values |= 128
            block_scales = (np.arange(np.prod(scale_shape), dtype=np.float32).reshape(scale_shape) + 0.25) * (int(expert) + 1)
            if bad_scale and expert == '7':
                block_scales[-1, -1] = np.nan
            raw_weights[expert] = values
            scales[expert] = block_scales
            base = f'model.layers.3.mlp.experts.{expert}.{projection}.'
            for key, dtype, dimensions, raw in (
                    (base + 'weight', 'F8_E4M3', shape, values.tobytes()),
                    (base + 'weight_scale_inv', 'F32', scale_shape, block_scales.astype('<f4').tobytes())):
                header[key] = {'dtype': dtype, 'shape': list(dimensions), 'data_offsets': [cursor, cursor + len(raw)]}
                mapping[key] = shard
                cursor += len(raw)
                payloads.append(raw)
        raw_header = json.dumps(header, separators=(',', ':')).encode()
        shards[shard] = struct.pack('<Q', len(raw_header)) + raw_header + b''.join(payloads)
    manifest = {'id': sample.MODEL_ID, 'sha': sample.PINNED_REVISION,
                'siblings': [{'rfilename': name, 'size': len(raw)} for name, raw in shards.items()]}
    hidden, intermediate = (shape[0], shape[1]) if projection == 'down_proj' else (shape[1], shape[0])
    config = {'hidden_size': hidden, 'moe_intermediate_size': intermediate,
              'quantization_config': {'quant_method': 'fp8', 'fmt': 'e4m3', 'weight_block_size': [128, 128]}}
    index = {'weight_map': mapping}
    for name, value in (('manifest.json', manifest), ('config.json', config),
                        ('model.safetensors.index.json', index)):
        (metadata / name).write_text(json.dumps(value))
    return metadata, shards, raw_weights, scales


class FakeRanges:
    def __init__(self, shards, *, byte_cap=sample.MAX_SAMPLE_BYTES, fail_call=None):
        self.shards = shards
        self.byte_cap = byte_cap
        self.received_bytes = 0
        self.requests = []
        self.fail_call = fail_call

    def get(self, url, start, stop, total_bytes):
        self.requests.append((url, start, stop, total_bytes))
        if self.fail_call == len(self.requests):
            raise sample.SampleError('simulated partial failure')
        raw = self.shards[url.rsplit('/', 1)[1]]
        assert total_bytes == len(raw)
        assert self.received_bytes + stop - start <= self.byte_cap
        self.received_bytes += stop - start
        return raw[start:stop]


def downloaded(tmp_path, *, shape=(130, 260), bad_scale=False, projection=sample.PROJECTION):
    metadata, shards, weights, scales = fixture_metadata(tmp_path, shape, bad_scale=bad_scale,
                                                        projection=projection)
    output = tmp_path / 'sample'
    client = FakeRanges(shards)
    receipt = sample.download_sample(metadata, output, byte_cap=sample.MAX_SAMPLE_BYTES,
                                     projection=projection, client=client)
    return output, receipt, client, weights, scales


def test_http_range_requires_exact_206_and_headers_without_auth():
    response = Response(b'abcd')
    opener = Opener([response])
    client = sample.RangeClient(4, opener=opener)
    assert client.get(url(), 2, 6, 10) == b'abcd'
    assert client.received_bytes == 4
    assert opener.requests[0].get_header('Range') == 'bytes=2-5'
    assert opener.requests[0].get_header('Accept-encoding') == 'identity'
    assert not opener.requests[0].has_header('Authorization')


@pytest.mark.parametrize('kwargs', [
    {'status': 200}, {'content_range': 'bytes 0-9/10'}, {'content_range': 'bytes 2-5/11'},
    {'content_range': 'bytes 2-5/*'}, {'content_length': '5'}, {'content_length': None},
    {'encoding': 'gzip'}, {'url': 'https://untrusted.test/tensor'},
])
def test_unbounded_or_mismatched_response_is_rejected_before_body_read(kwargs):
    response = Response(b'abcd', **kwargs)
    client = sample.RangeClient(10, opener=Opener([response]))
    with pytest.raises(sample.SampleError):
        client.get(url(), 2, 6, 10)
    assert response.bytes_read == 0


@pytest.mark.parametrize('data', [b'abc', b'abcde'])
def test_short_and_overlong_partial_body_are_rejected(data):
    client = sample.RangeClient(10, opener=Opener([Response(data)]))
    with pytest.raises(sample.SampleError):
        client.get(url(), 2, 6, 10)


def test_byte_cap_is_cumulative_and_never_falls_back_to_whole_shard():
    opener = Opener([Response(b'abcd')])
    client = sample.RangeClient(6, opener=opener)
    client.get(url(), 2, 6, 10)
    with pytest.raises(sample.SampleError, match='cap'):
        client.get(url(), 2, 6, 10)
    assert len(opener.requests) == 1
    with pytest.raises(sample.SampleError, match='128 MiB'):
        sample.RangeClient(sample.MAX_SAMPLE_BYTES + 1)


@pytest.mark.parametrize('target', ['http://huggingface.co/a', 'https://evil.hf.co/a',
                                   'https://token@huggingface.co/a', 'https://huggingface.co:80/a',
                                   'https://evil.us.aws.cdn.hf.co/a', 'https://eu.aws.cdn.hf.co/a'])
def test_storage_redirects_reject_unknown_host_http_or_credentials(target):
    with pytest.raises(sample.SampleError):
        sample.StorageRedirects().redirect_request(Request(url(), headers={'Range': 'bytes=2-5'}),
                                                    None, 302, '', {}, target)


def test_official_redirect_preserves_range_header():
    original = Request(url(), headers={'Range': 'bytes=2-5'})
    redirected = sample.StorageRedirects().redirect_request(original, None, 302, '', {},
                                                           'https://cas-bridge.xethub.hf.co/storage')
    assert redirected.get_header('Range') == 'bytes=2-5'


def test_observed_official_us_aws_cdn_redirect_is_allowed_exactly():
    original = Request(url(), headers={'Range': 'bytes=0-7'})
    redirected = sample.StorageRedirects().redirect_request(original, None, 302, '', {},
                                                           'https://us.aws.cdn.hf.co/bridge/storage?signature=public')
    assert redirected.get_header('Range') == 'bytes=0-7'
    assert sample.public_https(redirected.full_url) == redirected.full_url


def header(shape=(2, 3), dtype='F8_E4M3', offsets=(0, 6)):
    return json.dumps({'w': {'dtype': dtype, 'shape': list(shape), 'data_offsets': list(offsets)}}).encode()


def test_header_shape_dtype_and_complete_payload_are_checked():
    raw = header()
    assert sample.parse_header(raw, 8 + len(raw) + 6)['w']['shape'] == [2, 3]
    for altered in (header(dtype='unknown'), header(offsets=(0, 5)), header(offsets=(1, 7)),
                    header(shape=(True, 3)), header(offsets=(0, 7))):
        with pytest.raises(sample.SampleError):
            sample.parse_header(altered, 8 + len(altered) + 6)


def test_overlapping_gapped_or_duplicate_tensor_header_is_rejected():
    for offsets in ([0, 2], [4, 6]):
        raw = json.dumps({'a': {'dtype': 'U8', 'shape': [3], 'data_offsets': [0, 3]},
                          'b': {'dtype': 'U8', 'shape': [2], 'data_offsets': offsets}}).encode()
        with pytest.raises(sample.SampleError):
            sample.parse_header(raw, 8 + len(raw) + 6)
    duplicate = b'{"w":{},"w":{}}'
    with pytest.raises(sample.SampleError):
        sample.parse_header(duplicate, 8 + len(duplicate) + 1)


@pytest.mark.parametrize('name', ['../model.safetensors', '/model.safetensors',
                                 'weights/model-00001-of-00003.safetensors', 'model.py'])
def test_indexed_shard_paths_are_plain_safetensors_filenames(name):
    with pytest.raises(sample.SampleError):
        sample.shard_name(name)


def test_download_selects_only_eight_paired_tensors_and_pins_every_range(tmp_path):
    output, receipt, client, _, _ = downloaded(tmp_path)
    assert len(receipt['tensors']) == 16
    assert len(receipt['headers']) == 3
    assert len(client.requests) == 22  # three prefixes/headers plus sixteen tensors
    assert all('/resolve/' + sample.PINNED_REVISION + '/' in request[0] for request in client.requests)
    assert all(stop - start < total for _, start, stop, total in client.requests)
    assert receipt['network_bytes'] == sum(stop - start for _, start, stop, _ in client.requests)
    assert receipt['source_tensor_bytes_including_scales'] == 8 * (130 * 260 + 2 * 3 * 4)
    assert receipt['identity']['source_storage_bytes'] == receipt['source_tensor_bytes_including_scales']
    assert (output / 'sample_manifest.sha256').read_text() == hashlib.sha256(
        (output / 'sample_manifest.json').read_bytes()).hexdigest()
    assert not any(path.name.startswith('.sample.') for path in tmp_path.iterdir())


def test_default_gate_manifest_layout_and_format_remain_unchanged(tmp_path):
    metadata, shards, _, _ = fixture_metadata(tmp_path)
    output = tmp_path / 'sample'
    receipt = sample.download_sample(metadata, output, byte_cap=sample.MAX_SAMPLE_BYTES,
                                     client=FakeRanges(shards))
    assert receipt['format'] == sample.FORMAT == 'glm-fp8-eight-expert-range-sample-v1'
    assert receipt['identity']['projection'] == sample.PROJECTION == 'gate_proj'
    assert receipt['identity']['layout'] == sample.LAYOUT == sample.projection_layout('gate_proj')
    assert len(receipt['limitations']) == 3
    with sample.GLMWeightSampleReader(output) as reader:
        assert reader.identity.matrix_shape == (130, 260)


@pytest.mark.parametrize('projection,shape', [('up_proj', (130, 260)), ('down_proj', (260, 130))])
def test_explicit_projection_preserves_native_bytes_and_ordered_source_scope(tmp_path, projection, shape):
    output, receipt, client, weights, scales = downloaded(tmp_path, projection=projection, shape=shape)
    assert receipt['identity']['projection'] == projection
    assert receipt['identity']['matrix_shape'] == list(shape)
    assert receipt['identity']['layout'] == sample.projection_layout(projection)
    assert receipt['identity']['expert_ids'] == list(sample.EXPERT_IDS)
    assert receipt['identity']['revision'] == sample.PINNED_REVISION
    assert receipt['source_tensor_bytes_including_scales'] == 8 * (130 * 260 + 24)
    assert len(client.requests) == 22 and len(receipt['tensors']) == 16
    for tensor in receipt['tensors']:
        expert, kind = tensor['expert_id'], tensor['kind']
        suffix = 'weight' if kind == 'weight' else 'weight_scale_inv'
        assert tensor['key'] == f'model.layers.3.mlp.experts.{expert}.{projection}.{suffix}'
        expected = weights[expert].tobytes() if kind == 'weight' else scales[expert].astype('<f4').tobytes()
        assert (output / tensor['filename']).read_bytes() == expected
    with sample.GLMWeightSampleReader(output, max_chunk_columns=32) as reader:
        assert reader.shape == shape and reader.max_chunk_columns == 32
        assert reader.identity.source_dtype == 'float8_e4m3fn'
    if projection == 'down_proj':
        assert 'not pre-expert hidden states' in receipt['limitations'][-1]


@pytest.mark.parametrize('projection', ['gate_up_proj', '', None, ['up_proj']])
def test_unknown_projection_stops_before_any_network_request(tmp_path, projection):
    metadata, shards, _, _ = fixture_metadata(tmp_path)
    client = FakeRanges(shards)
    with pytest.raises(sample.SampleError, match='Projection must be'):
        sample.download_sample(metadata, tmp_path / 'sample', byte_cap=sample.MAX_SAMPLE_BYTES,
                               projection=projection, client=client)
    assert not client.requests and not (tmp_path / 'sample').exists()


@pytest.mark.parametrize('change', ['projection', 'layout', 'slots', 'matrix_shape', 'scale_shape'])
def test_reader_rejects_projection_or_dimension_contract_changes_even_with_new_manifest_hash(tmp_path, change):
    output, receipt, _, _, _ = downloaded(tmp_path)
    if change == 'projection':
        receipt['identity']['projection'] = 'up_proj'
        receipt['identity']['layout'] = sample.projection_layout('up_proj')
    elif change == 'layout':
        receipt['identity']['layout'] = sample.projection_layout('up_proj')
    elif change == 'slots':
        receipt['identity']['expert_ids'].reverse()
    elif change == 'matrix_shape':
        receipt['identity']['matrix_shape'].reverse()
    else:
        next(tensor for tensor in receipt['tensors'] if tensor['kind'] == 'scale')['shape'].reverse()
    raw = json.dumps(receipt).encode()
    (output / 'sample_manifest.json').write_bytes(raw)
    (output / 'sample_manifest.sha256').write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(sample.SampleError):
        sample.GLMWeightSampleReader(output)


@pytest.mark.parametrize('projection,shape', [('gate_proj', (130, 260)), ('up_proj', (130, 260)),
                                             ('down_proj', (260, 130))])
def test_decode_matches_torch_fp8_times_scales_across_rows_and_block_edges(tmp_path, projection, shape):
    output, _, _, weights, scales = downloaded(tmp_path, projection=projection, shape=shape)
    with sample.GLMWeightSampleReader(output) as reader:
        assert reader.identity.actual_source_bytes == 8 * (130 * 260 + 24)
        width = shape[1]
        for start, stop in ((120, 140), (width - 10, width + 20),
                            (127 * width + width - 10, 129 * width + 15)):
            observed = reader.read_columns(start, stop, expert_ids=('7', '0', '3'))
            positions = np.arange(start, stop)
            expected = []
            for expert in ('7', '0', '3'):
                raw = torch.from_numpy(weights[expert][start:stop].copy()).view(torch.float8_e4m3fn).float()
                multiplier = torch.from_numpy(scales[expert][positions // width // 128, positions % width // 128])
                expected.append(raw * multiplier)
            assert observed.dtype == torch.float32 and observed.device.type == 'cpu'
            torch.testing.assert_close(observed, torch.stack(expected), rtol=0, atol=0)


def test_e4m3_lookup_matches_every_cpu_torch_bit_pattern_including_signed_zero():
    raw = torch.arange(256, dtype=torch.uint8)
    expected = raw.view(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_array_equal(sample.FP8_VALUES, expected)
    np.testing.assert_array_equal(np.signbit(sample.FP8_VALUES), np.signbit(expected))


def test_reader_bounds_and_requested_expert_order(tmp_path):
    output, _, _, _, _ = downloaded(tmp_path)
    with sample.GLMWeightSampleReader(output, max_chunk_columns=32) as reader:
        for start, stop, ids in ((0, 33, None), (0, 0, None), (-1, 3, None),
                                 (0, 10, ('0', '0')), (0, 10, ('8',)), (0, 10, ())):
            with pytest.raises(sample.SampleError):
                reader.read_columns(start, stop, ids)
        full = reader.read_columns(0, 10)
        torch.testing.assert_close(reader.read_columns(0, 10, ('7', '2')), full[[7, 2]])


def test_partial_network_failure_never_publishes_sample_or_leaves_staging(tmp_path):
    metadata, shards, _, _ = fixture_metadata(tmp_path)
    output = tmp_path / 'sample'
    with pytest.raises(sample.SampleError, match='simulated'):
        sample.download_sample(metadata, output, byte_cap=sample.MAX_SAMPLE_BYTES,
                               client=FakeRanges(shards, fail_call=5))
    assert not output.exists()
    assert not any(path.name.startswith('.sample.') for path in tmp_path.iterdir())


def test_header_byte_bound_stops_before_header_body_request(tmp_path):
    metadata, shards, _, _ = fixture_metadata(tmp_path)
    first = next(iter(shards))
    shards[first] = struct.pack('<Q', sample.MAX_HEADER_BYTES + 1) + shards[first][8:]
    client = FakeRanges(shards)
    with pytest.raises(sample.SampleError, match='header length'):
        sample.download_sample(metadata, tmp_path / 'sample', byte_cap=sample.MAX_SAMPLE_BYTES, client=client)
    assert len(client.requests) == 1


def test_missing_scale_shape_mismatch_or_wrong_pin_stops_without_publish(tmp_path):
    metadata, shards, _, _ = fixture_metadata(tmp_path)
    index_path = metadata / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    del index['weight_map']['model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv']
    index_path.write_text(json.dumps(index))
    client = FakeRanges(shards)
    with pytest.raises(sample.SampleError):
        sample.download_sample(metadata, tmp_path / 'sample', byte_cap=sample.MAX_SAMPLE_BYTES, client=client)
    assert not client.requests
    with pytest.raises(sample.SampleError, match='immutable revision'):
        sample.download_sample(metadata, tmp_path / 'sample', byte_cap=sample.MAX_SAMPLE_BYTES,
                               expected_revision='b' * 40, client=client)


def test_too_small_explicit_cap_prevents_any_network_request(tmp_path):
    metadata, shards, _, _ = fixture_metadata(tmp_path)
    client = FakeRanges(shards)
    with pytest.raises(sample.SampleError, match='payloads alone'):
        sample.download_sample(metadata, tmp_path / 'sample', byte_cap=100, client=client)
    assert not client.requests


def test_verified_header_cost_must_fit_with_all_payloads_before_any_weight_request(tmp_path):
    metadata, shards, _, _ = fixture_metadata(tmp_path)
    payload_bytes = 8 * (130 * 260 + 24)
    cap = payload_bytes + 9
    client = FakeRanges(shards, byte_cap=cap)
    with pytest.raises(sample.SampleError, match='headers plus all paired'):
        sample.download_sample(metadata, tmp_path / 'sample', byte_cap=cap, client=client)
    assert len(client.requests) == 6
    for _, start, stop, _ in client.requests:
        assert start in (0, 8)
    assert not (tmp_path / 'sample').exists()


@pytest.mark.parametrize('projection,shape', [('gate_proj', (130, 260)), ('up_proj', (130, 260)),
                                             ('down_proj', (260, 130))])
def test_all_declared_matrix_shapes_are_checked_before_tensor_downloads(tmp_path, projection, shape):
    metadata, shards, _, _ = fixture_metadata(tmp_path, shape=shape, projection=projection)
    config_path = metadata / 'config.json'
    config = json.loads(config_path.read_text())
    config['moe_intermediate_size'] = 131
    config_path.write_text(json.dumps(config))
    client = FakeRanges(shards)
    with pytest.raises(sample.SampleError, match='shape or dtype'):
        sample.download_sample(metadata, tmp_path / 'sample', byte_cap=sample.MAX_SAMPLE_BYTES,
                               projection=projection, client=client)
    assert len(client.requests) == 6
    assert all(start in (0, 8) for _, start, _, _ in client.requests)
    assert not (tmp_path / 'sample').exists()


def test_invalid_block_scale_prevents_atomic_publish(tmp_path):
    metadata, shards, _, _ = fixture_metadata(tmp_path, bad_scale=True)
    with pytest.raises(sample.SampleError, match='finite and positive'):
        sample.download_sample(metadata, tmp_path / 'sample', byte_cap=sample.MAX_SAMPLE_BYTES,
                               client=FakeRanges(shards))
    assert not (tmp_path / 'sample').exists()


def test_existing_output_and_corrupt_saved_bytes_are_preserved_or_rejected(tmp_path):
    output, _, _, _, _ = downloaded(tmp_path)
    marker = output / 'operator.txt'
    marker.write_text('preserve')
    with pytest.raises(sample.SampleError, match='already exists'):
        sample.download_sample(tmp_path / 'metadata', output, byte_cap=sample.MAX_SAMPLE_BYTES)
    assert marker.read_text() == 'preserve'
    weight = output / 'expert_0.weight.bin'
    raw = bytearray(weight.read_bytes())
    raw[0] ^= 1
    weight.write_bytes(raw)
    with pytest.raises(sample.SampleError, match='hash'):
        sample.GLMWeightSampleReader(output)


def test_saved_manifest_corruption_and_path_changes_are_rejected(tmp_path):
    output, receipt, _, _, _ = downloaded(tmp_path)
    path = output / 'sample_manifest.json'
    raw = path.read_bytes() + b' '
    path.write_bytes(raw)
    with pytest.raises(sample.SampleError, match='manifest hash'):
        sample.GLMWeightSampleReader(output)
    receipt['tensors'][0]['filename'] = '../outside.bin'
    raw = json.dumps(receipt).encode()
    path.write_bytes(raw)
    (output / 'sample_manifest.sha256').write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(sample.SampleError, match='identity'):
        sample.GLMWeightSampleReader(output)
