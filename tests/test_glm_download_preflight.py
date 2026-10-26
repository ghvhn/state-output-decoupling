from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'glm_preflight_under_test', Path(__file__).resolve().parents[1] / 'scripts/glm_download_preflight.py')
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)
SHA = 'a' * 40


def manifest(*, size=400, parameters=1000):
    return {
        'id': preflight.OFFICIAL_REPO, 'sha': SHA,
        'siblings': [
            {'rfilename': 'config.json', 'size': 20},
            {'rfilename': 'tokenizer.json', 'size': 30},
            {'rfilename': 'README.md', 'size': 25},
            {'rfilename': 'LICENSE', 'size': 25},
            {'rfilename': 'model.safetensors.index.json', 'size': 50},
            {'rfilename': 'model-00001-of-00002.safetensors', 'size': size},
            {'rfilename': 'model-00002-of-00002.safetensors', 'lfs': {'size': size}},
            {'rfilename': 'modeling_glm.py', 'size': 25},
        ],
        'safetensors': {'parameters': {'BF16': parameters}},
    }


def index():
    return {'weight_map': {
        'lm_head.weight': 'model-00001-of-00002.safetensors',
        'model.layers.0.mlp.experts.0.up_proj.weight': 'model-00002-of-00002.safetensors',
        'model.layers.0.mlp.experts.0.down_proj.weight': 'model-00002-of-00002.safetensors',
    }}


def plan(data=None, **kwargs):
    return preflight.make_plan(data or manifest(), {'model_type': 'glm_moe_dsa'},
                              index=index(), architecture={'config_registry_support': False}, **kwargs)


@pytest.mark.parametrize('name', ['config.json', 'tokenizer.model', 'model.safetensors.index.json', 'LICENSE'])
def test_metadata_allowlist_accepts_only_explicit_metadata(name):
    assert preflight.metadata_allowed(name)


@pytest.mark.parametrize('name', ['model.safetensors', 'pytorch_model.bin', 'modeling_glm.py',
                                 'tokenizer_weights.bin', 'subdir/config.json'])
def test_weights_code_and_unlisted_files_are_not_downloadable(name):
    assert not preflight.metadata_allowed(name)


@pytest.mark.parametrize('name', ['../config.json', '/config.json', 'C:/config.json',
                                 'subdir/../config.json', 'subdir\\config.json', './config.json',
                                 'config.json?download=true', 'subdir//config.json'])
def test_manifest_paths_reject_traversal_absolute_and_url_injection(name):
    with pytest.raises(preflight.PreflightError):
        preflight.safe_filename(name)


@pytest.mark.parametrize('revision', ['main', 'v1', 'a' * 39, 'A' * 40, None])
def test_only_immutable_full_commit_is_accepted(revision):
    with pytest.raises(preflight.PreflightError):
        preflight.validate_revision(revision)


def test_manifest_revision_and_official_repository_are_checked():
    assert preflight.verify_manifest(manifest(), SHA) == SHA
    with pytest.raises(preflight.PreflightError, match='pinned'):
        preflight.verify_manifest(manifest(), 'b' * 40)
    changed = manifest()
    changed['id'] = 'someone-else/GLM-5.3'
    with pytest.raises(preflight.PreflightError, match='official'):
        preflight.verify_manifest(changed)


def test_checkpoint_bytes_use_unique_index_shards_and_manifest_sizes():
    result = plan()
    assert result['metadata_ready_for_budgeting']
    assert result['checkpoint']['shard_count'] == 2
    assert result['checkpoint']['total_bytes'] == 800
    assert result['checkpoint']['manifest_parameter_count'] == 1000
    assert result['ideal_quantization_floors']['4bit']['weight_only_floor_bytes'] == 500
    assert result['ideal_quantization_floors']['2bit']['weight_only_floor_bytes'] == 250
    assert result['ideal_quantization_floors']['4bit']['source_plus_output_floor_bytes'] == 1300
    assert not result['compression_or_inference_feasibility_established']
    assert not result['compression_export_pipeline_implemented']
    assert result['downloaded_weight_shards'] == 0


@pytest.mark.parametrize('missing', ['size', 'parameters'])
def test_missing_sizes_or_counts_remain_unknown_and_refuse_budget_readiness(missing):
    data = manifest()
    if missing == 'size':
        del data['siblings'][5]['size']
    else:
        del data['safetensors']['parameters']
    result = plan(data)
    assert not result['metadata_ready_for_budgeting']
    assert result['ideal_quantization_floors']['4bit']['source_plus_output_floor_bytes'] is None
    assert result['warnings']


def test_without_index_multiple_checkpoints_do_not_claim_budget_readiness():
    result = preflight.make_plan(manifest(), {'model_type': 'glm_moe_dsa'},
                                 architecture={'config_registry_support': None})
    assert result['checkpoint']['total_bytes'] == 800
    assert not result['metadata_ready_for_budgeting']
    assert any('alternative checkpoints' in warning for warning in result['warnings'])


def test_weight_floors_beyond_ram_gpu_and_disk_are_exposed_without_success_claim():
    result = plan(manifest(size=400_000_000_000, parameters=744_000_000_000),
                  ram_gib=64, vram_gib=16, free_disk_bytes=862_917_779_456)
    four = result['ideal_quantization_floors']['4bit']
    assert four['weight_only_floor_bytes'] == 372_000_000_000
    assert four['source_plus_output_floor_bytes'] == 1_172_000_000_000
    assert four['source_plus_output_floor_fits_free_disk'] is False
    assert four['weight_only_floor_fits_ram'] is False
    assert four['weight_only_floor_fits_vram'] is False
    assert four['weight_only_floor_fits_combined_ram_vram'] is False
    assert result['ideal_quantization_floors']['2bit']['weight_only_floor_fits_combined_ram_vram'] is False
    assert any('does not support' in warning for warning in result['warnings'])


def test_index_cannot_name_absent_shards_or_other_formats():
    for name in ('missing.safetensors', 'model.py', '../model.safetensors'):
        with pytest.raises(preflight.PreflightError):
            preflight.make_plan(manifest(), {}, index={'weight_map': {'weight': name}}, architecture={})


class Response(io.BytesIO):
    def __init__(self, data, *, declared=None, url='https://huggingface.co/config.json'):
        super().__init__(data)
        self.headers = {} if declared is None else {'Content-Length': str(declared)}
        self.url = url

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return next(self.responses)


def test_network_bounds_enforced_with_or_without_content_length():
    for response in (Response(b'123456', declared=6), Response(b'123456')):
        client = preflight.MetadataClient(opener=Opener([response]))
        with pytest.raises(preflight.PreflightError, match='budget'):
            client.get('https://huggingface.co/config.json', limit=5)


def test_total_network_budget_is_shared_across_requests_and_has_no_auth():
    opener = Opener([Response(b'1234'), Response(b'abc')])
    client = preflight.MetadataClient(opener=opener, max_total_bytes=6)
    assert client.get('https://huggingface.co/config.json') == b'1234'
    with pytest.raises(preflight.PreflightError, match='budget'):
        client.get('https://huggingface.co/tokenizer.json')
    assert all(not request.has_header('Authorization') for request in opener.requests)


@pytest.mark.parametrize('url', ['http://huggingface.co/config.json', 'https://evil.test/config.json',
                                'https://token@huggingface.co/config.json'])
def test_requests_and_redirects_are_restricted_to_public_https_hosts(url):
    with pytest.raises(preflight.PreflightError):
        preflight._public_https(url)


class FakeClient:
    def __init__(self, data=None, pinned_sha=SHA):
        self.data = data or manifest()
        self.pinned_sha = pinned_sha
        self.urls = []
        self.received_bytes = 0

    def get(self, url, *, limit=preflight.MAX_FILE_BYTES):
        self.urls.append(url)
        if '/api/models/' in url:
            data = dict(self.data)
            if '/revision/' in url:
                data['sha'] = self.pinned_sha
            value = json.dumps(data).encode()
        elif url.endswith('/config.json'):
            value = b'{"model_type":"glm_moe_dsa"}'
        elif url.endswith('/model.safetensors.index.json'):
            value = json.dumps(index()).encode()
        elif url.endswith('/tokenizer.json'):
            value = b'{}'
        else:
            value = b'Public documentation'
        self.received_bytes += len(value)
        return value


def test_fetch_pins_all_resolved_files_and_never_requests_weights_or_code(tmp_path):
    client = FakeClient()
    preflight.fetch_metadata(tmp_path, client=client)
    resolved = [url for url in client.urls if '/resolve/' in url]
    assert len(resolved) == 5
    assert all('/resolve/' + SHA + '/' in url for url in resolved)
    assert all(preflight.metadata_allowed(url.rsplit('/', 1)[1]) for url in resolved)
    assert not any(url.endswith('.safetensors') or url.endswith('.py') for url in client.urls)
    receipt = json.loads((tmp_path / 'metadata_receipt.json').read_text())
    assert receipt['downloaded_weight_shards'] == 0
    assert receipt['downloaded_model_code_files'] == 0
    assert receipt['network_metadata_bytes'] < preflight.MAX_TOTAL_BYTES
    saved = preflight.offline_plan(tmp_path, revision=SHA, architecture={})
    assert saved['checkpoint']['total_bytes'] == 800


def test_changed_pin_stops_before_metadata_files_are_written(tmp_path):
    client = FakeClient(pinned_sha='b' * 40)
    with pytest.raises(preflight.PreflightError, match='pinned'):
        preflight.fetch_metadata(tmp_path, client=client)
    assert not list(tmp_path.iterdir())
    assert not any('/resolve/' in url for url in client.urls)


def test_fetch_wont_overwrite_unrelated_or_different_revision_output(tmp_path):
    marker = tmp_path / 'unrelated.txt'
    marker.write_text('preserve me')
    with pytest.raises(preflight.PreflightError, match='unrelated'):
        preflight.fetch_metadata(tmp_path, client=FakeClient())
    assert marker.read_text() == 'preserve me'
    changed = manifest()
    changed['sha'] = 'b' * 40
    (tmp_path / 'manifest.json').write_text(json.dumps(changed))
    with pytest.raises(preflight.PreflightError, match='pinned'):
        preflight.fetch_metadata(tmp_path, client=FakeClient())
    assert marker.read_text() == 'preserve me'


def test_offline_does_not_use_network_and_checks_saved_pin(tmp_path, monkeypatch):
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest()))
    (tmp_path / 'config.json').write_text('{"model_type":"glm_moe_dsa"}')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index()))
    monkeypatch.setattr(preflight, 'MetadataClient', lambda: pytest.fail('offline opened network'))
    assert preflight.offline_plan(tmp_path, revision=SHA, architecture={})['metadata_ready_for_budgeting']
    with pytest.raises(preflight.PreflightError, match='pinned'):
        preflight.offline_plan(tmp_path, revision='b' * 40, architecture={})


def test_atomic_failure_preserves_existing_file_and_cleans_temporary_file(tmp_path, monkeypatch):
    destination = tmp_path / 'config.json'
    destination.write_bytes(b'old')
    def fail_replace(*args):
        raise OSError('simulated replacement failure')
    monkeypatch.setattr(preflight.os, 'replace', fail_replace)
    with pytest.raises(OSError):
        preflight.atomic_write(destination, b'new')
    assert destination.read_bytes() == b'old'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['config.json']


def test_invalid_hardware_and_bool_parameter_counts_are_not_numbers():
    for capacity in (-1, float('inf'), float('nan'), True):
        with pytest.raises(preflight.PreflightError):
            plan(ram_gib=capacity)
    data = manifest()
    data['safetensors']['parameters'] = {'BF16': True}
    assert plan(data)['checkpoint']['manifest_parameter_count'] is None


def test_saved_metadata_is_bounded_before_json_parsing(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_bytes(b'x' * 11)
    with pytest.raises(preflight.PreflightError, match='byte budget'):
        preflight._local_metadata(path, limit=10)


def test_integer_floor_rounds_up_and_missing_hardware_does_not_imply_capacity():
    result = plan(manifest(parameters=1001))
    assert result['ideal_quantization_floors']['4bit']['weight_only_floor_bytes'] == 501
    assert result['ideal_quantization_floors']['2bit']['weight_only_floor_bytes'] == 251
    assert result['ideal_quantization_floors']['4bit']['weight_only_floor_fits_ram'] is None
    assert result['ideal_quantization_floors']['4bit']['source_plus_output_floor_fits_free_disk'] is None


def test_duplicate_inventory_is_rejected_instead_of_double_counted():
    data = manifest()
    data['siblings'].append(data['siblings'][5].copy())
    with pytest.raises(preflight.PreflightError, match='duplicate'):
        plan(data)
