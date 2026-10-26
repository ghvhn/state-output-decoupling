import torch
from invariants.cache_probe_identity import migrate_probe_metadata
from invariants.exposure_language import probe_alias
from scripts.migrate_cache_probe_ids import migrate_file


def test_metadata_migration_is_additive_and_idempotent():
    source = {'expert': 'probe_regret', 'cache_write_scope': 'agent', 'reason': 'synthesis'}
    migrated = migrate_probe_metadata(source)
    assert migrated['expert'] == source['expert']
    assert migrated['probe_ids']['expert'] == probe_alias('regret')
    assert 'probe_ids' not in source
    assert migrate_probe_metadata(migrated) == migrated


def test_retired_cache_probe_labels_are_anonymized():
    from invariants.exposure_language import obscure_probe_identifiers
    assert obscure_probe_identifiers('expert=probe_retired', {}) == 'expert=' + probe_alias('retired')


def test_saved_migration_preserves_tensors_and_backup(tmp_path):
    path = tmp_path/'cognitive_cache.pt'
    trigger = torch.randn(4, dtype=torch.float16)
    delta = torch.randn(4)
    torch.save([{'trigger': trigger, 'delta': delta, 'metadata': {'expert': 'probe_curiosity'}}], path)
    original = path.read_bytes()
    assert migrate_file(path) == 1
    restored = torch.load(path, weights_only=True)
    assert torch.equal(restored[0]['trigger'], trigger)
    assert torch.equal(restored[0]['delta'], delta)
    assert restored[0]['trigger'].dtype == trigger.dtype
    assert path.with_suffix('.pt.before-probe-ids.bak').read_bytes() == original
    assert migrate_file(path) == 0
