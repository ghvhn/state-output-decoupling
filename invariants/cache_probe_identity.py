"""Add opaque references to old cache evidence without changing latent vectors."""
from copy import deepcopy
from invariants.exposure_language import probe_alias


def migrate_probe_metadata(metadata):
    if not isinstance(metadata, dict):
        return metadata
    result = deepcopy(metadata)
    refs = dict(result.get('probe_ids') or {})
    for field in ('probe', 'probe_name', 'expert'):
        name = result.get(field)
        if not isinstance(name, str):
            continue
        if field == 'expert' and not name.startswith('probe_'):
            continue
        name = name.removeprefix('probe_')
        if name and not name.startswith('p_'):
            refs[field] = probe_alias(name)
    result['probe_identity_schema'] = 1
    if refs:
        result['probe_ids'] = refs
    return result
