"""Metadata-only migration; preserves tensor dtype, values and entry order."""
import argparse
from pathlib import Path
import shutil
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invariants.cache_probe_identity import migrate_probe_metadata


def migrate_file(path):
    path = Path(path)
    entries = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(entries, list):
        raise ValueError(f'Unsupported cache format: {path}')
    changed = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('metadata'), dict):
            continue
        before = entry['metadata']
        after = migrate_probe_metadata(before)
        if before != after:
            entry['metadata'] = after
            changed += 1
    if changed:
        backup = path.with_suffix(path.suffix + '.before-probe-ids.bak')
        if not backup.exists():
            shutil.copy2(path, backup)
        temp = path.with_suffix(path.suffix + '.probe-ids.tmp')
        torch.save(entries, temp)
        temp.replace(path)
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path)
    for path in parser.parse_args().paths:
        print(f'{path}: {migrate_file(path)} metadata entries migrated')
