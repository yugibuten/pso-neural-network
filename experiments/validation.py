"""Validate complete saved suites against their recorded configuration list."""
import json
from pathlib import Path


def validate_complete_suite(frame, csv_path):
    """Reject missing settings, altered configs, duplicate or incomplete seeds.

    Uses identifiers/configurations only, not performance values. A companion
    metadata file is mandatory so an entirely absent setting cannot go unnoticed.
    """
    metadata_path = Path(csv_path).with_suffix('.metadata.json')
    if not metadata_path.exists():
        raise ValueError(f'Missing suite metadata: {metadata_path}')
    metadata = json.loads(metadata_path.read_text())
    configs = metadata['configs']
    expected = {config['name']: config for config in configs}
    seeds = metadata['seeds']
    if not expected or len(expected) != len(configs) or sorted(seeds) != list(range(10)):
        raise ValueError('Complete-study metadata must list unique settings and seeds 0–9')
    required = {'name', 'seed', 'config_json'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('CSV needs nonempty name, seed and config_json fields')
    actual = set(frame['name'])
    if actual != set(expected):
        raise ValueError(f'Setting mismatch: missing={sorted(set(expected)-actual)}, unexpected={sorted(actual-set(expected))}')
    for name, group in frame.groupby('name'):
        if sorted(group.seed.tolist()) != seeds:
            raise ValueError(f'{name}: expected exactly seeds 0–9')
        if any(json.loads(value) != expected[name] for value in group.config_json):
            raise ValueError(f'{name}: CSV configuration differs from metadata')
    return metadata
