"""Seeded experiment runner: runs each configuration over seeds 0-9 and saves results."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from src.ann import ANN
from src.coupling import make_fitness, apply_particle, particle_dim
from src.data import prepare_data, DEFAULT_PATH
from src.metrics import classification_metrics, confusion_matrix
from src.pso import PSO
from .validation import validate_complete_suite

ROOT = Path(__file__).resolve().parents[1]


def run_one(config, seed, data):
    X_train, X_test, y_train, y_test, scaler = data
    ann = ANN(config['layer_sizes'], config['activations'], seed)
    encoded = config['encode_activations']
    dim = particle_dim(ann, encoded)
    low = np.full(dim, config['bounds'][0], dtype=float)
    high = np.full(dim, config['bounds'][1], dtype=float)
    if encoded:
        low[ann.n_params:], high[ann.n_params:] = 0., np.nextafter(3., 0.)
    fitness = make_fitness(ann, X_train, y_train, config['fitness'], encoded)
    keys = ('swarm_size','alpha','beta','gamma','delta','epsilon','n_informants','boundary','alpha_end')
    solver = PSO(dim, fitness, seed=seed, bounds=(low, high), **{k:config[k] for k in keys})
    start = perf_counter()
    best, best_fitness, history = solver.run(config['iterations'])
    elapsed = perf_counter()-start
    apply_particle(ann, best, encoded)
    train = classification_metrics(y_train, ann.predict(X_train))
    # Single test-set forward pass per run; all test metrics reuse these predictions.
    test_predictions = ann.predict(X_test)
    test = classification_metrics(y_test, test_predictions)
    row = dict(name=config['name'], seed=seed, config_json=json.dumps(config, sort_keys=True),
               n_params=ann.n_params, particle_dim=dim, best_fitness=best_fitness,
               seconds=elapsed, evaluations=solver.n_evaluations,
               history_json=json.dumps(history.tolist()),
               test_confusion_json=json.dumps(confusion_matrix(y_test,test_predictions).tolist()),
               chosen_activations_json=json.dumps(ann.activations))
    row.update({'train_'+k:v for k,v in train.items()})
    row.update({'test_'+k:v for k,v in test.items()})
    artifact = dict(best_position=best, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                    config_json=json.dumps(config), seed=seed, history=history)
    return row, artifact


def _atomic_write(destination, writer):
    """Replace a file only after a successful write on the same filesystem."""
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name+'.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        writer(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _save_artifact(destination, artifact):
    # A file handle prevents NumPy appending .npz to the temporary filename.
    def write(path):
        with path.open('wb') as stream:
            np.savez_compressed(stream, **artifact)
    _atomic_write(destination, write)


def run_suite(configs, output, data_path=DEFAULT_PATH, seeds=range(10), resume=False):
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f'{output} already exists; use --resume to continue it')
    configs, seeds = list(configs), list(seeds)
    if len({c['name'] for c in configs}) != len(configs) or len(set(seeds)) != len(seeds):
        raise ValueError('Configuration names and seeds must be unique')
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output.parent / (output.stem+'_artifacts')
    artifacts.mkdir(exist_ok=True)
    metadata = dict(split_seed=42, seeds=seeds, numpy=np.__version__, pandas=pd.__version__,
                    dataset_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                    configs=configs, iteration_definition='assessed generations including initial swarm')
    metadata_path = output.with_suffix('.metadata.json')
    rows, completed = [], set()
    if resume:
        if not output.exists() or not metadata_path.exists():
            raise ValueError('Resume requires the existing CSV and metadata JSON')
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError('Cannot resume: configs, seeds, dataset or recorded library versions changed')
        existing = pd.read_csv(output)
        expected = {c['name']: c for c in configs}
        for row in existing.to_dict('records'):
            key = (row['name'], row['seed'])
            if (key in completed or row['name'] not in expected or row['seed'] not in seeds
                    or json.loads(row['config_json']) != expected[row['name']]):
                raise ValueError('CSV contains duplicate, unexpected or mismatched runs')
            slug = hashlib.sha256(row['name'].encode()).hexdigest()[:12]
            artifact_path = artifacts/f'{slug}_seed{row["seed"]}.npz'
            try:
                with np.load(artifact_path, allow_pickle=False) as stored:
                    for field in ('best_position','scaler_mean','scaler_scale','config_json','seed','history'):
                        stored[field]  # Force decompression so truncated files are rejected.
            except Exception as exc:
                raise ValueError(f'Completed run has a missing or damaged artifact: {artifact_path}') from exc
            completed.add(key)
        rows = existing.to_dict('records')
        print(f'Resuming: {len(completed)} completed runs will be skipped', flush=True)
    else:
        _atomic_write(metadata_path, lambda path: path.write_text(json.dumps(metadata, indent=2)))
    if len(completed) == len(configs)*len(seeds):
        print('All runs already saved', flush=True)
        return
    data = prepare_data(data_path)
    for config in configs:
        for seed in seeds:
            if (config['name'], seed) in completed:
                continue
            row, artifact = run_one(config, seed, data)
            slug = hashlib.sha256(config['name'].encode()).hexdigest()[:12]
            _save_artifact(artifacts/f'{slug}_seed{seed}.npz', artifact)
            # Atomically replace the CSV so a full disk cannot truncate prior rows.
            updated = rows + [row]
            _atomic_write(output, lambda path: pd.DataFrame(updated).to_csv(path, index=False))
            rows = updated
            print(f"Saved {config['name']} seed {seed}", flush=True)


def select_training(csv_path, destination):
    """Select architecture by mean training accuracy; never read test columns."""
    df = pd.read_csv(csv_path, usecols=['name','seed','train_accuracy','config_json'])
    validate_complete_suite(df, csv_path)
    ranking = df.groupby('name').train_accuracy.mean().sort_values(ascending=False, kind='stable')
    if ranking.empty:
        raise ValueError('No settings to select')
    selected = json.loads(df.loc[df.name == ranking.index[0], 'config_json'].iloc[0])
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selected, indent=2))
    print(f'Saved training-selected config to {destination}')


def common_parser(description, default_output, needs_base=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output', type=Path, default=ROOT/'results/csv'/default_output)
    parser.add_argument('--data', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--resume', action='store_true', help='Validate and skip completed runs')
    parser.add_argument('--list', action='store_true', help='Print configs without running')
    if needs_base:
        parser.add_argument('--base', required=True, type=Path, help='Training-selected config JSON')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['select'])
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    select_training(args.csv, args.output)
