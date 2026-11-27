"""Portable loader with a stratified split and train-only standardisation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'data/data_banknote_authentication.txt'


def load_data(path=DEFAULT_PATH):
    values = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if values.shape != (1372, 5) or not np.isfinite(values).all() or not np.isin(values[:, 4], [0, 1]).all():
        raise ValueError("Expected 1372 finite rows, four features and a binary label")
    return values[:, :4], values[:, 4].astype(int)


def prepare_data(path=DEFAULT_PATH, split_seed=42):
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=split_seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, scaler


if __name__ == '__main__':
    a, b, c, d, scaler = prepare_data()
    print('Train:', a.shape, c.shape, 'Test:', b.shape, d.shape)
    print('Train feature means:', np.round(a.mean(axis=0), 10))
