"""Elementwise activations; logistic is evaluated without exponential overflow."""
import numpy as np


def logistic(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0, -x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


sigmoid = logistic  # Compatibility with the original exercises.
REGISTRY = {"logistic": logistic, "sigmoid": logistic, "relu": relu, "tanh": tanh}


def apply_activation(x, activation):
    if activation not in REGISTRY:
        raise ValueError(f"Unknown activation: {activation}")
    return REGISTRY[activation](x)
