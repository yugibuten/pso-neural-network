"""Configurable feedforward ANN. No gradients or training libraries are used."""
import numpy as np
from .activations import REGISTRY


class ANN:
    def __init__(self, layer_sizes, activations, seed=0):
        if len(layer_sizes) < 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in layer_sizes):
            raise ValueError("Layer sizes must be positive integers")
        if len(activations) != len(layer_sizes)-1 or any(a not in REGISTRY for a in activations):
            raise ValueError("Provide one registered activation per non-input layer")
        if layer_sizes[-1] != 1 or activations[-1] not in ('logistic', 'sigmoid'):
            raise ValueError("Binary classification requires one logistic output")
        self.layer_sizes = list(layer_sizes)
        self.architecture = self.layer_sizes
        self.activations = list(activations)
        self.num_layers = len(activations)
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for ni, no in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6/(ni+no))
            self.weights.append(rng.uniform(-limit, limit, (ni, no)))
            self.biases.append(np.zeros((1, no)))

    @property
    def n_params(self):
        return sum((ni+1)*no for ni, no in zip(self.layer_sizes[:-1], self.layer_sizes[1:]))

    def set_params(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.n_params,) or not np.isfinite(vector).all():
            raise ValueError(f"Expected {self.n_params} finite parameters in a flat vector")
        offset = 0
        for w, b in zip(self.weights, self.biases):
            # In-place copies avoid aliasing the mutable particle array.
            w[:] = vector[offset:offset+w.size].reshape(w.shape)
            offset += w.size
            b[:] = vector[offset:offset+b.size].reshape(b.shape)
            offset += b.size

    def get_parameters_vector(self):
        return np.concatenate([a.ravel() for pair in zip(self.weights, self.biases) for a in pair])

    set_parameters_vector = set_params

    def forward(self, X):
        output = np.asarray(X, dtype=float)
        if output.ndim != 2 or output.shape[1] != self.layer_sizes[0] or not np.isfinite(output).all():
            raise ValueError("X must be finite with shape (samples, input_nodes)")
        for w, b, activation in zip(self.weights, self.biases, self.activations):
            output = REGISTRY[activation](output @ w + b)
        return output

    def predict(self, X):
        return (self.forward(X).ravel() >= 0.5).astype(int)
