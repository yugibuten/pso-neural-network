"""Training-only fitness closures. Every fitness is maximised."""
import numpy as np
from .activations import REGISTRY

ENCODED_ACTIVATIONS = ('tanh', 'logistic', 'relu')


def particle_dim(ann, encode_activations=False):
    return ann.n_params + (len(ann.activations)-1 if encode_activations else 0)


def apply_particle(ann, vector, encode_activations=False):
    vector = np.asarray(vector)
    if vector.shape != (particle_dim(ann, encode_activations),):
        raise ValueError('Wrong particle dimension')
    ann.set_params(vector[:ann.n_params])
    if encode_activations:
        # Extra coordinates in [0,3) select hidden activations by integer bins.
        for i, coordinate in enumerate(vector[ann.n_params:]):
            ann.activations[i] = ENCODED_ACTIVATIONS[int(np.clip(np.floor(coordinate), 0, 2))]


def make_fitness(ann, X_train, y_train, mode='accuracy', encode_activations=False):
    if mode not in ('accuracy', 'mse', 'bce'):
        raise ValueError('Fitness must be accuracy, mse or bce')
    X = np.asarray(X_train, dtype=float).copy()
    y = np.asarray(y_train).ravel().copy()
    if X.ndim != 2 or len(X) != len(y) or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected aligned training samples and binary labels')

    def fitness(vector):
        apply_particle(ann, vector, encode_activations)
        probabilities = ann.forward(X).ravel()
        if mode == 'accuracy':
            return float(np.mean((probabilities >= .5) == y))
        if mode == 'mse':
            return -float(np.mean((probabilities-y)**2))
        probabilities = np.clip(probabilities, 1e-12, 1-1e-12)
        return float(np.mean(y*np.log(probabilities)+(1-y)*np.log1p(-probabilities)))
    return fitness
