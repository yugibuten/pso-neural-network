"""Compatibility wrapper for the original exercise scripts; ANN lives in ann.py."""
from .ann import ANN


class NeuralNetwork(ANN):
    def __init__(self, architecture, activations, seed=0):
        super().__init__(architecture, activations, seed)
