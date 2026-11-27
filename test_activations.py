import numpy as np

from src.activations import apply_activation


values = np.array([-2, -1, 0, 1, 2])

print("Input:", values)

print("Sigmoid:", apply_activation(values, "sigmoid"))
print("ReLU:", apply_activation(values, "relu"))
print("Tanh:", apply_activation(values, "tanh"))