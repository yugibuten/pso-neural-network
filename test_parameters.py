
import numpy as np

from src.neural_network import NeuralNetwork


# Create ANN
ann = NeuralNetwork(
    architecture=[4, 8, 1],
    activations=["tanh", "sigmoid"]
)


# Get original predictions
X = np.random.randn(5, 4)
original_predictions = ann.forward(X)


# Flatten parameters
vector = ann.get_parameters_vector()

print("Vector shape:", vector.shape)
print("Total parameters:", len(vector))

print("\nFirst 10 parameters:")
print(vector[:10])


# Reconstruct parameters
ann.set_parameters_vector(vector)


# Get predictions again
new_predictions = ann.forward(X)

print("\nOriginal predictions:")
print(original_predictions)

print("\nReconstructed predictions:")
print(new_predictions)


# Check whether both predictions match
print(
    "\nRound-trip successful:",
    np.allclose(original_predictions, new_predictions)
)
