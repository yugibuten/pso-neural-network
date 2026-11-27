import numpy as np

from src.neural_network import NeuralNetwork


# Baseline architecture
architecture = [4, 8, 1]

activations = ["tanh", "sigmoid"]


# Create network
ann = NeuralNetwork(
    architecture=architecture,
    activations=activations
)


# Generate five sample inputs
X = np.random.randn(5, 4)


# Forward propagation
predictions = ann.forward(X)


print("Architecture:", architecture)

print("Weight shapes:")
for weight in ann.weights:
    print(weight.shape)

print("\nBias shapes:")
for bias in ann.biases:
    print(bias.shape)

print("\nPredictions:")
print(predictions)

print("\nPrediction shape:", predictions.shape)

print("\nTotal parameters:",
      sum(w.size for w in ann.weights) +
      sum(b.size for b in ann.biases))