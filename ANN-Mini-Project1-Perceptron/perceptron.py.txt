# ============================================
# PROJECT 1: Perceptron Model
# Course: Artificial Neural Network (24SBT113)
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# STEP 1: Define the dataset
# We are using AND gate logic
# ----------------------------
# Inputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected outputs (AND gate)
y = np.array([0, 0, 0, 1])

# ----------------------------
# STEP 2: Set starting values
# ----------------------------
weights = np.zeros(2)   # Start with weight = 0 for each input
bias = 0                # Bias starts at 0
learning_rate = 0.1     # How fast the model learns
epochs = 10             # How many times to repeat training

# ----------------------------
# STEP 3: Activation Function
# If sum >= 0, output = 1
# If sum < 0,  output = 0
# ----------------------------
def activation(value):
    return 1 if value >= 0 else 0

# ----------------------------
# STEP 4: Train the Perceptron
# ----------------------------
print("=" * 45)
print("      PERCEPTRON TRAINING - AND GATE")
print("=" * 45)

errors_per_epoch = []

for epoch in range(epochs):
    total_error = 0

    for i in range(len(X)):
        # Calculate output
        weighted_sum = np.dot(X[i], weights) + bias
        prediction = activation(weighted_sum)

        # Calculate error
        error = y[i] - prediction

        # Update weights and bias
        weights += learning_rate * error * X[i]
        bias    += learning_rate * error

        total_error += abs(error)

    errors_per_epoch.append(total_error)
    print(f"Epoch {epoch+1}: Total Error = {total_error} | Weights = {weights} | Bias = {bias:.2f}")

# ----------------------------
# STEP 5: Test the Perceptron
# ----------------------------
print("\n" + "=" * 45)
print("         TESTING THE PERCEPTRON")
print("=" * 45)
print(f"{'Input':<15} {'Expected':<12} {'Predicted'}")
print("-" * 45)

correct = 0
for i in range(len(X)):
    result = activation(np.dot(X[i], weights) + bias)
    status = "✅ Correct" if result == y[i] else "❌ Wrong"
    print(f"{str(X[i]):<15} {y[i]:<12} {result}   {status}")
    if result == y[i]:
        correct += 1

accuracy = (correct / len(X)) * 100
print(f"\nAccuracy: {accuracy}%")

# ----------------------------
# STEP 6: Plot the Error Graph
# ----------------------------
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs+1), errors_per_epoch, marker='o', color='blue')
plt.title('Perceptron Training - Error per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Total Error')
plt.grid(True)
plt.tight_layout()
plt.savefig('error_graph.png')  # Saves the graph as image
plt.show()
print("\nGraph saved as 'error_graph.pngs