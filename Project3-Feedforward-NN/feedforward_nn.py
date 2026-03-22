# ============================================================
# PROJECT 3: Feedforward Neural Network - Digit Recognition
# Course: Artificial Neural Network (24SBT113)
# IV Sem Sec-A | Srinivas University
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ------------------------------------------------------------
# STEP 1: Load Dataset
# We use sklearn's built-in digits dataset
# 1797 images of handwritten digits (0-9), each 8x8 pixels
# ------------------------------------------------------------
digits = load_digits()
X = digits.data / 16.0        # Normalize pixel values to 0-1
y = digits.target.reshape(-1, 1)

print("=" * 55)
print("  PROJECT 3: Feedforward Neural Network")
print("  Pattern Classification: Digit Recognition")
print("=" * 55)
print(f"  Total samples  : {X.shape[0]}")
print(f"  Features/Input : {X.shape[1]} (8x8 pixel image)")
print(f"  Classes        : 10 (digits 0 to 9)")

# One-hot encode labels (e.g. digit 3 -> [0,0,0,1,0,0,0,0,0,0])
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Train/Test split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Testing samples : {X_test.shape[0]}")

# ------------------------------------------------------------
# STEP 2: Activation Functions
# ------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ------------------------------------------------------------
# STEP 3: Initialize Network Weights
# Architecture: 64 (input) -> 64 (hidden) -> 10 (output)
# ------------------------------------------------------------
np.random.seed(42)

input_size  = 64   # 8x8 pixels flattened
hidden_size = 64   # Hidden layer neurons
output_size = 10   # Digits 0-9

# Xavier initialization for better learning
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))

print(f"\n  Network Architecture:")
print(f"  Input Layer  : {input_size} neurons")
print(f"  Hidden Layer : {hidden_size} neurons (Sigmoid)")
print(f"  Output Layer : {output_size} neurons (Softmax)")

# ------------------------------------------------------------
# STEP 4: Train the Feedforward Neural Network
# Using Backpropagation
# ------------------------------------------------------------
learning_rate = 0.1
epochs        = 500
batch_size    = 32

train_losses = []
train_accuracies = []

print(f"\n  Training started...")
print(f"  Epochs: {epochs} | Learning Rate: {learning_rate}")
print("-" * 55)

for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0
    correct = 0

    # Mini-batch training
    for start in range(0, X_train.shape[0], batch_size):
        Xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]

        # --- FORWARD PASS ---
        z1 = Xb @ W1 + b1          # Hidden layer input
        a1 = sigmoid(z1)            # Hidden layer output
        z2 = a1 @ W2 + b2          # Output layer input
        a2 = softmax(z2)            # Final predictions (probabilities)

        # --- LOSS (Cross Entropy) ---
        loss = -np.mean(np.sum(yb * np.log(a2 + 1e-8), axis=1))
        epoch_loss += loss

        # --- BACKWARD PASS (Backpropagation) ---
        dz2 = a2 - yb                          # Output error
        dW2 = a1.T @ dz2 / len(Xb)
        db2 = np.mean(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * sigmoid_derivative(z1)    # Hidden error
        dW1 = Xb.T @ dz1 / len(Xb)
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # --- UPDATE WEIGHTS ---
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        correct += np.sum(np.argmax(a2, axis=1) == np.argmax(yb, axis=1))

    avg_loss = epoch_loss / (X_train.shape[0] / batch_size)
    accuracy = correct / X_train.shape[0] * 100
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}%")

# ------------------------------------------------------------
# STEP 5: Test the Network
# ------------------------------------------------------------
z1_test = X_test @ W1 + b1
a1_test = sigmoid(z1_test)
z2_test = a1_test @ W2 + b2
a2_test = softmax(z2_test)

y_pred = np.argmax(a2_test, axis=1)
y_true = np.argmax(y_test, axis=1)

test_accuracy = np.mean(y_pred == y_true) * 100

print("\n" + "=" * 55)
print(f"  TEST ACCURACY : {test_accuracy:.2f}%")
print("=" * 55)
print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=[str(i) for i in range(10)]))

# ------------------------------------------------------------
# STEP 6: Visualize Results
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Project 3: Feedforward Neural Network - Digit Recognition\nANN (24SBT113) | Srinivas University',
             fontsize=13, fontweight='bold')

# Graph 1: Training Loss
axes[0, 0].plot(range(1, epochs+1), train_losses, color='#E74C3C', linewidth=2)
axes[0, 0].set_title('Training Loss over Epochs')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Cross-Entropy Loss')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].fill_between(range(1, epochs+1), train_losses, alpha=0.2, color='#E74C3C')

# Graph 2: Training Accuracy
axes[0, 1].plot(range(1, epochs+1), train_accuracies, color='#2ECC71', linewidth=2)
axes[0, 1].set_title('Training Accuracy over Epochs')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].fill_between(range(1, epochs+1), train_accuracies, alpha=0.2, color='#2ECC71')

# Graph 3: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10), ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted Digit')
axes[1, 0].set_ylabel('Actual Digit')

# Graph 4: Sample predictions
axes[1, 1].axis('off')
sample_indices = np.random.choice(len(X_test), 16, replace=False)
grid = np.zeros((4*8, 4*8))
pred_labels = []
for idx, si in enumerate(sample_indices):
    r, c = divmod(idx, 4)
    grid[r*8:(r+1)*8, c*8:(c+1)*8] = X_test[si].reshape(8, 8)
    pred_labels.append(f"{y_pred[si]}")

axes[1, 1].imshow(grid, cmap='gray_r')
axes[1, 1].set_title(f'Sample Predictions (Test Accuracy: {test_accuracy:.1f}%)')
axes[1, 1].axis('on')
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])

plt.tight_layout()
plt.savefig('feedforward_nn_result.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGraph saved as 'feedforward_nn_result.png'")
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Algorithm    : Feedforward NN + Backpropagation")
print(f"  Dataset      : Sklearn Digits (handwritten 0-9)")
print(f"  Architecture : 64 -> 64 -> 10")
print(f"  Epochs       : {epochs}")
print(f"  Test Accuracy: {test_accuracy:.2f}%")
print("=" * 55) 