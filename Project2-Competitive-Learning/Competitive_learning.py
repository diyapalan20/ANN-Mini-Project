# ============================================================
# PROJECT 2: 2-Layer Competitive Learning Network
# Course: Artificial Neural Network (24SBT113)
# IV Sem Sec-A | Srinivas University
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 1: Generate Noisy 2D Data (3 clusters/shapes)
# We create 3 groups of points with some random noise added
# ------------------------------------------------------------
np.random.seed(42)  # So results are same every time

num_points = 50  # Points per cluster

# Cluster 1: around center (2, 2)
cluster1 = np.random.randn(num_points, 2) * 0.5 + [2, 2]

# Cluster 2: around center (7, 3)
cluster2 = np.random.randn(num_points, 2) * 0.5 + [7, 3]

# Cluster 3: around center (4, 8)
cluster3 = np.random.randn(num_points, 2) * 0.5 + [4, 8]

# Combine all clusters into one dataset
X = np.vstack([cluster1, cluster2, cluster3])

print("=" * 55)
print("   PROJECT 2: Competitive Learning Network")
print("=" * 55)
print(f"Total data points: {len(X)}")
print(f"Each point has 2 features (x, y coordinates)")
print(f"3 clusters generated with added noise")

# ------------------------------------------------------------
# STEP 2: Set Up the Competitive Learning Network
# ------------------------------------------------------------
num_neurons   = 3      # 3 output neurons (one per cluster)
learning_rate = 0.3    # How fast neurons move toward data
epochs        = 100    # Number of training rounds

# Initialize neuron weights randomly within data range
weights = np.random.uniform(
    low  = X.min(axis=0),
    high = X.max(axis=0),
    size = (num_neurons, 2)
)

print(f"\nInitial neuron positions (random):")
for i, w in enumerate(weights):
    print(f"  Neuron {i+1}: ({w[0]:.2f}, {w[1]:.2f})")

# ------------------------------------------------------------
# STEP 3: Train the Network (Winner Takes All)
# ------------------------------------------------------------
print(f"\nTraining for {epochs} epochs...")

error_history = []

for epoch in range(epochs):
    total_error = 0

    # Shuffle data each epoch for better learning
    np.random.shuffle(X)

    for point in X:
        # --- Find the WINNER neuron ---
        # Calculate distance from point to each neuron
        distances = np.linalg.norm(weights - point, axis=1)

        # Winner = neuron with SMALLEST distance
        winner = np.argmin(distances)

        # --- Update ONLY the winner's weights ---
        # Move the winner neuron closer to this data point
        old_weight = weights[winner].copy()
        weights[winner] += learning_rate * (point - weights[winner])

        # Track how much the weight changed (error)
        total_error += np.linalg.norm(weights[winner] - old_weight)

    error_history.append(total_error)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | Total movement = {total_error:.4f}")

print("\nTraining Complete!")
print(f"\nFinal neuron positions (after learning):")
for i, w in enumerate(weights):
    print(f"  Neuron {i+1}: ({w[0]:.2f}, {w[1]:.2f})")

# ------------------------------------------------------------
# STEP 4: Assign Each Point to Its Winning Neuron (Testing)
# ------------------------------------------------------------
labels = []
for point in X:
    distances = np.linalg.norm(weights - point, axis=1)
    labels.append(np.argmin(distances))
labels = np.array(labels)

# Count how many points each neuron claimed
for i in range(num_neurons):
    count = np.sum(labels == i)
    print(f"  Neuron {i+1} claimed {count} points")

# ------------------------------------------------------------
# STEP 5: Plot Results - 2 Graphs
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Project 2: Competitive Learning Network\nANN (24SBT113) | Srinivas University',
             fontsize=13, fontweight='bold')

colors = ['#E74C3C', '#2ECC71', '#3498DB']  # Red, Green, Blue
labels_text = ['Cluster 1', 'Cluster 2', 'Cluster 3']

# --- Graph 1: Clustered Data ---
ax1 = axes[0]
for i in range(num_neurons):
    mask = labels == i
    ax1.scatter(X[mask, 0], X[mask, 1],
                c=colors[i], label=labels_text[i],
                alpha=0.6, s=40)

# Plot neuron positions as black stars
ax1.scatter(weights[:, 0], weights[:, 1],
            c='black', marker='*', s=300,
            zorder=5, label='Neuron Centers')

# Label each neuron
for i, w in enumerate(weights):
    ax1.annotate(f'N{i+1}', xy=(w[0], w[1]),
                 xytext=(w[0]+0.3, w[1]+0.3),
                 fontsize=11, fontweight='bold', color='black')

ax1.set_title('After Training: Clusters Detected', fontsize=12)
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Graph 2: Error/Convergence Graph ---
ax2 = axes[1]
ax2.plot(range(1, epochs+1), error_history,
         color='#8E44AD', linewidth=2, marker='o', markersize=2)
ax2.set_title('Learning Convergence (Error per Epoch)', fontsize=12)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Total Weight Movement')
ax2.grid(True, alpha=0.3)
ax2.fill_between(range(1, epochs+1), error_history, alpha=0.2, color='#8E44AD')

plt.tight_layout()
plt.savefig('competitive_learning_result.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGraph saved as 'competitive_learning_result.png'")
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Algorithm   : Competitive Learning (Winner Takes All)")
print(f"  Dataset     : 150 noisy 2D points (3 clusters)")
print(f"  Neurons     : 3 output neurons")
print(f"  Epochs      : {epochs}")
print(f"  Result      : Each neuron specialized in 1 cluster")
print("=" * 55)