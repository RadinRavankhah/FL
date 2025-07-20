import numpy as np
import matplotlib.pyplot as plt

# ----------- Non-IID Dirichlet Split Function -----------
def niid_labeldir_split(x_data, y_data, num_clients, beta, seed=None):
    num_classes = 10
    y_indices = np.array([np.argmax(label) for label in y_data])  # From one-hot to class index

    rng = np.random.default_rng(seed)

    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(y_indices == k)[0]
        rng.shuffle(idx_k)

        proportions = rng.dirichlet(np.repeat(beta, num_clients))
        proportions = np.array([int(p * len(idx_k)) for p in proportions])

        while sum(proportions) < len(idx_k):
            proportions[np.argmin(proportions)] += 1
        while sum(proportions) > len(idx_k):
            proportions[np.argmax(proportions)] -= 1

        start = 0
        for i in range(num_clients):
            size = proportions[i]
            client_indices[i].extend(idx_k[start:start + size])
            start += size

    return client_indices

# ----------- Test the Split Function -----------

# Simulate dummy MNIST-style data (1000 samples)
num_samples = 1000
num_classes = 10
num_clients = 30
beta = 0.5
seed = 1
global_rng = np.random.default_rng(seed)
# Replace all np.random with rng

# Random images (28x28 grayscale) and one-hot labels
x_dummy = global_rng.random((num_samples, 28, 28, 1))
y_dummy = np.eye(num_classes)[global_rng.integers(0, num_classes, size=num_samples)]

# Perform the NIID split
split_indices = niid_labeldir_split(x_dummy, y_dummy, num_clients, beta, seed)

# Print label distribution for each client
print("\nLabel distribution per client:")
for i in range(num_clients):
    labels = np.argmax(y_dummy[split_indices[i]], axis=1)
    print(f"Client {i}: {np.bincount(labels, minlength=10)}")

# Optional: Plot label histograms
fig, axs = plt.subplots(1, num_clients, figsize=(15, 3))
for i in range(num_clients):
    labels = np.argmax(y_dummy[split_indices[i]], axis=1)
    axs[i].bar(range(10), np.bincount(labels, minlength=10))
    axs[i].set_title(f"Client {i}")
plt.tight_layout()
plt.show()
