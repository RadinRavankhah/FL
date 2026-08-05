import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================
# STEP 1: Build distribution from your devices
# ============================================================

def build_distribution(devices):
    num_devices = len(devices)
    num_classes = 10
    raw_counts = np.zeros((num_devices, num_classes), dtype=int)
    
    for i, device in enumerate(devices):
        labels = np.array(device.y).flatten()
        counts = Counter(labels)
        for label, count in counts.items():
            raw_counts[i, int(label)] = count
    
    row_sums = raw_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return raw_counts / row_sums

# ============================================================
# STEP 2: Metrics (all take 'devices' directly, or 'dist')
# ============================================================

def calculate_emd_per_device(devices=None, distribution=None):
    """Pass EITHER devices OR distribution"""
    from scipy.stats import wasserstein_distance
    
    dist = distribution if distribution is not None else build_distribution(devices)
    num_classes = dist.shape[1]
    global_dist = np.ones(num_classes) / num_classes
    
    emd_scores = []
    for i in range(len(dist)):
        # Scale up to integer samples for wasserstein_distance
        device_samples = np.repeat(np.arange(num_classes), (dist[i] * 10000).astype(int))
        global_samples = np.repeat(np.arange(num_classes), (global_dist * 10000).astype(int))
        emd = wasserstein_distance(device_samples, global_samples)
        emd_scores.append(emd)
    
    return np.array(emd_scores)

# ============================================================
# STEP 3: Visualization (all take 'devices' directly)
# ============================================================

def plot_heatmap(devices):
    dist = build_distribution(devices)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(devices) * 0.4)))
    
    im = ax.imshow(dist, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'{i}' for i in range(10)])
    ax.set_yticks(range(len(devices)))
    ax.set_yticklabels([f'Device {i}' for i in range(len(devices))])
    
    ax.set_xlabel('MNIST Digit')
    ax.set_ylabel('Device')
    ax.set_title('Label Distribution Heatmap\n(Yellow=none, Dark Red=concentrated)')
    
    # Add text annotations
    for i in range(len(devices)):
        for j in range(10):
            if dist[i, j] > 0.05:  # Only show if significant
                text = ax.text(j, i, f'{dist[i, j]:.2f}',
                             ha="center", va="center", color="black" if dist[i,j] < 0.5 else "white",
                             fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    plt.show()
    
    return dist

# ============================================================
# USAGE: Just pass your devices list
# ============================================================

# dist = plot_heatmap(your_devices)
# emd_scores = calculate_emd_per_device(devices=your_devices)