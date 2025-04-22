import numpy as np
from types import SimpleNamespace

# Mock Device class
class Device:
    def __init__(self, weights, last_round_participated, data_length):
        self.model = SimpleNamespace(get_weights=lambda: [weights])
        self.last_round_participated = last_round_participated
        self.data = (np.zeros((data_length, 2)), np.zeros((data_length,)))

# Create mock devices
device1 = Device(weights=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), last_round_participated=5, data_length=100)
device2 = Device(weights=np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]]), last_round_participated=2, data_length=50)
devices = [device1, device2]

# Mock global learning iteration
current_learning_iteration = 10  # You should have this in your actual context

# Aggregation
aggregated_weights = aggregate_weights(devices)

# Create a dummy global model
class GlobalModel:
    def __init__(self):
        self.weights = None
    def set_weights(self, w):
        self.weights = w
    def get_weights(self):
        return self.weights

global_model = GlobalModel()
global_model.set_weights(aggregated_weights)

# Re-calculate using np.dot() for verification
device_weights = [device.model.get_weights()[0] for device in devices]
ratios = np.array([device.last_round_participated / current_learning_iteration for device in devices])
data_fractions = np.array([len(device.data[0]) for device in devices])
data_fractions = data_fractions / data_fractions.sum()
combined_weights = ratios * data_fractions  # shape: (num_devices,)

# Stack weights to shape (num_devices, num_rows, num_cols)
stacked_weights = np.stack(device_weights)  # (2, 3, 2)

# Multiply each device's weights by its scalar weight using np.dot()
# We'll reshape and use broadcasting
final_weights = np.tensordot(combined_weights, stacked_weights, axes=1)  # (3, 2)

# Output to compare
print("Global model weights AFTER aggregation:")
print(global_model.get_weights()[0])

print("\nWeights calculated using np.dot / tensordot:")
print(final_weights)

print("\nAre they close?:", np.allclose(global_model.get_weights()[0], final_weights))
