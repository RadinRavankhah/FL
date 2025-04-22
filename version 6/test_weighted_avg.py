import numpy as np
import keras
from keras import layers

# Dummy Device and Model classes
class DummyDevice:
    def __init__(self, weights, data_size, last_round_participated):
        self.model = DummyModel(weights)
        self.data = (np.zeros((data_size, 10)), np.zeros((data_size,)))  # Dummy features and labels
        self.last_round_participated = last_round_participated

class DummyModel:
    def __init__(self, weights):
        self._weights = weights
    
    def get_weights(self):
        return self._weights

# Global model for comparison
global_model = keras.Sequential([
    layers.Dense(2, input_shape=(3,), use_bias=False)
])
global_model_weights = [np.array([[1., 1.], [1., 1.], [1., 1.]])]
global_model.set_weights(global_model_weights)

# Dummy device weights
device1_weights = [np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])]
device2_weights = [np.array([[0.6, 0.5], [0.4, 0.3], [0.2, 0.1]])]

# Create devices
device1 = DummyDevice(weights=device1_weights, data_size=100, last_round_participated=5)
device2 = DummyDevice(weights=device2_weights, data_size=50, last_round_participated=2)

devices = [device1, device2]
current_learning_iteration = 10  # For participation ratio

# === Use your original function (pasted below if needed) ===
# def aggregate_weights(devices):
#     """Computes the weighted average of model weights from all devices and updates the global model."""

#     num_devices = len(devices)
#     if num_devices == 0:
#         print("No devices available for aggregation.")
#         return

#     # Get device weights and participation ratios
#     device_weights = [device.model.get_weights() for device in devices]
#     device_participation_ratio = np.array(
#         [device.last_round_participated / current_learning_iteration for device in devices]
#     )

#     for item in device_participation_ratio:
#         print("Participation ratio:", item)

#     len_total_devices_data = 0
#     for device in devices:
#         len_total_devices_data += len(device.data[0])

#     weighted_sums = []

#     for layer in range(len(device_weights[0])):
#         layer_sum = []
#         for i in range(num_devices):
#             weight = device_weights[i][layer]
#             ratio = device_participation_ratio[i]
#             data_fraction = len(devices[i].data[0]) / float(len_total_devices_data)
#             weighted = weight * ratio * data_fraction
#             layer_sum.append(weighted)
#         weighted_sums.append(np.sum(np.stack(layer_sum), axis=0))

#     weighted_avg_weights = weighted_sums
#     return weighted_avg_weights



def aggregate_weights(devices):
    """Computes the weighted average of model weights from all devices and updates the global model."""

    num_devices = len(devices)
    if num_devices == 0:
        print("No devices available for aggregation.")
        return

    data_lengths = []
    device_weights = []
    device_participation_ratio = []
    for device in devices:
        device_weights.append(device.model.get_weights()[0])
        device_participation_ratio.append(device.last_round_participated / current_learning_iteration)
        data_lengths.append(len(device.data[0]))

    device_participation_ratio = np.array(device_participation_ratio)
    data_lengths = np.array(data_lengths)
    

    data_fractions = data_lengths / data_lengths.sum()

    # Element-wise multiplication to get the final weighting per device
    device_final_weights = device_participation_ratio * data_fractions  # shape (num_devices,)

    # Stack all device weights to a 3D tensor (num_devices, num_rows, num_cols)
    stacked_weights = np.stack(device_weights)  # shape: (num_devices, num_rows, num_cols)

    # Compute the weighted average using tensordot
    weighted_avg_weights = [np.tensordot(device_final_weights, stacked_weights, axes=1)]

    return weighted_avg_weights




# === Run the test ===

print("Global model weights BEFORE aggregation:")
print(global_model.get_weights()[0])

# Call your function
new_weights = aggregate_weights(devices)

# Set the new weights manually
global_model.set_weights(new_weights)

print("\nGlobal model weights AFTER aggregation:")
print(global_model.get_weights()[0])
