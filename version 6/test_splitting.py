import numpy as np

# Simulated fake data
x_train = np.arange(80).reshape((20, 4))    # 20 samples, 4 features each
y_train = np.arange(20)                     # 20 labels

x_test = np.arange(40).reshape((10, 4))      # 10 samples, 4 features each
y_test = np.arange(10)                      # 10 labels

# Simulated device class
class Device:
    def __init__(self, device_id):
        self.device_id = device_id
        self.data = None
        self.test_data = None

# Create 2 fake devices
devices = [Device(device_id=i) for i in range(2)]

# --- Splitting ---

# Correct test split
split_index = int(0.8 * len(x_test))  # 80% for devices, 20% for global test
x_test_devices, y_test_devices = x_test[:split_index], y_test[:split_index]
x_test_global, y_test_global = x_test[split_index:], y_test[split_index:]

# Training data
x_train_devices, y_train_devices = x_train, y_train

# Split training data among devices
num_devices = len(devices)
split_size = len(x_train_devices) // num_devices

for i, device in enumerate(devices):
    start = i * split_size
    end = (i + 1) * split_size if i < num_devices - 1 else len(x_train_devices)
    device.data = (x_train_devices[start:end], y_train_devices[start:end])

# Split test data among devices
split_size = len(x_test_devices) // num_devices

for i, device in enumerate(devices):
    start = i * split_size
    end = (i + 1) * split_size if i < num_devices - 1 else len(x_test_devices)
    device.test_data = (x_test_devices[start:end], y_test_devices[start:end])

# --- Checking results ---
print("\nGlobal test set:")
print(f"x_test_global shape: {x_test_global.shape}")
print(f"y_test_global shape: {y_test_global.shape}")
print()

for device in devices:
    x_data, y_data = device.data
    x_test_data, y_test_data = device.test_data
    print(f"Device {device.device_id}:")
    print(f"  Training data shape: {x_data.shape}, {y_data.shape}")
    print(f"  Test data shape: {x_test_data.shape}, {y_test_data.shape}")
    print(f"  First training sample: {x_data[0]}, label: {y_data[0]}")
    print(f"  First test sample: {x_test_data[0]}, label: {y_test_data[0]}")
    print("-" * 40)
