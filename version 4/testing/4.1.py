import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Load the dataset from CSV
csv_file = 'version 4/data/pareto_front_results.csv'
df = pd.read_csv(csv_file)

# Ensure column names are correctly formatted
df.columns = df.columns.str.strip().str.lower()

# Device Class
class Device:
    def __init__(self, device_id, ram, storage, cpu, bandwidth, battery_status, pareto_rank):
        self.device_id = device_id
        self.ram = ram
        self.storage = storage
        self.cpu = cpu
        self.bandwidth = bandwidth
        self.battery_status = battery_status
        self.energy_consumption = ram + storage + cpu + bandwidth
        self.pareto_rank = pareto_rank
        self.model = self.create_model()
        self.data = None  # Placeholder for dataset partition

    def create_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

# Convert CSV rows into device objects
devices = []
for _, row in df.iterrows():
    device = Device(
        row['id'], row['ram'], row['storage'], row['cpu'], row['bandwidth'],
        row.get('charging', 0), row.get('pareto_rank', float('inf'))
    )
    devices.append(device)

# Create Global Device
global_device = Device('global', 0, 0, 0, 0, 0, 0)

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Fixed test dataset for global device
global_test_data = (x_test[:1000], y_test[:1000])

# Split MNIST across devices
num_devices = len(devices)
batch_size = len(x_train) // num_devices
for i, device in enumerate(devices):
    start = i * batch_size
    end = (i + 1) * batch_size
    device.data = (x_train[start:end], y_train[start:end])

# Train Top N Pareto Fronts e.g 3
top_n = 10
trained_devices = []
for device in devices:
    if device.pareto_rank < top_n:
        x_subset, y_subset = device.data
        device.model.fit(x_subset, y_subset, epochs=3, verbose=1)  # Train for 5 epochs
        trained_devices.append(device)

# Aggregate Model Weights
def average_weights(devices):
    num_models = len(devices)
    averaged_weights = []
    
    # Extract weights from the first model as a base
    base_weights = devices[0].model.get_weights()
    for i in range(len(base_weights)):
        avg_layer_weights = np.mean([device.model.get_weights()[i] for device in devices], axis=0)
        averaged_weights.append(avg_layer_weights)
    
    return averaged_weights

# Compute averaged weights
aggregated_weights = average_weights(trained_devices)

# Assign averaged weights to the global model
global_device.model.set_weights(aggregated_weights)

# Evaluate Global Model
global_test_x, global_test_y = global_test_data
global_loss, global_accuracy = global_device.model.evaluate(global_test_x, global_test_y, verbose=1)
print(f"Global Model - Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.4f}")
