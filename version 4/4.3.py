import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

# Load dataset from CSV
csv_file = 'version 4/data/pareto_front_results.csv'
df = pd.read_csv(csv_file)
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
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), 
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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

# Select balanced test set for global model
_, global_test_x, _, global_test_y = train_test_split(x_test, y_test, stratify=y_test, test_size=1000, random_state=42)

# Split MNIST across devices
num_devices = len(devices)
batch_size = len(x_train) // num_devices
for i, device in enumerate(devices):
    start = i * batch_size
    end = (i + 1) * batch_size
    device.data = (x_train[start:end], y_train[start:end])

# Function to average model weights
def average_weights(devices):
    total_data_points = sum(len(device.data[0]) for device in devices)
    averaged_weights = []

    base_weights = devices[0].model.get_weights()
    for i in range(len(base_weights)):
        weighted_sum = sum((len(device.data[0]) / total_data_points) * device.model.get_weights()[i] for device in devices)
        averaged_weights.append(weighted_sum)

    return averaged_weights

# Federated Training Loop
num_rounds = 2  # Number of times to distribute weights and retrain: 3      (6 => 80%)(2 => 50%)
top_n = 2  # Number of top Pareto front devices to train: 10  set to max for all devices
default_epoch = 10 # epoch = 10

for round_num in range(num_rounds):
    print(f"\n=== Federated Learning Round {round_num + 1} ===")
    
    # Assign global model weights to all devices
    global_weights = global_device.model.get_weights()
    for device in devices:
        device.model.set_weights(global_weights)
    
    # Train Top Pareto Front Devices
    trained_devices = []
    for device in devices:
        if device.pareto_rank < top_n:
            x_subset, y_subset = device.data
            device.model.fit(x_subset, y_subset, epochs=default_epoch, verbose=1) # epoch = 10
            trained_devices.append(device)
    
    # Aggregate new global weights
    aggregated_weights = average_weights(trained_devices)
    global_device.model.set_weights(aggregated_weights)

    # Evaluate Global Model
    global_loss, global_accuracy = global_device.model.evaluate(global_test_x, global_test_y, verbose=1)
    print(f"Global Model - Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.4f}")
