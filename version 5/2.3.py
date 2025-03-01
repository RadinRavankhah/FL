import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination.default import DefaultMultiObjectiveTermination

# Global iteration counter
current_learning_iteration = 0

# Device Class representing edge devices in federated learning
class Device:
    def __init__(self, device_id, ram, storage, cpu, bandwidth, battery, charging):
        self.device_id = device_id
        self.ram = ram
        self.storage = storage
        self.cpu = cpu
        self.bandwidth = bandwidth
        self.battery = battery
        self.charging = charging
        self.energy_consumption = ram + storage + cpu + bandwidth
        self.model = self.create_model()
        self.data = None  # Placeholder for dataset partition
        self.number_of_times_fitted = 0

    def create_model(self):
        """Creates a CNN model for training on MNIST."""
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

# Load device data from CSV
df = pd.read_csv('version 5/data/devices.csv')
df.columns = df.columns.str.strip().str.lower()

# Initialize devices
devices = [Device(row['id'], row['ram'], row['storage'], row['cpu'], row['bandwidth'], row['battery'], row.get('charging', 0))
           for _, row in df.iterrows()]

# Load and preprocess MNIST dataset
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)

# Shuffle dataset
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train, y_train = x_train[indices], y_train[indices]

# Split dataset into training (80%) and testing (20%)
split_index = int(0.8 * len(x_train))
x_train_devices, y_train_devices = x_train[:split_index], y_train[:split_index]
x_test_global, y_test_global = x_train[split_index:], y_train[split_index:]

# Distribute training data among devices
split_size = len(x_train_devices) // len(devices)
for i, device in enumerate(devices):
    start, end = i * split_size, (i + 1) * split_size if i < len(devices) - 1 else len(x_train_devices)
    device.data = (x_train_devices[start:end], y_train_devices[start:end])

# Read bitstring from file to determine which devices train
with open('version 5/data/bitstring.txt', 'r') as f:
    bitstring = [int(bit) for bit in f.read().split(',')]

# Train models on selected devices
current_learning_iteration += 1
for device in devices:
    if bitstring[int(device.device_id)] == 1:
        device.model.fit(device.data[0], device.data[1], epochs=2, verbose=1)
        device.number_of_times_fitted += 1

# Define global model
def create_global_model():
    """Creates the global model for federated learning."""
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

global_model = create_global_model()

# Aggregate model weights from devices
def aggregate_weights(global_model, devices):
    """Averages weights from all participating devices and updates the global model."""
    if not devices:
        print("No devices available for aggregation.")
        return
    avg_weights = [np.mean(np.array(layer_weights), axis=0) for layer_weights in zip(*[d.model.get_weights() for d in devices])]
    global_model.set_weights(avg_weights)

# Perform aggregation
aggregate_weights(global_model, devices)

test_loss, test_acc = global_model.evaluate(x_test_global, y_test_global)
print(f"Global Model Accuracy: {test_acc:.4f}")

# Define federated learning optimization problem
from pymoo.core.problem import Problem

class FederatedLearningProblem(Problem):
    def __init__(self, num_devices, devices, global_model, x_test_global, y_test_global):
        super().__init__(n_var=num_devices, n_obj=3, xl=np.zeros(num_devices), xu=np.ones(num_devices), type_var=np.bool_)
        self.devices = devices
        self.global_model = global_model
        self.x_test_global = x_test_global
        self.y_test_global = y_test_global
        self.initial_global_weights = global_model.get_weights()

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X), 3))
        for i, bitstring in enumerate(X):
            self.global_model.set_weights(self.initial_global_weights)
            selected_devices = [d for d, bit in zip(self.devices, bitstring) if bit == 1]
            for device in selected_devices:
                device.model.fit(device.data[0], device.data[1], epochs=1, verbose=0)
            aggregate_weights(self.global_model, selected_devices)
            for device in self.devices:
                device.model.set_weights(self.global_model.get_weights())
            F[i, 0] = -sum(d.ram + d.storage + d.cpu + d.bandwidth + d.battery + d.charging for d in selected_devices)
            local_accuracies = [device.model.evaluate(device.data[0], device.data[1], verbose=0)[1] for device in self.devices]
            F[i, 1] = -sum(1 - local_accuracies[j] for j, bit in enumerate(bitstring) if bit == 1)
            _, global_accuracy = self.global_model.evaluate(self.x_test_global, self.y_test_global, verbose=0)
            F[i, 2] = 1 - global_accuracy
        out["F"] = F

problem = FederatedLearningProblem(len(devices), devices, global_model, x_test_global, y_test_global)

algorithm = NSGA2(pop_size=10, sampling=BinaryRandomSampling(), crossover=TwoPointCrossover(), mutation=BitflipMutation(), eliminate_duplicates=True)
res = minimize(problem=problem, algorithm=algorithm, termination=DefaultMultiObjectiveTermination(n_max_gen=10), seed=42, verbose=True)
