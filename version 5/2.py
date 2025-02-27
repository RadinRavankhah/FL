import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
# from sklearn.model_selection import train_test_split


current_learning_iteration = 0

# Device Class
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
    

# Load dataset from CSV
csv_file = 'version 5/data/devices.csv'
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip().str.lower()


# Convert CSV rows into device objects
devices = []
for _, row in df.iterrows():
    device = Device(
        row['id'], row['ram'], row['storage'], row['cpu'], row['bandwidth'], row['battery'],
        row.get('charging', 0)
    )
    devices.append(device)






# Load MNIST dataset
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

# Normalize data and reshape for CNN
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # Add channel dimension

# Shuffle data
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train, y_train = x_train[indices], y_train[indices]

# Split into global test set (20%) and training set (80%)
split_index = int(0.8 * len(x_train))
x_train_devices, y_train_devices = x_train[:split_index], y_train[:split_index]
x_test_global, y_test_global = x_train[split_index:], y_train[split_index:]


# Split dataset among devices
num_devices = len(devices)
split_size = len(x_train_devices) // num_devices

for i, device in enumerate(devices):
    start = i * split_size
    end = (i + 1) * split_size if i < num_devices - 1 else len(x_train_devices)
    device.data = (x_train_devices[start:end], y_train_devices[start:end])





with open('version 5/data/bitstring.txt', 'r') as f:
    bitstring = f.read()

bitstring = [int(bit) for bit in bitstring.split(',')]

current_learning_iteration += 1
for device in devices:
    print(int(device.device_id))
    if bitstring[int(device.device_id)] == 1:
        device.model.fit(device.data[0], device.data[1], epochs=1, verbose=1)
        device.number_of_times_fitted += 1


# Global Model
# Define the global model with the same architecture
def create_global_model():
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


def aggregate_weights(global_model, devices):
    """Averages model weights from all devices and updates the global model."""
    
    num_devices = len(devices)
    if num_devices == 0:
        print("No devices available for aggregation.")
        return
    
    # Get the weights of all devices
    device_weights = [device.model.get_weights() for device in devices]
    
    # Compute the average of the weights across all devices
    avg_weights = [np.mean(np.array(layer_weights), axis=0) for layer_weights in zip(*device_weights)]
    
    # Set the global model's weights to the averaged weights
    global_model.set_weights(avg_weights)

# Call this function after training the local models:
aggregate_weights(global_model, devices)


test_loss, test_acc = global_model.evaluate(x_test_global, y_test_global)
print(f"Global Model Accuracy: {test_acc:.4f}")































import numpy as np
import random
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.selection.tournament import TournamentSelection
# from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination

# Parameters
NUM_DEVICES = num_devices   # Number of devices (length of bitstring)
POPULATION_SIZE = 50
NUM_GENERATIONS = 300

# Step 1: Define the Problem
class FederatedLearningProblem(Problem):
    def __init__(self, num_devices):
        super().__init__(
            n_var=num_devices,         # Number of variables (bitstring length)
            n_obj=3,                   # Number of objectives
            n_constr=0,                # No constraints
            xl=np.zeros(num_devices),  # Lower bound (0)
            xu=np.ones(num_devices),   # Upper bound (1)
            type_var=np.bool_          # Binary variables (bitstrings)
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluates objective values for each solution in the population."""
        out["F"] = np.random.rand(len(X), 3)  # Random objectives between 0 and 1

# Step 2: Configure NSGA-II Algorithm
algorithm = NSGA2(
    pop_size=POPULATION_SIZE,
    sampling=BinaryRandomSampling(),      # Random bitstrings
    crossover=TwoPointCrossover(),        # Two-point crossover
    mutation=BitflipMutation(),           # Bit flip mutation
    eliminate_duplicates=True             # Avoid duplicate solutions
)

# Step 3: Run Optimization
res = minimize(
    problem=FederatedLearningProblem(NUM_DEVICES),
    algorithm=algorithm,
    # termination=MultiObjectiveSpaceToleranceTermination(tol=1e-6, n_last=10, nth_gen=5, n_max_gen=NUM_GENERATIONS),
    termination=DefaultMultiObjectiveTermination(n_max_gen=NUM_GENERATIONS),
    seed=42,
    verbose=True
)

# Step 4: Extract the Best Pareto Front
pareto_front = res.F   # Objective values of solutions in Pareto front
pareto_solutions = res.X  # Corresponding bitstrings

# Print the Best Pareto Front Solutions
print("Best Pareto Front (Bitstrings):")
s = ""
for bitstring in pareto_solutions:
    print("".join(map(str, bitstring)).replace("True", "1").replace("False", "0"))
    print(len("".join(map(str, bitstring)).replace("True", "1").replace("False", "0")))
    s += "["
    for character in ("".join(map(str, bitstring)).replace("True", "1").replace("False", "0")):
        s += character + ","
    s = s[:-1]
    s += "]\n"
    print()

with open("version 5/data/best_pareto_front4.txt", 'w') as f:
        f.write(s)