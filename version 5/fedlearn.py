import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

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
        self.last_round_participated = 0
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
        device.model.fit(device.data[0], device.data[1], epochs=2, verbose=1)
        device.number_of_times_fitted += 1
        device.last_round_participated = current_learning_iteration


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



import numpy as np

def aggregate_weights(global_model, devices):
    """Computes the weighted average of model weights from all devices and updates the global model."""

    num_devices = len(devices)
    if num_devices == 0:
        print("No devices available for aggregation.")
        return

    # Get device weights and participation ratios
    device_weights = [device.model.get_weights() for device in devices]
    device_participation_ratio = np.array(
        [device.last_round_participated / current_learning_iteration for device in devices]
    )

    # Total weight for normalization
    total_weight = np.sum(device_participation_ratio)
    if total_weight == 0:
        print("Total weight is zero, cannot perform aggregation.")
        return

    # Compute weighted sum of weights
    weighted_sums = [
        np.sum(np.stack([device_weights[i][layer] * device_participation_ratio[i] for i in range(num_devices)]), axis=0)
        for layer in range(len(device_weights[0]))
    ]

    # Compute weighted average
    weighted_avg_weights = [layer_sum / total_weight for layer_sum in weighted_sums]

    # Set the global model's weights to the weighted averaged weights
    global_model.set_weights(weighted_avg_weights)


# Call this function after training the local models:
aggregate_weights(global_model, devices)


test_loss, test_acc = global_model.evaluate(x_test_global, y_test_global)
print(f"Global Model Accuracy: {test_acc:.4f}")

########################################################
# # Update device participation based on the bitstring
# selected_devices = [device for device in devices if bitstring[int(device.device_id)] == 1]

# current_learning_iteration += 1
# # Train local models for selected devices
# for device in selected_devices:
#     device.model.fit(device.data[0], device.data[1], epochs=1, verbose=0)
#     device.number_of_times_fitted += 1
#     device.last_round_participated = current_learning_iteration

# # Aggregate weights to update the global model
# aggregate_weights(self.global_model, selected_devices)

# # Distribute the updated global model back to all devices
# for device in devices:
#     device.model.set_weights(self.global_model.get_weights())

# test_loss, test_acc = global_model.evaluate(x_test_global, y_test_global)
# print(f"Global Model Accuracy: {test_acc:.4f}")

#################################################################




import numpy as np
import random
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination

# Parameters
NUM_DEVICES = num_devices   # Number of devices (length of bitstring)
POPULATION_SIZE = 100
NUM_GENERATIONS = 10

# Step 1: Define the Problem
import numpy as np
from pymoo.core.problem import Problem

class FederatedLearningProblem(Problem):
    def __init__(self, num_devices, devices, global_model, x_test_global, y_test_global):
        super().__init__(
            n_var=num_devices,         # Number of variables (bitstring length)
            n_obj=3,                   # Number of objectives
            n_constr=0,                # No constraints
            xl=np.zeros(num_devices),  # Lower bound (0)
            xu=np.ones(num_devices),   # Upper bound (1)
            type_var=np.bool_          # Binary variables (bitstrings)
        )
        self.devices = devices
        self.global_model = global_model
        self.x_test_global = x_test_global
        self.y_test_global = y_test_global

        # Save the initial global model weights
        self.initial_global_weights = global_model.get_weights()

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluates objective values for each solution in the population."""
        num_solutions = len(X)
        F = np.zeros((num_solutions, 3))  # Initialize objective matrix

        for i, bitstring in enumerate(X):
            # Reset the global model to its initial state
            self.global_model.set_weights(self.initial_global_weights)
            # Update device participation based on the bitstring
            selected_devices = [device for device, bit in zip(self.devices, bitstring) if bit == 1]
            # Objective 1: Hardware Objectives (maximize)
            hardware_score = sum(
                device.ram + device.storage + device.cpu + device.bandwidth + device.battery + device.charging
                for device in selected_devices
            )
            F[i, 0] = 6 - hardware_score  # Minimize (negative of hardware score)
            # Objective 2: Fairness (prioritize devices with lowest local accuracy)
            local_accuracies = []
            for device in self.devices:
              # just the devices in this solution:
                if bitstring[int(device.device_id)] == 1:
                  _, accuracy = global_model.evaluate(device.data[0], device.data[1], verbose=0)
                  local_accuracies.append(accuracy)
                else:
                  local_accuracies.append(0)
            # Fairness score: Sum of (1 - accuracy) for selected devices should be minimized
            # This prioritizes devices with lower local accuracy

            fairness_score = 0  # Initialize fairness score

            for j, bit in enumerate(bitstring):
                # print(j)
                # print(bit)
                if bit == 1:  # Check if the device is selected
                    fairness_score += local_accuracies[j]  # Add the corresponding accuracy value
            # Now, fairness_score contains the final sum

            # fairness_score = sum(local_accuracies[j] for j, bit in enumerate(bitstring) if bit == 1)
            F[i, 1] = fairness_score  # Minimize (negative of fairness score)
            # Objective 3: Global Model Accuracy (maximize)
            _, global_accuracy = self.global_model.evaluate(self.x_test_global, self.y_test_global, verbose=0)
            F[i, 2] = 1 - global_accuracy  # Minimize (1 - accuracy)
        out["F"] = F  # Set the objective values

'''
            # Update device participation based on the bitstring
            selected_devices = [device for device, bit in zip(self.devices, bitstring) if bit == 1]

            # Train local models for selected devices
            for device in selected_devices:
                device.model.fit(device.data[0], device.data[1], epochs=1, verbose=0)

            # Aggregate weights to update the global model
            aggregate_weights(self.global_model, selected_devices)

            # Distribute the updated global model back to all devices
            for device in self.devices:
                device.model.set_weights(self.global_model.get_weights())
            '''


problem = FederatedLearningProblem(
    num_devices=NUM_DEVICES,
    devices=devices,
    global_model=global_model,
    x_test_global=x_test_global,
    y_test_global=y_test_global
)


# Step 2: Configure NSGA-II Algorithm
algorithm = NSGA2(
    pop_size=POPULATION_SIZE,
    sampling=BinaryRandomSampling(),      # Random bitstrings
    crossover=TwoPointCrossover(),        # Two-point crossover
    mutation=BitflipMutation(),           # Bit flip mutation
    eliminate_duplicates=True             # Avoid duplicate solutions
)


NUMBER_OF_LEARNING_ITERATIONS = 2


for i in range(NUMBER_OF_LEARNING_ITERATIONS):
    # Step 3: Run Optimization
    res = minimize(
        problem=problem,
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
    for bitstring in pareto_solutions:
        print("".join(map(str, bitstring)))

    bitstring = pareto_solutions[0] # for now!
    print(bitstring)
    # Convert to bitstring (list of 1s and 0s)
    bitstring = [1 if word == "True" else 0 for word in str(bitstring).replace("True", "1 ").replace("False", "0 ").split()]


    ########################################################
    # Update device participation based on the bitstring
    selected_devices = [device for device in devices if bitstring[int(device.device_id)] == 1]

    # Aggregate weights to update the global model
    aggregate_weights(global_model, selected_devices)

    # Distribute the updated global model back to all devices
    for device in devices:
        device.model.set_weights(global_model.get_weights())

    current_learning_iteration += 1
    # Train local models for selected devices
    for device in selected_devices:
        device.model.fit(device.data[0], device.data[1], epochs=1, verbose=0)
        device.number_of_times_fitted += 1
        device.last_round_participated = current_learning_iteration


    test_loss, test_acc = global_model.evaluate(x_test_global, y_test_global)
    print(f"Global Model Accuracy: {test_acc:.4f}")

    ################################################################
