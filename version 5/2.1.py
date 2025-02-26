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
import os
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.callback import Callback
# from pymoo.operators.sampling.rnd import FloatRandomSampling
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.selection.tournament import TournamentSelection


# Define the multi-objective problem
class FLProblem(Problem):
    def __init__(self, n_devices):
        super().__init__(n_var=n_devices, n_obj=3, n_constr=0, xl=0, xu=1, type_var=np.bool_)
    
    def _evaluate(self, x, out, *args, **kwargs):
        # Generate three random objectives between 0 and 1 for each solution
        out["F"] = np.random.rand(x.shape[0], 3)


# Callback to save each generation’s details
class SaveGenerationCallback(Callback):
    def __init__(self, save_path="version 5/data/"):
        super().__init__()
        self.save_path = save_path

    def notify(self, algorithm):
        gen_number = algorithm.n_gen  # Current generation number
        filename = os.path.join(self.save_path, f"gen{gen_number}.txt")
        
        with open(filename, "w") as f:
            for sol in algorithm.pop:
                # Assuming device_id is in the decision variables (X) at a specific index, e.g., index 0
                device_id = sol.X[0]  # Replace with the correct index if needed
                f.write(f"Device ID: {device_id}\n")
                f.write(f"n_eval: {algorithm.evaluator.n_eval}\n")
                f.write(f"n_nds: {algorithm.opt.shape[0]}\n")
                # f.write(f"eps: {algorithm.evaluator.eps}\n")
                # Check if the algorithm has an optimal solution
                if hasattr(algorithm, "opt") and algorithm.opt is not None:
                    f.write(f"Best Solution Fitness: {algorithm.opt.get('F')}\n")  # Get fitness values
                else:
                    f.write("No optimal solution found.\n")
                f.write(f"indicator: {sol.F}\n\n")





# Number of devices and population size
N_DEVICES = 10
POP_SIZE = 20
NUM_GENERATIONS = 50

# Create problem instance
problem = FLProblem(n_devices=N_DEVICES)



# # Ensure population consists of numeric values
# pop_size = np.array(POP_SIZE, dtype=float)  # Convert to float


# Define NSGA-II algorithm
algorithm = NSGA2(
    pop_size=POP_SIZE,
    # sampling=FloatRandomSampling("bin_random"),
    # sampling=Sampling(),
    sampling=FloatRandomSampling(),
    # crossover = Crossover("bin_two_point", n_offsprings=2),
    # crossover = Crossover(n_parents=2, n_offsprings=2),  # Set both n_parents and n_offsprings
    # mutation=Mutation("bin_bitflip"),
    eliminate_duplicates=True
)

# Define termination condition (generations)
termination = DefaultMultiObjectiveTermination(n_max_gen=NUM_GENERATIONS)


# Run optimization
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    callback=SaveGenerationCallback(),
    verbose=True
)
