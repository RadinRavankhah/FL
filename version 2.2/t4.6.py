import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Define the Device class with a Sequential model
class Device:
    def __init__(self, qualities):
        self.qualities = np.array(qualities, dtype=object)
        self.qualities[0] = int(self.qualities[0])  # Ensure the first quality is an integer
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def train(self, x_train, y_train, epochs=1):
        self.model.fit(x_train, y_train, epochs=epochs, verbose=0)  # Silent training

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

# Define the federated learning problem
class FederatedLearningProblem(Problem):
    def __init__(self, devices, objective_functions):
        self.devices = devices
        self.objective_functions = objective_functions
        super().__init__(
            n_var=len(devices[0].qualities),
            n_obj=len(objective_functions),
            n_constr=0,
            xl=np.array([1] + [0] * (len(devices[0].qualities) - 1)),
            xu=np.array([64] + [1] * (len(devices[0].qualities) - 1))
        )

    def _evaluate(self, X, out, *args, **kwargs):
        objectives = [func(X) for func in self.objective_functions]
        out["F"] = np.column_stack(objectives)

# Objective functions

def objective1(X):
    return -(X[:, 0] - 1) / 63  # Normalize first quality to [0,1]

def objective2(X):
    return X[:, 1] ** 2

def objective3(X):
    return np.sin(X[:, 2])

def objective4(X):
    return X[:, 3]

def objective5(X):
    return X[:, 4] + X[:, 5]

def objective6(X):
    return -np.sum(X, axis=1)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Create devices
np.random.seed(42)
devices = [Device([np.random.randint(1, 65)] + list(np.random.rand(5))) for _ in range(100)]

# Federated learning simulation
num_rounds = 10  # Number of global training rounds
selected_device_count = 10  # Devices chosen per round

for round_num in range(num_rounds):
    print(f"\n🟢 Round {round_num + 1} of Federated Learning 🟢")
    
    # Optimize device selection
    problem = FederatedLearningProblem(devices, [objective1, objective2, objective3, objective4, objective5, objective6])
    algorithm = NSGA2(pop_size=100, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
    res = minimize(problem, algorithm, ('n_gen', 50), seed=42, verbose=False)

    # Get Pareto-optimal devices
    pareto_indices = NonDominatedSorting().do(res.F, only_non_dominated_front=True)
    selected_devices = [devices[i] for i in pareto_indices[:selected_device_count]]
    non_selected_devices = [devices[i] for i in range(len(devices)) if i not in pareto_indices[:selected_device_count]]
    
    print(f"Selected Devices: {[i+1 for i in pareto_indices[:selected_device_count]]}")
    print(f"Non-Selected Devices: {[i+1 for i in range(len(devices)) if i not in pareto_indices[:selected_device_count]]}")

    # Train selected devices on local data partitions
    for i, device in enumerate(selected_devices):
        start = i * (len(x_train) // selected_device_count)
        end = (i + 1) * (len(x_train) // selected_device_count)
        device.train(x_train[start:end], y_train[start:end], epochs=1)
        print(f"Device {pareto_indices[i] + 1} trained on its local dataset partition.")
    
    # Aggregate model weights
    # averaged_weights = np.mean([device.get_weights() for device in selected_devices], axis=0)
    weights_list = [device.get_weights() for device in selected_devices]
    averaged_weights = []
    for layer_weights in zip(*weights_list):
        averaged_weights.append(np.mean(layer_weights, axis=0))
    
    # Update all devices with averaged weights
    for device in devices:
        device.set_weights(averaged_weights)
    
    print("✅ Weights averaged and distributed to all devices.")

# Evaluate final global model
main_model = Device([0]*6)  # Create a dummy device for final evaluation
main_model.set_weights(averaged_weights)
loss, accuracy = main_model.model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Global Model Accuracy: {accuracy:.4f}")
