import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0,1]

# Define a simple Sequential model for each device
def create_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the Device class with a model
class Device:
    def __init__(self, qualities, train_data, train_labels):
        self.qualities = np.array(qualities, dtype=object)
        self.qualities[0] = int(self.qualities[0])
        self.model = create_model()
        self.train_data = train_data
        self.train_labels = train_labels
    
    def train(self, epochs=1):
        self.model.fit(self.train_data, self.train_labels, epochs=epochs, verbose=0)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

# Federated Averaging Function
def federated_averaging(devices):
    num_devices = len(devices)
    avg_weights = [np.zeros_like(w) for w in devices[0].get_weights()]
    
    for device in devices:
        weights = device.get_weights()
        for i in range(len(weights)):
            avg_weights[i] += weights[i] / num_devices
    
    return avg_weights

# Define a custom problem class
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
    return -(X[:, 0] - 1) / (64 - 1)

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

# Create devices and assign MNIST data
np.random.seed(42)
devices = []
data_per_device = len(x_train) // 100  # Split data
for i in range(100):
    start_idx = i * data_per_device
    end_idx = (i + 1) * data_per_device
    qualities = [np.random.randint(1, 65)] + list(np.random.rand(5))
    device = Device(qualities, x_train[start_idx:end_idx], y_train[start_idx:end_idx])
    devices.append(device)

# Train devices and perform federated averaging for multiple rounds
for round in range(5):
    print(f"\nFederated Learning Round {round+1}\n")
    for device in devices:
        device.train(epochs=1)
    avg_weights = federated_averaging(devices)
    for device in devices:
        device.set_weights(avg_weights)

# Create the problem instance
problem = FederatedLearningProblem(devices, [objective1, objective2, objective3, objective4, objective5, objective6])

# Configure NSGA-II
algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# Solve the problem
res = minimize(
    problem,
    algorithm,
    ('n_gen', 50),
    seed=42,
    verbose=False
)

# Extract Pareto-optimal solutions
pareto_indices = NonDominatedSorting().do(res.F, only_non_dominated_front=True)
pareto_front = res.F[pareto_indices]

# Plot Pareto front
for i in range(problem.n_obj):
    for j in range(i + 1, problem.n_obj):
        plt.figure(figsize=(8, 6))
        plt.scatter(res.F[:, i], res.F[:, j], color="red", alpha=0.5, label="All Solutions")
        plt.scatter(pareto_front[:, i], pareto_front[:, j], color="blue", s=30, label="Pareto Front")
        plt.xlabel(f"Objective {i + 1}")
        plt.ylabel(f"Objective {j + 1}")
        plt.legend(loc="best")
        plt.title(f"Objective {i + 1} vs Objective {j + 1}")
        plt.show()

# Print Pareto-optimal devices
print("Pareto-optimal devices (indices):", pareto_indices)
for idx in pareto_indices:
    qualities = devices[idx].qualities
    formatted_qualities = [qualities[0]] + [f"{q:.4f}" for q in qualities[1:]]
    print(f"Device {idx + 1}: Qualities = {formatted_qualities}")
