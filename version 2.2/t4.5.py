import numpy as np
import tensorflow as tf
from tensorflow import keras
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class Device:
    def __init__(self, qualities, data):
        self.qualities = np.array(qualities)
        self.model = create_model()
        self.x_train, self.y_train = data

    def train(self, epochs=1):
        print(f"Training Device with qualities {self.qualities}")
        self.model.fit(self.x_train, self.y_train, epochs=epochs, verbose=1)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

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

def objective1(X):
    return -((X[:, 0] - 1) / (64 - 1))

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

np.random.seed(42)
devices = [Device([np.random.randint(1, 65)] + list(np.random.rand(5)), 
                  (x_train[i*600:(i+1)*600], y_train[i*600:(i+1)*600])) for i in range(100)]

problem = FederatedLearningProblem(devices, [objective1, objective2, objective3, objective4, objective5, objective6])

algorithm = NSGA2(
    pop_size=20,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

res = minimize(problem, algorithm, ('n_gen', 10), seed=42, verbose=False)

pareto_indices = NonDominatedSorting().do(res.F, only_non_dominated_front=True)
selected_devices = [devices[idx] for idx in pareto_indices]

def average_weights(models):
    avg_weights = [np.mean([model.get_weights()[i] for model in models], axis=0) for i in range(len(models[0].get_weights()))]
    return avg_weights

for round_num in range(5):
    print(f"\nFederated Learning Round {round_num + 1}")
    for device in selected_devices:
        device.train(epochs=1)
    new_weights = average_weights([device.model for device in selected_devices])
    for device in selected_devices:
        device.set_weights(new_weights)

global_model = create_model()
global_model.set_weights(new_weights)
loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
print(f"Final Global Model Accuracy: {acc:.4f}")
