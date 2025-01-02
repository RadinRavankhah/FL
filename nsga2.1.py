import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.datasets import mnist
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# Device class with a "qualities" list
class Device:
    def __init__(self, qualities=[], id=0):
        self.id = id
        self.model = Sequential()
        self.training_images = []
        self.training_labels = []
        self.qualities = qualities  # List of qualities between 0 and 1

    def train(self):
        training_images_nd = np.array(self.training_images)
        training_labels_nd = np.array(self.training_labels)
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(training_images_nd, training_labels_nd, epochs=10, batch_size=8)


# Function to partition data among devices
def partition(number_of_devices, list_of_image_labels):
    devices = []
    for i in range(number_of_devices):
        qualities = [random.random() for _ in range(3)]  # Generate random qualities
        device = Device(qualities=qualities, id=i)
        devices.append(device)

    while len(list_of_image_labels) > 0:
        for device in devices:
            if len(list_of_image_labels) > 1:
                image_label = list_of_image_labels.pop(random.randint(0, len(list_of_image_labels) - 1))
            else:
                image_label = list_of_image_labels.pop(0)
            device.training_images.append(image_label[0])
            device.training_labels.append(image_label[1])
    return devices


# Define the multi-objective problem for NSGA-II
class NodeSelectionProblem(Problem):
    def __init__(self, devices):
        self.devices = devices
        n_var = len(devices)  # Number of devices (binary decision for each)
        n_obj = len(devices[0].qualities)  # Number of qualities as objectives
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0, xu=1, type_var=np.bool_)

    def _evaluate(self, X, out, *args, **kwargs):
        # X is the population (rows = individuals, columns = binary decisions)
        F = []
        for individual in X:
            objectives = np.zeros(len(self.devices[0].qualities))  # Initialize objectives
            for i, decision in enumerate(individual):
                if decision:  # If the device is selected
                    objectives += np.array(self.devices[i].qualities)
            F.append(-objectives)  # Minimize the negative of qualities (maximize qualities)
        out["F"] = np.array(F)


# Run NSGA-II to select devices
def nsga2_node_selection(devices, population_size=50, generations=100):
    problem = NodeSelectionProblem(devices)
    algorithm = NSGA2(pop_size=population_size)
    termination = get_termination("n_gen", generations)
    result = minimize(problem, algorithm, termination, seed=1, verbose=True)
    # Return the best binary selection (first Pareto front solution)
    binary_selection = (result.X[np.argmin(result.F[:, 0])] > 0.5).astype(int)
    return binary_selection


def train_all_devices_return_averaged_weights(devices, binary_node_selection_list):
    print(f"Binary Node Selection List: {binary_node_selection_list}")
    for i, device in enumerate(devices):
        if binary_node_selection_list[i]:
            print(f"Device number {i+1} is now training...")
            device.train()
        else:
            print(f"Device number {i+1} is not selected for training...")

    list_of_weights = [device.model.get_weights() for i, device in enumerate(devices) if binary_node_selection_list[i]]
    averaged_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*list_of_weights)]
    return averaged_weights


def make_main_model_and_test_it(averaged_weights, x_test, y_test):
    main_model = Sequential()
    main_model.add(Flatten(input_shape=(28, 28)))
    main_model.add(Dense(128, activation='relu'))
    main_model.add(Dense(10, activation='softmax'))
    main_model.set_weights(averaged_weights)
    main_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    test_loss, test_accuracy = main_model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


# Federated learning process with NSGA-II for node selection
def federated_learning_with_nsga2(n=0, population_size=50, generations=100):
    x_train, y_train, x_test, y_test = load_mnist_data()
    train_image_label_list = [[x_train[i], y_train[i]] for i in range(len(x_train))]
    devices = partition(n, train_image_label_list)

    # Run NSGA-II to select nodes
    binary_node_selection_list = nsga2_node_selection(devices, population_size, generations)
    print(f"Selected Nodes: {binary_node_selection_list}")

    # Train and test with the selected nodes
    averaged_weights = train_all_devices_return_averaged_weights(devices, binary_node_selection_list)
    make_main_model_and_test_it(averaged_weights, x_test, y_test)


if __name__ == "__main__":
    federated_learning_with_nsga2(n=6, population_size=20, generations=50)
