import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Define a simple Device class
class Device:
    def __init__(self, id, data, qualities):
        self.id = id
        self.data = data  # This will store image-label pairs
        self.qualities = qualities
        self.training_images = []  # Initialize training_images list
        self.training_labels = []  # Initialize training_labels list

# Load MNIST data
def load_mnist_data():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test

# Partition the data among devices
def partition(number_of_devices, list_of_image_labels):
    devices = []
    for i in range(number_of_devices):
        qualities = [random.random() for _ in range(3)]  # Generate random qualities
        device = Device(id=i, data=[], qualities=qualities)  # Initialize with empty data
        devices.append(device)
    
    # Split the data manually into approximately equal parts
    chunk_size = len(list_of_image_labels) // number_of_devices
    for i, device in enumerate(devices):
        start_index = i * chunk_size
        if i == number_of_devices - 1:
            # Last device gets all remaining data
            device_data = list_of_image_labels[start_index:]
        else:
            device_data = list_of_image_labels[start_index:start_index + chunk_size]
        
        for image_label in device_data:
            device.training_images.append(image_label[0])
            device.training_labels.append(image_label[1])
    
    return devices

# Node selection problem for NSGA-II
class NodeSelectionProblem(Problem):
    def __init__(self, devices, global_model):
        super().__init__(n_var=len(devices), n_obj=3, n_constr=0, xl=0, xu=1, type_var=np.bool_)
        self.devices = devices
        self.global_model = global_model

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = []  # Computational cost
        f2 = []  # Data quality
        f3 = []  # Model accuracy

        for individual in x:
            selected_devices = [self.devices[i] for i in range(len(individual)) if individual[i]]
            computational_cost = sum([device.qualities[0] for device in selected_devices])
            data_quality = sum([device.qualities[1] for device in selected_devices])
            model_accuracy = sum([device.qualities[2] for device in selected_devices])

            f1.append(computational_cost)
            f2.append(-data_quality)  # Minimize negative data quality for maximization
            f3.append(-model_accuracy)  # Minimize negative model accuracy for maximization

        out["F"] = np.column_stack([f1, f2, f3])

# NSGA-II for device selection
def nsga2_node_selection(devices, global_model, population_size=50, generations=100):
    problem = NodeSelectionProblem(devices, global_model)
    algorithm = NSGA2(pop_size=population_size)
    termination = get_termination("n_gen", generations)
    result = minimize(problem, algorithm, termination, seed=random.randint(0, 10000), verbose=False)
    binary_selection = (result.X[np.argmin(result.F[:, 0])] > 0.5).astype(int)
    return binary_selection

# Train all devices and return averaged weights
def train_all_devices_return_averaged_weights(devices, binary_node_selection_list):
    selected_devices = [devices[i] for i in range(len(binary_node_selection_list)) if binary_node_selection_list[i]]
    all_weights = []

    for device in selected_devices:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # One-hot encode labels
        x_train = np.array([data[0] for data in device.data])
        y_train = np.array([data[1] for data in device.data])
        y_train = to_categorical(y_train, 10)  # Convert labels to one-hot encoding
        
        model.fit(x_train, y_train, epochs=1, verbose=0)

        all_weights.append(model.get_weights())

    averaged_weights = [np.mean([weights[layer] for weights in all_weights], axis=0) for layer in range(len(all_weights[0]))]
    return averaged_weights

# Test the final global model
def make_main_model_and_test_it(global_weights, x_test, y_test):
    global_model = Sequential()
    global_model.add(Flatten(input_shape=(28, 28)))
    global_model.add(Dense(128, activation='relu'))
    global_model.add(Dense(10, activation='softmax'))
    global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    global_model.set_weights(global_weights)

    loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Global Model Accuracy: {accuracy:.4f}")

# Main function for federated learning
def federated_learning_with_nsga2(n=100, population_size=50, generations=100, iterations=5):
    x_train, y_train, x_test, y_test = load_mnist_data()
    train_image_label_list = [[x_train[i], y_train[i]] for i in range(len(x_train))]
    devices = partition(n, train_image_label_list)

    global_model = Sequential()
    global_model.add(Flatten(input_shape=(28, 28)))
    global_model.add(Dense(128, activation='relu'))
    global_model.add(Dense(10, activation='softmax'))
    global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    for iteration in range(iterations):
        print(f"\n### Iteration {iteration + 1} ###")
        
        # Add slight randomness to device qualities
        for device in devices:
            device.qualities = [q + random.uniform(-0.05, 0.05) for q in device.qualities]
        
        # Select devices using NSGA-II
        binary_node_selection_list = nsga2_node_selection(devices, global_model, population_size, generations)
        print(f"Devices selected: {''.join(map(str, binary_node_selection_list))}")

        # Train selected devices and update global model
        averaged_weights = train_all_devices_return_averaged_weights(devices, binary_node_selection_list)
        global_model.set_weights(averaged_weights)

    # Test the final global model
    make_main_model_and_test_it(global_model.get_weights(), x_test, y_test)

# Run the federated learning process
if __name__ == "__main__":
    federated_learning_with_nsga2()
