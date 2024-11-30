import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Check shapes
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # Magic number and dimensions
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        # Magic number and number of items
        magic, num_labels = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def make_image_label_list(x_list, y_list):  # x_list and y_list should have the same length
    return [[x_list[i],y_list[i]] for i in range(len(x_list))]

class Device:
    def __init__(self):
        # Initialization code
        self.model = Sequential()
        self.training_images = []
        self.training_labels = []

    def train(self):
        # Device:
        # model = sequential()
        # training_images = []  # appends images here
        # then turns into ndarray
        # training_labels = []  # appends labels here
        # then turns into ndarray
        
        training_images_nd = np.array(self.training_images)
        training_labels_nd = np.array(self.training_labels)
        # Flatten the 28x28 images into a 784-dimensional vector
        self.model.add(Flatten(input_shape=(28, 28)))

        # Add a fully connected hidden layer with 128 neurons
        self.model.add(Dense(128, activation='relu'))

        # Add the output layer with 10 neurons (one for each digit 0-9)
        self.model.add(Dense(10, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(training_images_nd, training_labels_nd, epochs=10, batch_size=8) # defaults epoch = 10, batch_size = 32
        # for epoch = 10:
        # batch_size = 32 => accuracy = 57
        # batch_size = 16 => accuracy = 62
        # batch_size = 8 => accuracy = 67 to 74

def partition(number_of_devices, list_of_image_labels):
    devices = []
    for i in range(number_of_devices):
        device = Device()
        devices.append(device)

    # counter = 0  was just for testing
    while len(list_of_image_labels) > 0:
        for device in devices:
            if len(list_of_image_labels) > 1:
                image_label = list_of_image_labels.pop(random.randint(0,len(list_of_image_labels)-1))
            else:
                image_label = list_of_image_labels.pop(0)
            device.training_images.append(image_label[0])
            device.training_labels.append(image_label[1])
        counter +=1
        # print(counter)  was just for testing
    
    return devices

def train_all_devices_return_averaged_weights(devices):
    counter = 1
    for device in devices:
        print(f"Device number {counter} is now training...")
        counter += 1
        device.train()

    list_of_weights = []
    for device in devices:
        list_of_weights.append(device.model.get_weights())

    # Compute the average weights across all models
    averaged_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*list_of_weights)]
    return averaged_weights

def make_main_model_and_test_it(averaged_weights):

    main_model = Sequential()

    # Flatten the 28x28 images into a 784-dimensional vector
    main_model.add(Flatten(input_shape=(28, 28)))

    # Add a fully connected hidden layer with 128 neurons
    main_model.add(Dense(128, activation='relu'))

    # Add the output layer with 10 neurons (one for each digit 0-9)
    main_model.add(Dense(10, activation='softmax'))

    # Assign the averaged weights to the new model
    main_model.set_weights(averaged_weights)

    # Verify the averaged weights
    for i, layer_weights in enumerate(main_model.get_weights()):
        print(f"Layer {i // 2 + 1} {'weights' if i % 2 == 0 else 'biases'} shape: {layer_weights.shape}")

    # Compile the averaged model (necessary after setting weights)
    main_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Now, test the averaged model on the test data (x_test, y_test)
    # Make sure x_test and y_test are prepared and preprocessed properly

    # Evaluate the model on the test data
    test_loss, test_accuracy = main_model.evaluate(x_test, y_test)

    # Print the results
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")



# training and testing data
x_train # (train_images)
y_train # (train_labels)
x_test # (test_images)
y_test # (test_labels)

print(f"Training images shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test images shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Normalize the pixel values to [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


print(len(x_train))
print(len(y_train))
print(type(x_train))

remaining_indexes = make_image_label_list(x_train,y_train)

devices = partition(6, remaining_indexes)

averaged_weights = train_all_devices_return_averaged_weights(devices)

make_main_model_and_test_it(averaged_weights)