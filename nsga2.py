import random 
from main import partition, federated_learning, load_mnist_data, make_image_label_list

# Generate a list of 10^6 random floating point numbers between 0 and 1
NUMBER_OF_DEVICES = 6
x_train, y_train, x_test, y_test = load_mnist_data()


devices = partition(NUMBER_OF_DEVICES, make_image_label_list(x_train,y_train))

for i in range(NUMBER_OF_DEVICES):
    devices[i].id = i+1
    qualities = [random.randint(0, 10**5)/ 10**5 for _ in range(5)]
    devices[i].qualities = qualities

# bnsl = [random.randint(0,1) for _ in range(NUMBER_OF_DEVICES)]
bnsl = [1 for _ in range(NUMBER_OF_DEVICES)]
print(bnsl)
federated_learning(devices=devices, binary_node_selection_list=bnsl)
